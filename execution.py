"""Reference staging, consent, and the duration decision (T036, T038).

The duration logic here is the part worth reading carefully. Because H3 generates
audio and video **jointly**, there is no separate speech synthesis stage whose
output could be measured before video is produced. Duration is therefore an
**input** to generation, not a measurement of it.

That has a consequence the earlier three-model design did not have: there is no
pre-generation fit check, and **no request is ever rejected for script length**.
A long script yields a suggestion clamped to what the model supports; if delivery
sounds rushed, the operator raises the duration and regenerates. The only
`duration` error is an operator override outside the profile's supported range.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from domain import (
    ConsentAttestation,
    DurationDecision,
    ModelProfile,
    ReferenceSet,
    SpeakingRate,
)
from errors import ConsentError, DurationError, ReferenceError, ValidationError
from media import (
    FaceDetector,
    check_faces,
    detect_retained_origin,
    inspect_audio,
    inspect_image,
    looks_like_video,
    normalize_image,
    sha256_file,
)
from prompting import assemble_prompt
from validation import validate_guidance, validate_language, validate_seed

# --------------------------------------------------------------------------
# Consent
# --------------------------------------------------------------------------


def build_consent(
    *, request_id: uuid.UUID, audio_sha256: str, confirmed: bool
) -> ConsentAttestation:
    """Create an attestation from a true, current submission — or refuse.

    There is no path that produces a record from `confirmed=False`, which is why
    a stale or absent checkbox cannot become consent later.
    """
    if not confirmed:
        raise ConsentError(
            "voice-cloning consent is required. Confirm that you own this "
            "recording or have permission to clone the voice."
        )
    try:
        return ConsentAttestation(
            request_id=request_id,
            reference_audio_sha256=audio_sha256,
            confirmed=True,
            confirmed_at=datetime.now(UTC),
        )
    except Exception as exc:
        raise ConsentError(f"consent could not be recorded: {exc}") from exc


def verify_consent(
    consent: ConsentAttestation | None, *, request_id: uuid.UUID, audio_sha256: str
) -> ConsentAttestation:
    """Re-check the binding server-side at the point of use."""
    if consent is None:
        raise ConsentError("voice-cloning consent is required.")
    if consent.request_id != request_id:
        raise ConsentError("consent was recorded for a different request. Confirm consent again.")
    if consent.reference_audio_sha256.lower() != audio_sha256.lower():
        raise ConsentError(
            "the reference audio changed after consent was given. Confirm consent "
            "again for the new recording."
        )
    return consent


# --------------------------------------------------------------------------
# Duration
# --------------------------------------------------------------------------


def suggest_duration(
    *, script: str, language: str, profile: ModelProfile
) -> tuple[float, SpeakingRate | None]:
    """Suggest a duration from script length and the profile's speaking rate.

    Advisory. When the profile has no measured rate for the language, fall back
    to its default duration and report that no rate was used, rather than
    inventing one.
    """
    rate = profile.speaking_rate_for(language)
    if rate is None:
        return profile.duration_range_seconds.default_seconds, None

    characters = len((script or "").strip())
    raw_seconds = characters / rate.rate if rate.rate else 0.0
    return profile.duration_range_seconds.clamp(raw_seconds), rate


def decide_duration(
    *,
    script: str,
    language: str,
    profile: ModelProfile,
    requested_seconds: float | None = None,
) -> DurationDecision:
    """Settle the duration handed to the adapter."""
    suggested, rate_used = suggest_duration(script=script, language=language, profile=profile)
    supported = profile.duration_range_seconds
    overrides: list[str] = []

    if requested_seconds is None:
        effective = suggested
        overrode = False
    else:
        if not supported.contains(requested_seconds):
            raise DurationError(
                f"{requested_seconds:g}s is outside this model's supported range of "
                f"{supported.min_seconds:g}-{supported.max_seconds:g}s. "
                "Choose a duration inside the range."
            )
        effective = float(requested_seconds)
        overrode = effective != suggested
        if overrode:
            overrides.append(
                f"operator set the duration to {effective:g}s (suggested {suggested:g}s)"
            )

    return DurationDecision(
        suggested_duration_seconds=suggested,
        speaking_rate_used=rate_used,
        requested_duration_seconds=requested_seconds,
        operator_overrode=overrode,
        effective_duration_seconds=effective,
        effective_num_frames=max(1, round(effective * profile.frame_rate)),
        frame_rate=profile.frame_rate,
        resolution=profile.resolutions[0],
        audio_sample_rate=profile.audio_output.sample_rate,
        overrides=overrides,
        profile_id=profile.profile_id,
    )


# --------------------------------------------------------------------------
# Reference staging
# --------------------------------------------------------------------------


def stage_references(
    *,
    request_id: uuid.UUID,
    image_paths,
    audio_path: Path | str,
    consent: ConsentAttestation,
    profile: ModelProfile,
    staging_root: Path | str,
    detector: FaceDetector,
    transcript: str | None = None,
    outputs_root: Path | str | None = None,
) -> ReferenceSet:
    """Validate references and copy them into request-owned staging.

    References are copied, never used in place: the operator's file may be moved
    or edited mid-run, and a bundle that records a digest must own the bytes that
    digest describes.
    """
    images = [Path(p) for p in (image_paths or [])]
    if not images:
        raise ValidationError("at least one reference image is required")

    limits = profile.reference_limits
    image_limit = limits.accepted.get("image")
    if image_limit is not None and len(images) > image_limit:
        raise ReferenceError(
            f"this model accepts at most {image_limit} reference image(s); "
            f"{len(images)} were supplied."
        )

    for candidate in images:
        if looks_like_video(candidate):
            reason = limits.rejected.get("video", "video references are not accepted")
            raise ReferenceError(f"{candidate.name}: {reason}")

    audio = Path(audio_path)
    if looks_like_video(audio):
        raise ReferenceError(
            f"{audio.name}: {limits.rejected.get('video', 'video references are not accepted')}"
        )

    audio_info = inspect_audio(audio)
    clip_bounds = limits.audio_clip_seconds
    if clip_bounds is not None and not clip_bounds.contains(audio_info.duration_seconds):
        raise ReferenceError(
            f"the reference recording is {audio_info.duration_seconds:.1f}s; this "
            f"model needs between {clip_bounds.min_seconds:g}s and "
            f"{clip_bounds.max_seconds:g}s."
        )

    inputs_dir = Path(staging_root) / str(request_id) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    measured: dict[str, dict] = {}
    source_paths: dict[str, str] = {}
    staged_images: list[str] = []

    for index, source in enumerate(images):
        info = inspect_image(source)
        check_faces(source, detector)

        staged = normalize_image(source, inputs_dir / f"image-{index:03d}.png")
        staged_images.append(str(staged))
        source_paths[str(staged)] = str(source)
        measured[str(staged)] = {
            "kind": "image",
            "width": info.width,
            "height": info.height,
            "format": info.format,
            "sha256": sha256_file(staged),
            "source_sha256": info.sha256,
            "size_bytes": staged.stat().st_size,
        }

    staged_audio = inputs_dir / f"reference-audio{audio.suffix or '.wav'}"
    shutil.copy2(audio, staged_audio)
    staged_digest = sha256_file(staged_audio)

    # Consent binds to the exact bytes we will use, not to what was uploaded.
    verify_consent(consent, request_id=request_id, audio_sha256=staged_digest)

    source_paths[str(staged_audio)] = str(audio)
    measured[str(staged_audio)] = {
        "kind": "audio",
        "duration_seconds": audio_info.duration_seconds,
        "sample_rate": audio_info.sample_rate,
        "channels": audio_info.channels,
        "format": audio_info.format,
        "sha256": staged_digest,
        "size_bytes": staged_audio.stat().st_size,
    }

    origin = detect_retained_origin(audio, outputs_root) if outputs_root else None

    return ReferenceSet(
        request_id=request_id,
        image_paths=staged_images,
        audio_path=str(staged_audio),
        source_paths=source_paths,
        measured=measured,
        speaker_result={"single_speaker": True, "checked": False},
        transcript=transcript,
        consent=consent,
        origin=origin,
        profile_limits={
            "accepted": dict(limits.accepted),
            "rejected": dict(limits.rejected),
        },
        rejected_kinds=sorted(limits.rejected),
    )


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def preflight(
    *,
    request_id: uuid.UUID,
    motion_prompt: str,
    speech_script: str,
    language: str,
    requested_seconds: float | None,
    seed: int | None,
    guidance_scale: float | None,
    profile: ModelProfile,
    tokenizer=None,
):
    """Validate scalar inputs and settle prompt and duration, in a stable order.

    Order is part of the contract. Language is checked before duration so a
    request wrong in both ways always reports the language problem — the one the
    operator can actually act on.
    """
    validate_language(language, profile=profile)
    effective_seed = validate_seed(seed)
    validate_guidance(guidance_scale, profile=profile)

    duration = decide_duration(
        script=speech_script,
        language=language,
        profile=profile,
        requested_seconds=requested_seconds,
    )
    prompt = assemble_prompt(
        motion_prompt=motion_prompt,
        speech_script=speech_script,
        language=language,
        profile=profile,
        tokenizer=tokenizer,
    )
    return prompt, duration, effective_seed
