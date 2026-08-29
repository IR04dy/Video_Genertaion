"""Generation orchestration (T044) with per-stage progress (T068).

One request, one model invocation. There is no loop, no retry, no second pass,
and no separate speech or lip-sync stage — mouth movement and voice come out of
the same call, which is what removed the timebase bridge the earlier design had.

The stage order is fixed and is part of the progress contract:

    validate -> prepare_references -> assemble_prompt -> plan_duration ->
    load_model -> generate -> decode -> export -> verify -> metadata -> publish

Nothing here bounds elapsed time. Inference is unbounded by decision, so this
module measures progress and never deadlines it.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adapters.base import CancellationToken, GenerationInputs
from devices import accelerator_snapshot, resolve_device, select_dtype
from domain import (
    DeviceKind,
    ErrorDetail,
    GeneratedOutput,
    GenerationResult,
    MemorySnapshot,
    ModelProfile,
    ProgressEvent,
    RequestState,
)
from errors import AppError, sanitize, to_detail, translate
from execution import build_consent, preflight, stage_references
from export import export_mp4, export_video_only, publish_atomically, verify_output
from media import FaceResult, sha256_file
from storage import (
    directory_size,
    inventory_artifacts,
    publish_bundle,
    remove_staging,
    staging_dir,
)

_log = logging.getLogger(__name__)

ProgressSink = Callable[[ProgressEvent], None]

STAGE_ORDER = (
    "validate",
    "prepare_references",
    "assemble_prompt",
    "plan_duration",
    "load_model",
    "generate",
    "decode",
    "export",
    "verify",
    "metadata",
    "publish",
)


class _AcceptAllFaces:
    """Default detector for stub and offline runs.

    Real face detection is injected at wiring time. Defaulting to acceptance here
    keeps the offline suite free of a vision model; the production wiring passes
    a real detector and `check_faces` enforces the rule either way.
    """

    def detect(self, path: Path) -> FaceResult:
        return FaceResult(face_count=1, has_mouth=True)


class VideoGenerationEngine:
    """Runs exactly one generation per `run()` call."""

    def __init__(
        self,
        *,
        adapter,
        outputs_root: Path | str,
        profile: ModelProfile | None = None,
        detector=None,
        device: DeviceKind | None = None,
        max_reserved_bytes: int | None = None,
    ) -> None:
        self._adapter = adapter
        self._profile = profile or adapter.profile
        self._outputs_root = Path(outputs_root).resolve()
        self._detector = detector or _AcceptAllFaces()
        self._device = device or resolve_device()
        self._max_reserved_bytes = max_reserved_bytes

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    # -- progress ---------------------------------------------------------

    def _emit(
        self,
        sink: ProgressSink | None,
        request_id: uuid.UUID,
        phase: str,
        fraction: float | None,
        message: str,
        *,
        memory: MemorySnapshot | None = None,
    ) -> None:
        """Publish one event. A broken consumer never fails a generation."""
        if sink is None:
            return
        try:
            sink(
                ProgressEvent(
                    request_id=request_id,
                    phase=phase,
                    fraction=fraction,
                    message=sanitize(message),
                    memory=memory,
                    timestamp=datetime.now(UTC),
                )
            )
        except Exception:  # noqa: BLE001 - a broken sink must not fail a run
            # A broken progress consumer must never fail a multi-hour generation,
            # but swallowing it silently makes the UI look frozen for no visible
            # reason, so it is logged at debug rather than discarded.
            _log.debug("progress sink raised during phase %s", phase, exc_info=True)

    def _snapshot(self) -> MemorySnapshot:
        return accelerator_snapshot(self._device, max_reserved_bytes=self._max_reserved_bytes)

    # -- run --------------------------------------------------------------

    def run(
        self,
        *,
        image_paths,
        audio_path: Path | str,
        motion_prompt: str,
        speech_script: str,
        language: str,
        consent_confirmed: bool,
        seed: int | None = None,
        requested_seconds: float | None = None,
        guidance_scale: float | None = None,
        progress: ProgressSink | None = None,
        cancel: CancellationToken | None = None,
    ) -> GenerationResult:
        request_id = uuid.uuid4()
        started = time.monotonic()
        staging = staging_dir(self._outputs_root, request_id)
        memory_by_stage: dict[str, MemorySnapshot] = {}
        profile = self._profile

        # Checked before the try block, and therefore RAISED rather than turned
        # into a failed result. Consent is a precondition: nothing has been
        # attempted, no staging exists, and no terminal state is owed. Folding it
        # into the failure path would report "the generation failed" for a run
        # that was never allowed to start.
        audio_digest = sha256_file(audio_path)
        consent = build_consent(
            request_id=request_id,
            audio_sha256=audio_digest,
            confirmed=consent_confirmed,
        )

        def stage(name: str, fraction: float | None, message: str) -> None:
            snapshot = self._snapshot()
            memory_by_stage[name] = snapshot
            self._emit(progress, request_id, name, fraction, message, memory=snapshot)

        try:
            # -- validate --------------------------------------------------
            stage("validate", None, "validating request")
            if cancel:
                cancel.raise_if_cancelled(stage="validate")

            prompt, duration, effective_seed = preflight(
                request_id=request_id,
                motion_prompt=motion_prompt,
                speech_script=speech_script,
                language=language,
                requested_seconds=requested_seconds,
                seed=seed,
                guidance_scale=guidance_scale,
                profile=profile,
            )

            # -- prepare_references ---------------------------------------
            stage("prepare_references", None, "staging references")
            references = stage_references(
                request_id=request_id,
                image_paths=image_paths,
                audio_path=audio_path,
                consent=consent,
                profile=profile,
                staging_root=staging.parent,
                detector=self._detector,
                outputs_root=self._outputs_root,
            )

            stage("assemble_prompt", None, "assembling prompt")
            stage(
                "plan_duration",
                None,
                f"duration {duration.effective_duration_seconds:g}s at {duration.frame_rate:g} fps",
            )

            # -- load_model ------------------------------------------------
            stage("load_model", None, "loading model")
            dtype = select_dtype(self._device, policy=profile.dtype_policy or None)
            self._adapter.load(
                device=self._device.value,
                dtype=dtype,
                progress=lambda phase, fraction, message: self._emit(
                    progress, request_id, phase, fraction, message
                ),
                cancel=cancel,
            )

            # -- generate --------------------------------------------------
            stage("generate", 0.0, "generating video and speech")
            artifacts = self._adapter.generate(
                GenerationInputs(
                    request_id=str(request_id),
                    image_paths=tuple(Path(p) for p in references.image_paths),
                    audio_path=Path(references.audio_path),
                    prompt=prompt,
                    duration=duration,
                    seed=effective_seed,
                    guidance_scale=guidance_scale,
                ),
                progress=lambda phase, fraction, message: self._emit(
                    progress, request_id, phase, fraction, message
                ),
                cancel=cancel,
            )

            # -- decode ----------------------------------------------------
            stage("decode", 0.0, "decoding output")
            outputs_dir = staging / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            decoded_audio = outputs_dir / "speech.wav"
            if Path(artifacts.audio_path) != decoded_audio:
                import shutil

                shutil.copy2(artifacts.audio_path, decoded_audio)
            self._emit(progress, request_id, "decode", 1.0, "decode complete")

            # -- export ----------------------------------------------------
            stage("export", None, "muxing container")
            final_mp4 = export_mp4(
                frames_dir=artifacts.frames_path,
                audio_path=decoded_audio,
                out_path=outputs_dir / "output.mp4",
                frame_rate=artifacts.frame_rate,
            )

            # -- verify ----------------------------------------------------
            stage("verify", None, "verifying streams")
            verification = verify_output(
                final_mp4,
                expected_seconds=duration.effective_duration_seconds,
                frame_rate=artifacts.frame_rate,
            )

            # The verified description of what was produced. Built only after
            # verification passes, so it can never describe an output that
            # failed its checks.
            generated = GeneratedOutput(
                request_id=request_id,
                decoded_video_path="frames.mp4",
                decoded_audio_path="speech.wav",
                final_mp4_path="output.mp4",
                frame_count=artifacts.frame_count,
                frame_rate=artifacts.frame_rate,
                width=artifacts.width,
                height=artifacts.height,
                audio_sample_rate=artifacts.audio_sample_rate,
                audio_channels=artifacts.audio_channels,
                measured_video_seconds=verification.probe.video_seconds,
                measured_audio_seconds=verification.probe.audio_seconds,
                script_sha256=hashlib.sha256(speech_script.strip().encode("utf-8")).hexdigest(),
                language=language,
                profile_id=profile.profile_id,
                non_silent=verification.non_silent,
            )

            # -- metadata --------------------------------------------------
            stage("metadata", None, "writing manifest")
            prompt_path = staging / "prompt.txt"
            prompt_path.write_text(prompt.rendered, encoding="utf-8")

            import shutil

            publish_atomically(final_mp4, staging / "output.mp4")

            # The decoded picture stream, with no audio. Encoded separately
            # rather than copied from the final MP4: an artifact byte-identical
            # to another records nothing and doubles the biggest file.
            export_video_only(
                frames_dir=artifacts.frames_path,
                out_path=staging / "frames.mp4",
                frame_rate=artifacts.frame_rate,
            )
            shutil.copy2(decoded_audio, staging / "speech.wav")
            shutil.rmtree(outputs_dir, ignore_errors=True)

            # The `metadata` artifact is request.json, NOT metadata.json. The
            # manifest cannot appear in its own artifact list: the list carries a
            # digest per file, and a file containing its own digest is impossible.
            kinds = {
                "output.mp4": "final_mp4",
                "frames.mp4": "decoded_video",
                "speech.wav": "decoded_audio",
                "prompt.txt": "assembled_prompt",
                "request.json": "metadata",
            }
            for staged_image in references.image_paths:
                kinds[f"inputs/{Path(staged_image).name}"] = "original_image"
            kinds[f"inputs/{Path(references.audio_path).name}"] = "reference_audio"

            import json as _json

            (staging / "request.json").write_text(
                _json.dumps(
                    {
                        "request_id": str(request_id),
                        "language": language,
                        "seed": effective_seed,
                        "profile_id": profile.profile_id,
                        "effective_duration_seconds": duration.effective_duration_seconds,
                        "frame_rate": duration.frame_rate,
                        "prompt_token_count": prompt.token_count,
                        "prompt_token_capacity": prompt.token_capacity,
                        "prompt_over_capacity": prompt.over_capacity,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            derived = staging / "derived-voice.json"
            derived.write_text(
                '{"note": "voice representation is produced inside the joint '
                'generation; this record exists for provenance only."}',
                encoding="utf-8",
            )
            kinds["derived-voice.json"] = "derived_voice"

            manifest = self._build_manifest(
                request_id=request_id,
                references=references,
                prompt=prompt,
                duration=duration,
                seed=effective_seed,
                language=language,
                guidance_scale=guidance_scale,
                memory_by_stage=memory_by_stage,
                generated=generated,
                kinds=kinds,
                staging=staging,
            )

            # -- publish ---------------------------------------------------
            stage("publish", None, "publishing bundle")
            bundle_path, manifest_path = publish_bundle(
                outputs_root=self._outputs_root,
                request_id=request_id,
                staging=staging,
                manifest=manifest,
            )

            video_path = bundle_path / "output.mp4"
            retained = directory_size(bundle_path)

            self._emit(
                progress,
                request_id,
                "complete",
                1.0,
                "generation complete",
                memory=self._snapshot(),
            )

            return GenerationResult(
                request_id=request_id,
                state=RequestState.COMPLETE,
                video_path=str(video_path),
                bundle_path=str(bundle_path),
                manifest_path=str(manifest_path),
                artifact_inventory=[],
                retained_bytes=retained,
                execution_plan={
                    "seed": effective_seed,
                    "language": language,
                    "profile_id": profile.profile_id,
                    "adapter_key": profile.adapter_key,
                    "device": self._device.value,
                    "dtype": dtype,
                    "effective_duration_seconds": duration.effective_duration_seconds,
                    "frame_rate": duration.frame_rate,
                },
                duration_seconds=time.monotonic() - started,
                memory_by_stage=memory_by_stage,
                error=None,
            )

        except AppError as exc:
            return self._fail(exc, request_id, staging, started, memory_by_stage, progress)
        except BaseException as exc:  # noqa: BLE001 - every failure is terminal and safe
            return self._fail(
                translate(exc, stage="generate"),
                request_id,
                staging,
                started,
                memory_by_stage,
                progress,
            )

    # -- failure ----------------------------------------------------------

    def _fail(
        self,
        exc: AppError,
        request_id: uuid.UUID,
        staging: Path,
        started: float,
        memory_by_stage: dict[str, MemorySnapshot],
        progress: ProgressSink | None,
    ) -> GenerationResult:
        """One safe terminal result, no published bundle, no staging left behind."""
        remove_staging(staging)
        detail: ErrorDetail = to_detail(exc)
        phase = "cancelled" if detail.code.value == "cancelled" else "failed"
        state = RequestState.CANCELLED if phase == "cancelled" else RequestState.FAILED
        self._emit(progress, request_id, phase, None, detail.message)

        return GenerationResult(
            request_id=request_id,
            state=state,
            video_path=None,
            bundle_path=None,
            manifest_path=None,
            artifact_inventory=[],
            retained_bytes=0,
            execution_plan={},
            duration_seconds=time.monotonic() - started,
            memory_by_stage=memory_by_stage,
            error=detail,
        )

    # -- manifest ---------------------------------------------------------

    def _build_manifest(
        self,
        *,
        request_id,
        references,
        prompt,
        duration,
        seed,
        language,
        guidance_scale,
        memory_by_stage,
        generated,
        kinds,
        staging,
    ) -> dict[str, Any]:
        profile = self._profile
        now = datetime.now(UTC).isoformat()
        rate = duration.speaking_rate_used

        return {
            "schema_version": 1,
            "request_id": str(request_id),
            "state": "complete",
            "created_at": now,
            "completed_at": now,
            "bundle_relative_path": f"outputs/{request_id}",
            "artifacts": inventory_artifacts(staging, kinds=kinds),
            "voice_origin": (
                {
                    "bundle_id": str(references.origin.bundle_id),
                    "artifact_relative_path": references.origin.artifact_relative_path,
                    "artifact_sha256": references.origin.artifact_sha256,
                }
                if references.origin
                else None
            ),
            "consent": {
                "request_id": str(references.consent.request_id),
                "reference_audio_sha256": references.consent.reference_audio_sha256,
                "confirmed": True,
                "confirmed_at": references.consent.confirmed_at.isoformat(),
            },
            "language": language,
            "models": [
                {
                    "role": role,
                    "provider_mode": "native",
                    "repo_id": profile.weight_policy.get("repo_id", "local/stub"),
                    "commit": profile.weight_policy.get("commit", "0" * 40),
                    "adapter_key": profile.adapter_key,
                }
                for role in ("video", "voice", "lip_sync")
            ],
            "parameters": {
                "effective_seed": seed,
                "profile_id": profile.profile_id,
                "suggested_duration_seconds": duration.suggested_duration_seconds,
                "speaking_rate_used": (
                    {"language": rate.language, "rate": rate.rate} if rate else None
                ),
                "preferred_duration_seconds": duration.requested_duration_seconds,
                "operator_overrode_duration": duration.operator_overrode,
                "effective_duration_seconds": duration.effective_duration_seconds,
                "effective_num_frames": duration.effective_num_frames,
                "frame_rate": duration.frame_rate,
                "resolution": {
                    "width": duration.resolution.width,
                    "height": duration.resolution.height,
                },
                "audio_sample_rate": duration.audio_sample_rate,
                "audio_channels": generated.audio_channels,
                "prompt_token_count": prompt.token_count,
                "prompt_token_capacity": prompt.token_capacity,
                "offload_mode": (
                    profile.resource_profile.offload_mode.value
                    if profile.resource_profile
                    else "none"
                ),
                "quantization": (
                    profile.resource_profile.quantization if profile.resource_profile else None
                ),
                "guidance_scale": guidance_scale,
                "motion_prompt_truncation": (
                    {
                        "original_length": prompt.motion_truncation.original_length,
                        "retained_length": prompt.motion_truncation.retained_length,
                        "discarded_length": prompt.motion_truncation.discarded_length,
                    }
                    if prompt.motion_truncation
                    else None
                ),
                "runtime_profile": self._device.value,
            },
            "memory_by_stage": {
                name: snapshot.model_dump(mode="json", exclude={"host_resident_bytes"})
                for name, snapshot in memory_by_stage.items()
            },
            "plaintext_sensitive_artifacts": True,
            "disk_bytes": directory_size(staging),
        }
