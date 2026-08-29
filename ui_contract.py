"""UI state and handler logic (T045/T046), independent of Gradio.

Kept separate from `app.py` on purpose: the rules that matter — consent resetting
whenever the recording changes, the timbre-anchor warning being shown, a failure
clearing the video and download outputs — are testable without launching a
server or installing a UI framework. `app.py` binds widgets to these functions
and owns nothing of the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from domain import GenerationResult, ModelProfile, RequestState
from execution import suggest_duration

BIND_HOST = "127.0.0.1"
BIND_PORT = 7860


@dataclass(frozen=True)
class UiState:
    profile_id: str
    image_limit: int | None
    duration_min: float
    duration_max: float
    suggested_duration: float
    duration_editable: bool = True
    image_paths: tuple[str, ...] = ()
    audio_path: str | None = None
    motion_prompt: str = ""
    speech_script: str = ""
    language: str | None = None
    consent_confirmed: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResultOutputs:
    video_path: str | None
    download_path: str | None
    error_message: str | None
    suggestions: list[str]


def initial_state(profile: ModelProfile) -> UiState:
    supported = profile.duration_range_seconds
    return UiState(
        profile_id=profile.profile_id,
        image_limit=profile.reference_limits.accepted.get("image"),
        duration_min=supported.min_seconds,
        duration_max=supported.max_seconds,
        suggested_duration=supported.default_seconds,
        language=profile.dialogue_languages[0] if profile.dialogue_languages else None,
    )


def language_choices(profile: ModelProfile) -> list[str]:
    return list(profile.dialogue_languages)


def reference_audio_help(profile: ModelProfile) -> str:
    """The rule operators most often get wrong, stated plainly.

    A reference that repeats the script sounds like it should help and does not:
    the recording anchors timbre only and is never played back, so matching words
    just makes the reference less useful as a voice sample.
    """
    bounds = profile.reference_limits.audio_clip_seconds
    length = f" Aim for {bounds.min_seconds:g}-{bounds.max_seconds:g} seconds." if bounds else ""
    return (
        "This recording is a voice timbre anchor. It is never played back and is "
        "never mixed into the output. It should say DIFFERENT words from your "
        "speech script — matching the script does not help." + length
    )


def on_images_changed(state: UiState, paths) -> UiState:
    return replace(state, image_paths=tuple(str(p) for p in (paths or [])))


def on_audio_changed(state: UiState, path) -> UiState:
    """Changing the recording always clears consent.

    Unconditionally, even when the same file is reselected: consent is bound to
    specific bytes server-side, and a UI that kept the box ticked across a change
    would be showing agreement to something the operator did not agree to.
    """
    return replace(state, audio_path=str(path) if path else None, consent_confirmed=False)


def on_motion_changed(state: UiState, text: str) -> UiState:
    return replace(state, motion_prompt=text or "")


def on_script_changed(state: UiState, text: str, *, language: str | None = None) -> UiState:

    language = language or state.language
    script = text or ""
    state = replace(state, speech_script=script, language=language)

    profile = _profile_for(state)
    if profile is None or language is None:
        return state
    suggested, _ = suggest_duration(script=script, language=language, profile=profile)
    return replace(state, suggested_duration=suggested)


def on_consent_changed(state: UiState, confirmed: bool) -> UiState:
    return replace(state, consent_confirmed=bool(confirmed))


def after_submit(state: UiState) -> UiState:
    """Consent resets after every submission, successful or not."""
    return replace(state, consent_confirmed=False)


def can_submit(state: UiState) -> tuple[bool, str]:
    if not state.image_paths:
        return False, "Add at least one reference image."
    if not state.audio_path:
        return False, "Add a reference voice recording."
    if not state.speech_script.strip():
        return False, "Enter a speech script."
    if not state.consent_confirmed:
        return False, "Confirm voice-cloning consent before generating."
    return True, ""


def result_outputs(result: GenerationResult) -> ResultOutputs:
    """Map a terminal result onto the four output widgets.

    A failure must clear both the player and the download button: leaving the
    previous run's video visible beside a new error is how someone downloads the
    wrong file believing it is the new one.
    """
    if result.state is RequestState.COMPLETE:
        return ResultOutputs(
            video_path=result.video_path,
            download_path=result.video_path,
            error_message=None,
            suggestions=[],
        )
    error = result.error
    return ResultOutputs(
        video_path=None,
        download_path=None,
        error_message=error.message if error else "The run did not complete.",
        suggestions=list(error.suggestions) if error else [],
    )


def render_progress(*, phase: str, fraction: float | None, elapsed_seconds: float) -> str:
    """Phase, completion, and elapsed time — deliberately no estimate.

    Inference is unbounded and carries no runtime model, so any "time remaining"
    would be invented. Elapsed time is a measurement; an ETA would not be.
    """
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    stamp = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
    percent = f" - {round((fraction or 0) * 100)}%" if fraction is not None else ""
    return f"{phase}{percent} - elapsed {stamp}"


def memory_summary(*, accelerator_available: bool | None = None, snapshot=None) -> str:
    """Both budgets in one line: the card and the host.

    Host RAM is reported even with no accelerator, because layer-wise offload
    spends host memory as a first-class budget rather than an incidental one.
    """
    from devices import accelerator_snapshot, host_resident_bytes, resolve_device

    if snapshot is None:
        snapshot = accelerator_snapshot(resolve_device())
    available = snapshot.available if accelerator_available is None else accelerator_available

    gib = 1024**3
    host = (snapshot.host_resident_bytes or host_resident_bytes()) / gib

    if not available:
        return (
            f"Accelerator: unavailable ({snapshot.unavailable_reason or 'no accelerator'}). "
            f"Host memory: {host:.1f} GiB resident."
        )
    reserved = (snapshot.reserved_bytes or 0) / gib
    return (
        f"Accelerator: {snapshot.device_name or 'device'}, {reserved:.1f} GiB reserved. "
        f"Host memory: {host:.1f} GiB resident."
    )


def launch_kwargs() -> dict:
    """Loopback only, sharing off. This is a single-user local application."""
    return {
        "server_name": BIND_HOST,
        "server_port": BIND_PORT,
        "share": False,
        "show_api": False,
        "inbrowser": False,
    }


_PROFILE_CACHE: dict[str, ModelProfile] = {}


def register_profile(profile: ModelProfile) -> None:
    """Let state helpers resolve a profile by id without a global engine."""
    _PROFILE_CACHE[profile.profile_id] = profile


def _profile_for(state: UiState) -> ModelProfile | None:
    return _PROFILE_CACHE.get(state.profile_id)
