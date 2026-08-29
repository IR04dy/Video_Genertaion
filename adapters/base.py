"""The joint audio/video adapter protocol (T021).

One invocation produces video and speech together. There is no separate TTS step
and no lip-sync step, so this protocol has exactly one generating method — adding
a second would reintroduce the timebase bridge the architecture removed.

An adapter's only contract with the rest of the application is its `ModelProfile`.
Everything measured about the model — durations, frame rate, resolutions, sample
rate, languages, speaking rates, reference limits, token capacity — is read from
that profile. Nothing here declares a default for any of them.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from domain import AssembledPrompt, DurationDecision, ModelProfile
from errors import CancelledError

ProgressCallback = Callable[[str, float | None, str], None]
"""(phase, fraction_or_None, message). Fractions are monotonic within a phase."""


class CancellationToken:
    """Cooperative cancellation, checked at bounded intervals inside long stages.

    Cooperative rather than pre-emptive because a killed thread mid-write leaves
    a half-written bundle. `raise_if_cancelled()` is called between denoising
    steps and between decode chunks, so the stage unwinds through its own
    `finally` blocks and cleans up staging.

    Bounds cleanup, never runtime: inference itself is unbounded by decision and
    is never cancelled on elapsed time.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self, *, stage: str = "generation") -> None:
        if self._event.is_set():
            raise CancelledError(f"Cancelled during {stage}.")


@dataclass(frozen=True)
class GenerationInputs:
    """Everything one invocation needs, already validated and staged."""

    request_id: str
    image_paths: tuple[Path, ...]
    audio_path: Path
    prompt: AssembledPrompt
    duration: DurationDecision
    seed: int
    guidance_scale: float | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationArtifacts:
    """Raw joint output, before export and verification."""

    frames_path: Path
    audio_path: Path
    frame_rate: float
    audio_sample_rate: int
    audio_channels: int
    width: int
    height: int
    frame_count: int


@runtime_checkable
class JointAdapter(Protocol):
    """A reviewed adapter producing video and speech in one invocation."""

    @property
    def profile(self) -> ModelProfile:
        """The measured capability manifest. The only contract with the app."""
        ...

    def load(
        self,
        *,
        device: str,
        dtype: str,
        progress: ProgressCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> None:
        """Bring weights into memory using the profile's declared offload mode.

        Offload mode and quantization are read from the profile, never probed.
        """
        ...

    def generate(
        self,
        inputs: GenerationInputs,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> GenerationArtifacts:
        """Produce video and speech jointly. Called exactly once per request."""
        ...

    def unload(self) -> None:
        """Release weights and accelerator memory. Must be idempotent."""
        ...


def emit(
    progress: ProgressCallback | None, phase: str, fraction: float | None, message: str
) -> None:
    """Report progress if anyone is listening, and never fail because of it.

    A broken progress consumer must not be able to fail a multi-hour generation.
    """
    if progress is None:
        return
    try:
        progress(phase, fraction, message)
    except Exception:  # noqa: BLE001 - a broken consumer must not fail generation
        logging.getLogger(__name__).debug(
            "progress callback raised during phase %s", phase, exc_info=True
        )
