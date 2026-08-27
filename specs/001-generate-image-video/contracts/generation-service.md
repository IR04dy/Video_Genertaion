# Contract: Generation Service

This is the in-process boundary between `app.py` and `pipeline.py`. The UI depends on this contract
and domain records, never on PyTorch, Diffusers, Hub clients, model objects, or tensors.

## Protocol

```python
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from domain import (
    GenerationRequest,
    GenerationResult,
    ModelSetDraft,
    ModelSetResolution,
    ProgressEvent,
    RuntimeSummary,
)

ProgressCallback = Callable[[ProgressEvent], None]


class GenerationService(Protocol):
    def runtime_summary(self) -> RuntimeSummary:
        """Return device/runtime capabilities without loading model weights."""

    def resolve_model_set(self, draft: ModelSetDraft) -> ModelSetResolution:
        """Resolve role coverage, overlaps, constraints, and required provider choices."""

    def generate(
        self,
        request: GenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Run exactly one accepted request and return exactly one terminal result."""

    def cancel(self, request_id: str) -> bool:
        """Request cooperative cancellation; return whether the request was active/queued."""

    def cleanup(self, request_id: str) -> list[Path]:
        """Delete artifacts owned by a terminal request and return deleted paths."""
```

## Preconditions

- Scalar/path input matches `generation-request.schema.json` and all paths resolve beneath the
  configured upload workspace.
- `voice_consent_confirmed` is true; `voice_consent_at` is assigned server-side for this request.
- The image passes still-image normalization and exactly-one-face/mouth preflight.
- The reference recording and selected language satisfy the effective voice provider profile.
- Every selected model is ready, references an immutable commit, and has an installed adapter.
- Video, voice, and lip-sync roles resolve exactly once. Every native/dedicated overlap has an
  explicit provider choice.
- The selected runtime profile is compatible with all effective providers.
- The service admits at most one active generation.

Validation and execution planning complete before any heavy model allocation.

## Model-set resolution guarantees

`resolve_model_set()` performs no model-weight loading and returns:

- effective provider candidates and capability coverage;
- required Native/Dedicated choices for overlaps;
- effective language and reference-audio constraints;
- video parameter/default constraints;
- device/memory compatibility warnings;
- safe validation errors when coverage is incomplete or incompatible.

If exactly one compatible provider covers a role, it may be selected directly. If two providers
cover it, resolution is incomplete until the user supplies an explicit choice.

## Progress guarantees

- Events are ordered, sanitized, and share the request ID.
- A normal dedicated flow emits `queued`, `validating`, `planning`, provider loading/inference,
  `voice`, `lip_sync`, `mux`, `verify`, `metadata`, and one terminal event.
- Native-composite plans emit only their actual stages; the immutable effective plan remains in
  result metadata.
- Fractions do not decrease within a stage. CUDA memory is reported when available and explicitly
  marked unavailable otherwise.
- Exactly one of `complete`, `failed`, or `cancelled` is emitted last.
- Callback failures are logged safely and do not corrupt generation state.

## Result guarantees

- Success returns an absolute `Path` to an atomically published, browser-playable MP4 containing
  exactly one video stream and one non-silent speech stream whose duration differs by at most one
  frame, plus a metadata path.
- Technical completion is not gated by an automated lip-sync quality score; the successful MP4 is
  returned for visual review.
- Metadata records immutable repository commits, adapters, effective role providers, explicit
  provider choices, consent timestamp, language, seed, normalized parameters, devices/dtypes,
  per-stage timings, and memory snapshots.
- Failure/cancellation returns no video path, a stable error code, safe message, and recovery steps.
- No raw framework exception, token, prompt, absolute upload path, or voice representation crosses
  the boundary or appears in normal logs.
- Every acquired model lease and temporary artifact is released/cleaned on every terminal path.

## Error mapping

| Domain code | Typical UI guidance |
|-------------|---------------------|
| `consent` | Confirm ownership or explicit permission for this reference voice |
| `face` | Supply an image with exactly one clear face and usable mouth region |
| `language` | Choose a language supported by the effective voice provider |
| `model_incompatible` | Select providers that cover all roles and satisfy device/interface constraints |
| `oom` | Lower valid frame/resolution preset, unload other GPU apps, or select stronger offload |
| `speech` | Check provider-specific audio/script rules and model access |
| `lip_sync` | Check face visibility, input video/audio, and lip model compatibility |
| `codec`/`mux` | Install/repair FFmpeg and verify output directory permissions |

## Gradio mapping

The generation handler maps a terminal result to:

```text
(video_path, download_path, status_markdown, progress_log, memory_markdown, metadata_summary)
```

On failure, both file outputs are `None`. `gr.Progress` renders `ProgressEvent` values but is never
passed into the engine. The same verified path feeds the embedded player and download control.
