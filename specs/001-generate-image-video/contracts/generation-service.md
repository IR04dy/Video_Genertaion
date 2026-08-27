# Contract: Generation Service

This is the in-process boundary between `app.py` and `pipeline.py`. The UI depends on this contract
and domain records, never on PyTorch, Diffusers, Hub clients, model objects, or tensors.

## Protocol

```python
from collections.abc import Callable
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
```

## Preconditions

- Scalar/path input matches `generation-request.schema.json`. Each input path is either a current
  upload-workspace file or a contained, non-symlink-escaping file beneath a valid earlier bundle; the
  service copies it into fixed `outputs/.work/<request-id>/inputs/` before inference.
- Output root, `.work`, bundle, and artifact components pass strict resolution plus symlink/Windows
  reparse-point checks; server UUIDs—not client paths—name staging/published directories.
- The current submit event carries `voice_consent_confirmed=true`; the server creates a fresh
  `ConsentAttestation` bound to request ID, staged reference-audio SHA-256, and current timestamp.
- The image passes still-image normalization and exactly-one-face/mouth preflight.
- The reference recording and selected language satisfy the effective voice provider profile.
- Every selected model is ready, references an immutable commit, and has an installed adapter.
- Video, voice, and lip-sync roles resolve exactly once. Every native/dedicated overlap has an
  explicit provider choice.
- The selected runtime profile is compatible with all effective providers.
- Conservative complete-bundle size plus the configured reserve (10 GiB default) fits before inference.
- The service admits at most one active generation.

Static validation and the `ExecutionBlueprint` complete before voice allocation. The immutable
`EffectiveExecutionPlan` is finalized only after verified speech yields a valid `DurationPlan`.

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

## Speech-first and duration guarantees

1. Validate consent, input media, face target, provider-specific audio/language rules, model coverage,
   paths, device feasibility, and the initial disk estimate before any model inference.
2. Acquire leases and synthesize the complete speech first. Persist the reference-derived voice data
   and decoded non-silent speech inside the staging bundle.
3. Derive effective frame count and FPS from exact decoded speech duration and the video adapter's
   versioned temporal rules. Record requested preferences and every override.
4. If no supported combination fits within one effective frame, return `duration` before loading the
   video model. Never trim, time-stretch, or partially omit speech.
5. Recheck the refined bundle estimate, then run video and lip-sync stages sequentially. The fixed
   CogVideoX profile offers only 49 frames at 8 FPS; for LatentSync 1.5, resample the complete clip to
   25 FPS without changing duration and invoke the versioned isolated worker.
6. Mux at the recorded effective final timebase, verify, write the manifest, and atomically publish.

## Progress guarantees

- Events are ordered, sanitized, and share the request ID.
- A normal dedicated flow emits `queued`, `validating`, `voice_synthesis`, `duration_planning`,
  `video_inference`, `lip_sync`, `mux`, `verify`, `metadata`, `publish`, and one terminal event.
- No video-loading or video-inference event may precede successful speech verification and an accepted
  duration plan. A native composite emits a distinct speech phase before any fused video/lip phase.
- Fractions do not decrease within a stage. CUDA memory is reported when available and explicitly
  marked unavailable otherwise.
- Exactly one of `complete`, `failed`, or `cancelled` is emitted last.
- Callback failures are logged safely and do not corrupt generation state.

## Result guarantees

- Success returns an absolute `Path` to an atomically published, browser-playable MP4 containing
  exactly one video stream and one non-silent speech stream whose duration differs by at most one
  frame, plus the fixed `outputs/<request-id>` bundle, manifest, complete artifact inventory, and size.
- A successful bundle retains the copied original image/reference audio, derived voice data, speech,
  pre-lip video, post-lip video, final MP4, and metadata. The service exposes no successful-bundle
  cleanup/delete method and never expires published bundles.
- Technical completion is not gated by an automated lip-sync quality score; the successful MP4 is
  returned for visual review.
- Metadata records immutable repository commits, adapters, effective role providers, explicit
  provider choices, consent timestamp, language, seed, normalized parameters, devices/dtypes,
  per-stage timings, and memory snapshots.
- Failure/cancellation returns no video path, a stable error code, safe message, and recovery steps.
- No raw framework exception, token, prompt, absolute upload path, or voice representation crosses
  the boundary or appears in normal logs.
- No model-license field, notice, acknowledgement, or compatibility decision appears in request/result
  records or UI summaries; gated/private authentication errors remain ordinary access failures.
- Every acquired model lease is released on every terminal path. Failed/cancelled staging directories
  are removed; successful staged artifacts are published and retained unchanged.
- Staging cleanup accepts only an internal verified handle for the exact inactive
  `outputs/.work/<uuid>` and never follows links or accepts a published/user-supplied path. Startup
  orphan recovery uses owner/lock markers before the same bounded cleanup.

## Error mapping

| Domain code | Typical UI guidance |
|-------------|---------------------|
| `consent` | Confirm ownership or explicit permission for this reference voice |
| `face` | Supply an image with exactly one clear face and usable mouth region |
| `language` | Choose a language supported by the effective voice provider |
| `model_incompatible` | Select providers that cover all roles and satisfy device/interface constraints |
| `disk` | Free model/bundle space manually; required, available, and reserve values are shown |
| `oom` | Lower valid frame/resolution preset, unload other GPU apps, or select stronger offload |
| `speech` | Check provider-specific audio/script rules and model access |
| `duration` | Shorten the script or choose a video adapter that supports the full speech duration |
| `lip_sync` | Check face visibility, input video/audio, and lip model compatibility |
| `codec`/`mux` | Install/repair FFmpeg and verify output directory permissions |

## Gradio mapping

The generation handler maps a terminal result to:

```text
(video_path, download_path, status_markdown, progress_log, memory_markdown, metadata_summary)
```

On failure, both file outputs are `None`. `gr.Progress` renders `ProgressEvent` values but is never
passed into the engine. The same verified path feeds the embedded player and download control. The
consent control defaults to false and resets after every submit and whenever reference audio changes.
