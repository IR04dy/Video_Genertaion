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

A precondition failure **raises**; it does not produce a terminal `GenerationResult`. Nothing has been
attempted, no staging directory exists, and no terminal state is owed. Returning `failed` here would
report that a generation was tried and did not work, for a run that was never allowed to start.
Everything after the preconditions — an OOM, a codec failure, a cancellation — returns exactly one
terminal result instead.

- Scalar/path input matches `generation-request.schema.json`. Each input path is either a current
  upload-workspace file or a contained, non-symlink-escaping file beneath a valid earlier bundle; the
  service copies it into fixed `outputs/.work/<request-id>/inputs/` before inference.
- Output root, `.work`, bundle, and artifact components pass strict resolution plus symlink/Windows
  reparse-point checks; server UUIDs—not client paths—name staging/published directories.
- The current submit event carries `voice_consent_confirmed=true`; the server creates a fresh
  `ConsentAttestation` bound to request ID, staged reference-audio SHA-256, and current timestamp.
- Every image passes still-image normalization and an independent exactly-one-face/mouth preflight.
- The trimmed motion prompt and speech script are non-empty. The motion prompt has no application
  maximum and is truncated only to the profile's measured token capacity, with a reported override. The
  speech script has no application maximum and is never refused for length; it drives the suggested
  duration through the profile's per-language speaking rate.
- One or more image references and exactly one audio timbre anchor are supplied, all satisfying the
  effective profile's measured accepted kinds, counts, and per-clip duration bounds. The service imposes
  no image-count maximum of its own. Video references are refused.
- The selected language is a member of the effective profile's measured dialogue-language set.
- Every selected model is ready, references an immutable commit, and has an installed adapter.
- Video, voice, and lip-sync roles resolve exactly once; a single profile may supply all three natively,
  in which case no dedicated providers are selected. Any remaining native/dedicated overlap has an
  explicit provider choice.
- The selected runtime profile is compatible with the effective provider and satisfies **both** the
  accelerator-memory ceiling and the host system-memory ceiling.
- Conservative complete-bundle size plus the configured reserve (10 GiB default) fits before inference.
- The service admits at most one active generation, and refuses to start model downloads while one is
  active.

Static validation and the `ExecutionBlueprint` complete before any model allocation. The immutable
`EffectiveExecutionPlan` is finalized once a valid `DurationDecision` and `AssembledPrompt` exist.

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

## Single-generation guarantees

1. Validate consent, input media, a face target per image, all references against the effective profile's accepted
   kinds and per-clip limits, the selected language against the profile's measured language set, model
   coverage, paths, device and host-memory feasibility, and the initial disk estimate before any model
   inference. Reject video references with a specific reason.
2. Assemble the prompt: motion description plus the speech script rendered in the profile's documented
   dialogue-tag form with the selected language. Measure the token count against the profile's measured
   capacity. If the motion portion exceeds capacity, truncate **only** the motion portion and record
   `original_length`, `retained_length`, and `discarded_length` as an explicit override. Dialogue
   segments are never truncated, dropped, or reordered.
3. Suggest a duration from the trimmed script and the profile's per-language speaking rate, clamped to
   the profile's supported range, and accept an operator override anywhere in that range. Duration is an
   **input** to joint generation and cannot be measured from synthesized speech beforehand, so the
   service performs no pre-generation fit check and never returns `duration` for script length. Never
   trim, time-stretch, truncate, or partially omit speech.
4. Acquire leases, recheck the refined bundle estimate, and run **exactly one** joint audio/video
   generation under the profile's declared offload mode and quantization. Video and stereo audio are
   produced together. There is no separate speech phase, lip-sync phase, timebase conversion,
   post-generation face pass, loop expansion, or mux input assembly.
5. Decode video and audio, export the container, and verify exactly one video stream and one non-silent
   speech stream whose durations agree within one frame at the profile's frame rate.
6. Write the manifest, including the assembled prompt actually submitted, and atomically publish.

The service performs no repetition, looping, reversal, padding, cross-timebase resampling, or
concatenation of generated media under any circumstance.

### Reference contract

References per request:

- `image_paths` — one or more still images anchoring subject identity and appearance, all of the same
  subject. No service-imposed maximum; the profile's measured limit is the only bound, and each image
  must independently contain exactly one usable face and mouth region.
- `audio_path` — recording anchoring **voice timbre only**. It is never played back, never mixed into
  the output, and never treated as spoken content. The UI states it must say different words from the
  script. Spoken content arrives solely through dialogue tags in the assembled prompt.

Video references are refused with `reference`. This is not a policy preference: their token cost alone
breaches the accelerator ceiling before generation begins.

Consent is unchanged and still required, because this remains voice cloning. The service creates a fresh
attestation bound to request ID and staged reference-audio SHA-256, defaulting and resetting to false,
reset whenever the audio changes, and recorded in sanitized metadata.
## Progress guarantees

- Events are ordered, sanitized, and share the request ID.
- A normal flow emits `queued`, `validating`, `preparing_references`, `assembling_prompt`,
  `planning_duration`, `loading_model`, `generating`, `decoding`, `exporting`, `verifying`, `metadata`,
  `publish`, and one terminal event.
- No model-loading or generation event may precede successful validation, an accepted duration decision,
  and an assembled prompt.
- Because inference time is unbounded and layer-wise offload can make a single run last hours, the
  `generating` and `decoding` phases MUST report a monotonic completion fraction refreshed at least every
  few seconds. A long request must remain observably progressing rather than appearing stalled. No phase
  carries a latency target, and no event may be interpreted as a timeout signal.
- Fractions do not decrease within a stage. CUDA current/peak allocated, current/peak reserved, and
  free/total memory are reported when available, together with host resident memory; production
  acceptance gates peak reserved at 13.5 GiB and host resident at the configured ceiling.
- Exactly one of `complete`, `failed`, or `cancelled` is emitted last.
- Callback failures are logged safely and do not corrupt generation state.
## Result guarantees

- Success returns an absolute `Path` to an atomically published, browser-playable MP4 containing
  exactly one video stream and one non-silent speech stream whose duration differs by at most one
  frame, plus the fixed `outputs/<request-id>` bundle, manifest, complete artifact inventory, and size.
- A successful bundle retains the copied original image and reference audio, derived voice data, the
  assembled prompt actually submitted, the decoded video and audio, the final MP4, and metadata. The
  service exposes no successful-bundle cleanup/delete method and never expires published bundles.
- Technical completion is not gated by an automated lip-sync quality score; the successful MP4 is
  returned for visual review.
- Metadata records immutable repository commits, the adapter profile identity and revision, effective
  role providers, explicit provider choices, consent timestamp, language, seed, the suggested duration
  and the speaking-rate entry it came from, whether the operator overrode it, the effective duration,
  frame rate, resolution, audio sample rate, assembled-prompt token count and capacity, any
  motion-prompt truncation override, declared offload mode and quantization, devices/dtypes, per-stage
  timings, and allocated/reserved/free accelerator plus host resident memory snapshots. Every
  model-specific value is recorded as read from the profile, never as an application constant.
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
| `face` | Supply an image containing exactly one clear usable face and mouth region |
| `reference` | Supply one or more images and one audio timbre anchor within the profile's limits; video references are not accepted |
| `language` | Choose a language the effective profile lists as supported |
| `duration` | Choose a duration inside the profile's supported range; the range and the suggested value are shown |
| `model_incompatible` | Select a profile that covers all roles and satisfies device, interface, and both memory ceilings |
| `host_memory` | Free system memory, or select a profile whose declared quantization and offload fit the host ceiling |
| `disk` | Free model/bundle space manually; required, available, and reserve values are shown |
| `oom` | Lower a valid duration or resolution the profile supports, unload other GPU apps, or select stronger offload |
| `generation` | Check reference validity, prompt assembly, and model access |
| `export`/`codec` | Install/repair FFmpeg and verify output directory permissions |

## Gradio mapping

The generation handler maps a terminal result to:

```text
(video_path, download_path, status_markdown, progress_log, memory_markdown, metadata_summary)
```

On failure, both file outputs are `None`. `gr.Progress` renders `ProgressEvent` values but is never
passed into the engine. The same verified path feeds the embedded player and download control. The
consent control defaults to false and resets after every submit and whenever reference audio changes. The
reference-audio control displays the timbre-anchor rule: the recording is never played back and should
say different words from the speech script.
