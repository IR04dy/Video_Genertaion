# Implementation Plan: Generate Image-Conditioned Lip-Synced Video

**Branch**: `main` (feature `001-generate-image-video`) | **Date**: 2026-08-28 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-generate-image-video/spec.md`

## Summary

Build a local, single-user Python 3.11 application that accepts one or more still images, a motion prompt, a
speech script, a reference-voice recording, a language, and a per-request voice-consent attestation,
and returns a browser-playable MP4 whose speech is cloned from the reference voice with natively
synchronized mouth movement.

> **Retargeted 2026-09-01.** `MiniMaxAI/MiniMax-H3` was abandoned: its largest indivisible component is
> 15.5 GiB at int4, over the 13.5 GiB accelerator ceiling, so no offload strategy could make it run here.
> Sections describing H3 specifics are superseded — see `research.md` → "Stack decision superseded".

The default reviewed profile covers three roles with **two models behind one adapter**:
`Wan-AI/Wan2.2-S2V-14B` for video and lip sync, and a TTS for voice. Because audio conditions the Wan
denoiser directly, mouth movement is a property of generation rather than a separate pass — so there is
still no separate lip-synchronization stage, no cross-provider timebase bridge, and no post-generation
face preflight.

Speech is synthesized first, inside the same `generate()` call, which means its duration is **measured
rather than estimated**. `JointAdapter` describes the *interface*, not the model, so composing two models
behind one invocation needs no protocol change. The adapter registry and Hugging Face URL selection are
unchanged: this is the default reviewed profile, not the only one.

Each request is exactly one generation at a duration the adapter profile declares as supported. Output
duration is therefore bounded by the profile, while inference time is explicitly unbounded: layer-wise
and sequential CPU offload and quantized checkpoints are expected on the target card, and a run taking
hours is acceptable. Successful requests retain every input, derived voice artifact, decoded output,
assembled prompt, final MP4, and metadata as ordinary unencrypted files beneath fixed
`outputs/<request-id>/`. A read-only Request History scans that directory; reuse and deletion remain
external filesystem actions. The application neither inspects nor records model-license information.

## Technical Context

**Language/Version**: Python 3.11 (`>=3.11,<3.13`)

**Primary Dependencies**: PyTorch 2.13.x with the official Windows CUDA 13.0 wheel (floor: torch 2.5 —
Transformers 5.x disables its model classes below 2.5, and 2.6 is where `torch.load` defaults to
`weights_only=True`, which is the gate on Wan's `.pth` pickles); `diffsynth==2.1.5`, providing
`WanVideoPipeline`, `ModelConfig`, and the `wan_video_dit_s2v` denoiser — Diffusers is **not** the load
path, because neither `WanSpeechToVideoPipeline` nor `WanS2VTransformer3DModel` exists in any Diffusers
release; `transformers==5.16.1`, providing the Wav2Vec2 audio encoder and processor;
Accelerate for layer-wise/sequential offload; a reviewed quantization backend; Hugging Face Hub,
Safetensors, Gradio, Pillow, NumPy, SoundFile/librosa, imageio, imageio-ffmpeg, Pydantic 2.x, psutil, and
filelock. The Qwen3-TTS dependency set and the separately locked LatentSync worker stack are removed
along with their stages, which eliminates the previous cross-provider dependency conflict and the local
worker-process boundary. The blocking stack-compatibility gate was **rewritten for this stack**; its
network assertions pass and its package assertions await an install on the production host. See
`tests/integration/test_stack_compatibility.py`. The **voice** dependency is deliberately unpinned:
`chatterbox-tts` hard-pins `torch==2.6.0`, which would remove the cu130 build and disable the RTX 5080 —
see `research.md` → "Voice packaging conflict".

**Storage**: Fixed project `outputs/` for complete successful request bundles and read-only history;
fixed project `.model-cache/` for application-owned model snapshots/inventory, separate from the global
Hub cache; atomic JSON manifests and locks; ordinary unencrypted files; no database and no configurable
bundle root in v1

**Testing**: pytest, pytest-cov, fake Hub/model/media adapters for offline tests; a blocking
`stack_compatibility` spike plus opt-in `mps`, `cuda`, and `model_download` markers for real backends

**Target Platform**: Development/control-path testing on macOS 13+; production on 64-bit Windows 11 with
an Intel Core Ultra 9 285K, **64 GB host system memory**, one NVIDIA RTX 5080 (**16 GB VRAM**), driver
580.88 or newer, and official PyTorch 2.13 CUDA 13.0 wheels

**Project Type**: Local browser-based Python orchestrator with in-process catalog, history, and a single
in-process joint audio/video generation adapter

**Performance Goals**: Stub UI available within 5 seconds; at least one status event per phase; model
download status at least every 2 seconds or completed chunk; a monotonic completion fraction during
inference and decoding at least every few seconds; every heavy stage at or below **13.5 GiB peak
allocator-reserved accelerator memory** and at or below the configured **host system-memory ceiling** on
the target machine, with peak allocated reported diagnostically; final audio and video duration differing
by no more than one frame; a configurable disk reserve defaulting to 10 GiB. **There is no latency target
or SLA of any kind.** Inference time is explicitly unbounded and a multi-hour run is a supported outcome,
not a defect.

**Constraints**: One active generation; the model library is read-only while a generation is active; one
joint audio/video generation per request at an adapter-declared supported duration; no chaining,
looping, repetition, or concatenation; speech is never trimmed, time-stretched, truncated, or partially
omitted, and no request is rejected for script length because duration is an input rather than a
measurement; one or more image references plus exactly one voice timbre anchor per request, bounded only
by the profile's measured limits, and video references rejected; the
reference recording is never played back and must say different words from the script; no application
maximum on the motion prompt, which is truncated to the profile's prompt capacity with a reported
override; immutable commit-pinned models and manual update checks; no remote code; no runtime
model-license handling; exactly one face in the input image; provider-specific language/audio validation;
fixed `outputs/`; full successful-bundle retention; filesystem-only bundle reuse/deletion; read-only
advisory dependency history; plaintext artifacts; loopback binding; no automated lip-sync quality gate;
offline stub tests download no weights; every adapter's duration, resolution, frame rate, audio sample
rate, language set, reference limits, and token capacity are measured profile fields, never
architecture-level invariants

**Scale/Scope**: One local operator, one generation at a time, one default reviewed profile with an open
registry for more, multiple immutable model revisions, exactly one model invocation per request, and one
complete retained bundle per success

## Constitution Check

*GATE: Passed before Phase 0 and re-checked after Phase 1 design.*

| Gate | Pre-research | Post-design evidence |
|------|--------------|----------------------|
| Cross-platform/device safety | PASS | `pathlib.Path`, fixed project-relative roots, CUDA/MPS/CPU resolver, PyTorch 2.13/CUDA 13.0 Windows profile, no platform-specific subprocess boundary, platform quickstart |
| Inference/UI separation | PASS | `GenerationService`, `ModelCatalogService`, and `RequestHistoryService` isolate Gradio from tensors, Hub, and storage scans; `prompting.py` keeps prompt assembly out of the UI |
| Bounded/reproducible inference | PASS | Commit-pinned dependency closures, measured per-profile capability fields, one model invocation per request, recorded suggested/effective duration with its speaking-rate provenance, recorded assembled prompt, seeded plans, declared offload/quantization policy, leases, dual accelerator and host-memory gates, disk gates |
| Test-first reliability | PASS | Failing offline unit/contract tests precede the adapter; a blocking stack-compatibility spike precedes architecture lock; real network/MPS/CUDA tests are opt-in |
| Observable/local operation | PASS | Phase progress plus a completion fraction during inference and decoding, redacted logs, accelerator and host memory metrics, fixed local bundles, explicit plaintext/retention disclosure, no public sharing, no hosted-API calls |

Plaintext storage is an explicit product decision, not a constitutional exception: the constitution
requires local handling and disclosure but does not mandate application-layer encryption. No
constitutional exceptions are required.

The post-design gate approves the architecture, not unmeasured hardware claims. The following remain
mandatory release measurements rather than assumptions: that the pinned Diffusers/Transformers releases
export the model-stack classes with `trust_remote_code=False`; the resident host footprint per precision against
64 GB; peak allocator-reserved accelerator memory against 13.5 GiB; and **the real supported duration
ceiling on this card, which is measured rather than assumed to be the card's stated 15 seconds**. A failed
clean-install, class-availability, offline, duration, or dual-ceiling test leaves the profile
`incompatible` and blocks the production release rather than weakening a constraint.

Three design facts are recorded here because they carry quality risk the gate does not cover.
No hosted prompt-structuring service is called, so prompt structuring is built locally. The output
resolution ceiling is a measured profile field rather than a vendor claim. And
because audio and video are generated jointly, whether a script actually fits the requested duration is
unknowable before generation: the system offers a speaking-rate-derived suggestion and records its
provenance, but delivery quality is judged by the operator after the fact, exactly as lip-sync quality
already is. No automated gate is claimed for either.

## Project Structure

### Documentation (this feature)

```text
specs/001-generate-image-video/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── generation-service.md
│   ├── model-catalog-service.md
│   ├── request-history-service.md
│   ├── generation-request.schema.json
│   ├── model-source.schema.json
│   └── request-bundle.schema.json
└── tasks.md                         # 94 tasks; 47 complete through User Story 1
```

### Source Code (repository root)

```text
app.py                               # Gradio Blocks, read-only history, event mapping
pipeline.py                          # Public VideoGenerationEngine facade
config.py                            # Bounds, fixed roots, reserve, dual resource ceilings, settings
domain.py                            # Pydantic requests, profiles, plans, bundles, events/results
devices.py                           # CUDA/MPS/CPU resolution, dtype, accelerator + host memory snapshots
errors.py                            # Stable safe domain errors
logging_config.py                    # Redacted structured logging
media.py                             # Image/audio inspection, face preflight, retained-path detection
prompting.py                         # Dialogue-tag assembly, token measurement, truncation overrides
validation.py                        # Language/seed/guidance checks, all bounds read from the profile
ui_contract.py                       # UI state and handler rules, testable without Gradio
export.py                            # Container export, stream/duration validation
execution.py                         # Single-generation plan: duration selection and resource preflight
model_catalog.py                     # URL inspection, immutable downloads/updates, inventory/deletion
model_registry.py                    # Reviewed adapter fingerprints and measured capability profiles
request_history.py                   # Read-only outputs scan, manifests, advisory dependency graph
storage.py                           # Fixed roots, atomic manifests, disk estimation, bundle publication
adapters/
├── __init__.py
├── base.py                          # Joint audio/video adapter protocol and capability profile
├── wan_s2v.py                       # Default reviewed profile (TTS then Wan2.2-S2V, one generate())
└── stub.py
spikes/                              # Throwaway hardware probes; NOT product code
└── wan_s2v_feasibility.py           # Loads real weights, measures both ceilings
requirements-core.txt
requirements.txt
requirements-dev.txt
pytest.ini                           # Offline default; stack_compatibility/cuda/mps/model_download markers
ruff.toml
.gitignore
.env.example                        # Non-secret runtime options; no bundle-root override
outputs/.gitkeep                     # Successful bundles ignored except placeholder
tests/
├── conftest.py                      # Fakes, fixtures, and a fixture profile with non-production values
├── contract/
│   ├── test_generation_service.py
│   ├── test_model_catalog_service.py
│   ├── test_request_history_service.py
│   └── test_ui_contract.py
├── integration/
│   ├── test_stub_end_to_end.py
│   ├── test_history_reconciliation.py
│   ├── test_inventory_restart.py
│   ├── test_model_download.py
│   ├── test_stack_compatibility.py  # Blocking: stack loads with trust_remote_code=False, no pickles
│   ├── test_long_runtime.py         # Progress + cancellation under offload; no latency assertion
│   ├── test_cuda_smoke.py
│   └── test_mps_smoke.py
└── unit/
    ├── test_model_urls.py
    ├── test_model_registry.py
    ├── test_model_updates.py
    ├── test_model_dependencies.py
    ├── test_model_license_exclusion.py
    ├── test_config.py
    ├── test_domain.py
    ├── test_inventory.py
    ├── test_duration_selection.py
    ├── test_prompting.py            # Dialogue tags, token measurement, motion-prompt truncation
    ├── test_references.py           # Two-reference rule, clip limits, video-reference rejection
    ├── test_profile_agnostic.py     # Same suite against a fixture profile with different values
    ├── test_resource_ceilings.py    # Accelerator and host RAM gates
    ├── test_disk_preflight.py
    ├── test_request_history.py
    ├── test_request_bundle.py
    ├── test_paths.py
    ├── test_devices.py
    ├── test_validation.py
    ├── test_consent.py
    ├── test_media.py
    ├── test_export.py
    └── test_errors.py
```

**Structure Decision**: Root-level `app.py` and `pipeline.py` remain as requested. Catalog, immutable
updates, single-generation planning, fixed bundle publication, and read-only history stay in separate
modules because each owns an independent state machine and failure mode. There is intentionally no
request-bundle deletion service and no in-app voice library.

Collapsing to one joint audio/video adapter **deletes** the previous worker architecture entirely:
`worker_protocol.py`, `workers/`, `requirements-latentsync-windows.txt`, the LatentSync worker contract,
and their tests are gone, along with `adapters/cogvideox.py`, `adapters/qwen3_tts.py`,
`adapters/latentsync.py`, and `adapters/native.py`. The cross-provider dependency conflict that forced
process isolation no longer exists, because there is no second heavyweight provider.

`prompting.py` is new and load-bearing: no hosted structuring service is called, so assembling the dialogue-tag
prompt, measuring it against the profile's token capacity, and recording truncation overrides is this
application's responsibility. `model_registry.py` holds every model-specific number as a measured profile
field, and `test_profile_agnostic.py` exists to fail if any of those values leak back into shared code.

## Reviewed Adapter Profiles

Every field below is a **measured profile value**, recorded per adapter and verified on the target
machine. None of them may be embedded in architecture-level invariants, shared constants, or validation
code that assumes one model. The previous stack encoded 49 frames, 8 FPS, and 226 tokens as global
truths; that mistake is not repeated.

| Adapter key | Reference model | Validated roles | Runtime | Key profile fields |
|-------------|-----------------|-----------------|---------|--------------------|
| `wan-s2v` | `Wan-AI/Wan2.2-S2V-14B` via DiffSynth, plus a TTS for voice | video + native lip sync; voice supplied by the second model (`native_capabilities={VIDEO,LIP_SYNC}`) | CUDA BF16 under DiffSynth's disk-offload path with an explicit `vram_limit`; each stage loaded and freed, nothing co-resident | 16 FPS; frame count must be **4n+1**; duration ceiling **unmeasured** (FramePack plus multi-clip removes the hard wall); audio conditioning at 16 kHz with delivery at the TTS rate; dialogue languages from the TTS; prompt/token capacity measured; video references rejected |
| `stub` | Local fixtures | all roles | CPU | No network/weights; deterministic control-path tests |

Additional reviewed profiles are added later through the unchanged adapter registry and Hugging Face URL
selection. This is the default, not the only permitted model.

### H3 facts that shape the design

> **Superseded 2026-09-01.** Retained as the record of what was established about MiniMax-H3 before it
> was abandoned. Not the current stack; see `research.md` → "Stack decision superseded".

- **Two task-specific checkpoints** exist (`FL2VA/`, `Ref2VA/`). Only the `ref2va` workflow is used,
  because it is the mode accepting an image reference plus an audio reference.
- **Loaded from the repository root, not `Ref2VA/`.** The subfolder's `model_index.json` names classes no
  upstream release exports, and its two VAE component configs carry `auto_map` entries pointing at bundled
  `.py` modules — so that path requires `trust_remote_code=True`. The root `modular_model_index.json`
  describes the same weights in Diffusers' modular format using only classes the pinned wheels export, and
  ships no Python beside its weights. `trust_remote_code` stays false on the root path.
- **Only the `ref2va` components are loaded.** `load_components(workflow="ref2va")` resolves to
  `transformer_ref`, `text_encoder`, `vae`, `audio_vae`, and the two schedulers, excluding the separate
  61.73 GiB `transformer` that only `fl2va`/`t2va` use.
- **The `ref2va` working set is 134.12 GiB at BF16**, against a 64 GB host ceiling. Quantization is
  therefore mandatory, not a fallback: roughly INT4 to be resident. The precision that fits is a spike
  measurement, not a planning assumption.
- **~20B effective inference parameters.** The transformer is 33B dense, but roughly 13B sit in AdaLN
  branches whose modulation outputs can be precomputed and cached and therefore need not be loaded for
  inference-only deployment. The text encoder additionally carries Qwen3-VL-32B weights, used only up to
  its 50th hidden layer. Both facts are load-time decisions with large memory consequences and must be
  measured, not assumed.
- **H3-Context-IR is not open-sourced.** The model card recommends it for output quality and offers a
  hosted API, which local-only operation forbids. Prompt structuring is therefore this application's work,
  built from the published Prompting Guidance, and the assembled prompt is retained per request.
- **H3-Regenerate-2K is not open-sourced**, so **768p short side is the local ceiling**. 2K is out of scope.
- **Sparse attention is not in the initial release**; inference is full-attention only, which raises memory
  on long packed multimodal sequences and is a direct argument for the low end of the duration range.
- **Video references are rejected on token cost.** A 15 s, 1280x768 reference at 24 FPS costs roughly 86k
  tokens on its own under f16t4d24 latents with 1x2x2 patchify, which breaches the memory ceiling before
  any generation begins. Image and audio references only.
- The released weights are **CFG-distilled**. The root layout supplies a real `MiniMaxH3Scheduler` for
  both `scheduler` and `audio_scheduler`; the sampling path is fixed by the checkpoint and recorded rather
  than user-configurable. (The `Ref2VA/` index's `"scheduler": null` is an artifact of the fork layout and
  does not apply to the path in use.)

The application deliberately does not inspect, show, persist, acknowledge, or enforce model-license
terms. Repository authentication/access errors remain supported; all license responsibility belongs to
the operator outside the application.

## Device and Memory Profiles

Every profile is gated on **two** ceilings. A profile that satisfies accelerator memory but breaches host
system memory is not ready, and vice versa.

| Profile | Device | Precision | Accelerator ceiling | Host RAM ceiling | Memory policy | Intended use |
|---------|--------|-----------|---------------------|------------------|---------------|--------------|
| `production` | CUDA 13.0 | BF16 compute over a reviewed quantized checkpoint | 13.5 GiB peak allocator-reserved | configured, measured against 64 GB installed | layer-wise/sequential CPU offload, component unload between phases, cached AdaLN modulation, truncated text-encoder load | RTX 5080 16 GB |
| `cuda-highmem` | CUDA | BF16/FP16 compute | 13.5 GiB peak allocator-reserved | configured | reviewed lower-compression checkpoint where host RAM allows | Explicit quality/headroom |
| `mps-experimental` | MPS | Adapter-declared | n/a | configured | no CUDA calls, constrained supported adapter | Opt-in Mac smoke |
| `cpu-experimental` | CPU | FP32 | n/a | configured | bounded small supported adapter | Correctness smoke |
| `stub` | CPU | N/A | n/a | trivial | fixture frames/waveform | Default tests/UI |

Host RAM is a first-class budget because layer-wise offload holds the resident model in system memory and
streams it to the card. On 64 GB the transformer at INT8 is comfortably resident; at BF16 it is not, which
is why a quantized checkpoint is an expected part of the production profile rather than a fallback. The
exact resident footprint per precision is a measured release-gate value.

Quantized and GGUF checkpoints are supported only through a reviewed adapter/component configuration. A
link never becomes ready solely because of its extension. No profile silently changes model, quantization,
precision, seed, duration, or effective media parameters.

## Model Catalog, Update, and Deletion Flow

1. Accept only canonical HTTPS Hugging Face repository URLs and approved revision forms; reject
   credentials, queries, fragments, file/blob paths, alternate hosts, and ambiguous routes.
2. Pin the Hub endpoint to `https://huggingface.co`, resolve repository metadata to a commit SHA, match
   a reviewed adapter without executing repository code, and never request/read license fields.
3. Resolve the complete immutable dependency closure (including reviewed auxiliary snapshots), perform
   bounded metadata/dry-run calls, and block before model-content transfer unless each destination
   filesystem preserves the configured disk reserve (10 GiB default).
4. Download every immutable commit into the application's dedicated cache. Emit progress and mark ready
   only after required-file manifests/digests, formats, revisions, adapter, dependency, device, and memory
   validation. Inference loads verified local snapshot paths with local-only behavior.
5. Persist/reconcile inventory atomically; ready snapshots remain selectable offline.
6. Persist an optional mutable `tracking_ref` separately from every pinned `resolved_commit`. Never
   poll/update automatically. `Check for updates` explicitly resolves that ref; downloading a different
   commit creates a separate inventory entry and never replaces/selects/deletes the old one. A source
   pinned only to a commit has no update check until the user establishes a tracking ref.
7. Model deletion remains an in-app confirmed action. Reject active/leased/dependency-referenced entries,
   bind confirmation to the recomputed deletion strategy, preserve shared blobs/auxiliaries, remeasure
   the dedicated cache afterward, and report only verified reclaimed bytes. Partial/incomplete-download
   cleanup is a separate confirmed app-owned operation; no broad cache prune runs automatically.

## Fixed Bundle Storage and Read-Only History

- Successful work is staged under `outputs/.work/<request-id>/` and atomically renamed to
  `outputs/<request-id>/` after verification. No setting/environment/request may redirect this root.
- The published bundle retains the original images and reference audio, the plaintext derived voice
  representation, the assembled prompt actually submitted, the decoded video and audio, the final MP4,
  and sanitized manifest/metadata.
- Failed/cancelled `.work` directories are removed. Successful bundle artifacts are never cleaned or
  expired by the application, even if redundant.
- Before inference, estimate the worst-case complete bundle plus the model operation and preserve 10 GiB
  by default. Refine the estimate once the effective duration is fixed, and monitor free space during
  every write stage.
- Request History scans only `outputs/<request-id>/manifest.json`, validates paths without following
  symlinks outside `outputs/`, and exposes preview, artifacts, size, voice origin/dependents, and state.
- History has no delete/reuse actions. Users re-upload retained audio through the regular filesystem
  picker and delete bundle directories externally.
- If a re-upload path resolves beneath a known bundle, record an advisory origin dependency. Refresh
  detects externally missing/corrupt origins and disables affected reuse; it never repairs, deletes, or
  modifies later bundles automatically.
- Before the first submission that can retain artifacts, the generation form discloses that reference
  and derived voice artifacts are ordinary unencrypted files and may be visible to filesystem
  permissions, backups, sync tools, or device users.

## Single-Generation Execution Flow

1. Gradio refreshes model inventory/history and builds a request from the still image, the reference
   recording, the motion prompt, the speech script, language, consent, requested duration preference,
   model set, and runtime profile.
2. Validate consent, fixed roots, disk reserve, every still image, exactly one face/mouth per image, all references
   against the profile's accepted types and per-clip limits, the selected language against the profile's
   stable dialogue-language set, and device/host-memory feasibility. Reject video references outright.
3. Assemble the prompt: the motion description plus the speech script embedded in the profile's dialogue
   tag form carrying the selected language. Measure the token count against the profile's capacity. If the
   motion prompt exceeds capacity, truncate only the motion portion and record original, retained, and
   discarded lengths as an explicit override. **Never truncate the speech script.**
4. Suggest a duration from the trimmed script and the profile's per-language speaking rate, clamped to
   the profile's supported range, and present it as an editable default. The operator may override it
   anywhere in that range. Because audio and video are generated jointly, duration is an **input** and
   cannot be measured from synthesized speech beforehand, so there is no pre-generation fit check and no
   rejection for script length. Record suggested and effective duration, whether it was overridden, and
   the speaking-rate field used.
5. Acquire model leases, recheck the disk reserve, and run **one** joint audio/video generation under the
   profile's declared offload and quantization policy. Video and stereo audio are produced together; there
   is no separate speech phase, lip-sync phase, timebase bridge, or post-generation face preflight.
6. Decode video and audio, export the container, and verify exactly one video stream and one non-silent
   speech stream whose duration matches the video within one frame.
7. Publish the complete bundle atomically, write immutable model/provider/consent/effective-parameter/
   memory metadata including the assembled prompt actually submitted, and release leases.
8. Technically valid output is previewable/downloadable without automated lip-sync scoring.
9. On failure or cancellation, clean the unpublished staging bundle, release leases and accelerator
   resources, and return one safe terminal error. Successful bundles remain untouched.

The heavy-stage sequence is simply
`validate -> assemble prompt -> plan duration -> one joint generation -> decode -> export -> verify -> publish`.
Collapsing three models into one removes the entire cross-provider ordering problem: there is no speech-first
constraint to enforce, no provider overlap to resolve for voice or lip roles when the default profile
supplies them natively, and no second heavyweight process to schedule.

### Reference semantics

References accepted per request:

- **One or more still images** - anchor subject identity and appearance, all of the same subject. The
  application imposes no maximum; the profile's measured reference limit is the only bound, and each
  image must independently contain exactly one usable face and mouth region. Subject consistency across
  images is the operator's responsibility.
- **Reference recording** - anchors **voice timbre only**. It is never played back, never mixed into the
  output, and never treated as spoken content. It **must say different words from the speech script**, and
  the UI states this rule at the point of upload. Spoken content comes exclusively from the script via the
  profile's dialogue tags.

Video references are rejected: their token cost alone breaches the memory ceiling before generation starts.

Because this is still voice cloning, the consent gate is unchanged. Per-request attestation defaults and
resets to false, resets whenever the reference audio changes, is bound server-side to the request ID and
the reference-audio SHA-256, and is recorded in sanitized metadata. Only the model performing the cloning
has changed.

## Validation and Error Policy

- Motion prompt: trimmed, required, no application maximum. Truncated to the effective profile's measured
  prompt/token capacity with original, retained, and discarded lengths reported as an explicit override.
  Silent truncation is a contract failure.
- Speech script: trimmed, required, no application maximum and no length-based rejection. It drives the
  suggested duration through the profile's per-language speaking rate; the operator may override that
  suggestion within the supported range. The script is never trimmed, time-stretched, truncated, or
  partially omitted by the application.
- Consent: true per request with server timestamp, bound to request ID and reference-audio SHA-256, reset
  whenever the audio changes; prior consent never carries forward to a re-upload.
- References: exactly one still image and exactly one audio recording. Validate against the profile's
  measured accepted types, clip counts, and per-clip duration bounds. Video references are rejected with a
  specific reason. A recording path under a valid bundle adds an advisory origin.
- Image/face: still image, bounded decode, EXIF/RGB normalization, exactly one usable face and mouth
  region before generation. No post-generation face stage exists.
- Language: explicit member of the effective profile's measured stable dialogue-language set.
- Seed: `0..2^63-1`; blank generates and records a secure random effective seed.
- Duration: a suggestion derived from script length and the profile's per-language speaking rate, clamped
  to the supported range and editable by the operator. Suggested, requested, and effective values are all
  recorded. There is no architecture-level duration constant and no length-based rejection.
- Models: ready immutable commits, adapter-matched, compatible interface/language/device/host-memory,
  declared offload and quantization policy satisfied; license data intentionally excluded.
- Resources: every profile is validated against both the accelerator-memory and host system-memory
  ceilings before it is selectable.
- Storage: fixed non-symlinked project `outputs/`, application-owned model cache, required bytes plus a
  configurable 10 GiB reserve, monitored during writes as well as at preflight; the bundle root cannot be
  overridden.
- Errors: stable input/consent/face/reference/language, URL/auth/download/update/inventory, disk, history,
  incompatibility, load/backend/OOM, duration, generation, export/codec, cancellation, filesystem, and
  internal codes. Logs redact prompts, tokens, absolute upload paths, and voice data.

## Testing Strategy

1. **Blocking stack-compatibility gate, before architecture locks.** Confirm the pinned Diffusers and
   Transformers releases export every class named by the root `modular_model_index.json` —
   `WanVideoPipeline`, `ModelConfig`, `wan_video_dit_s2v`, the Wav2Vec2 encoder and processor,
   `Qwen3VLProcessor`, and `Qwen2TokenizerFast` — with `trust_remote_code=False`, and that no root
   component config declares an `auto_map`. If either fails, the profile stays `incompatible`; remote code
   is never enabled as a workaround. **Status: passed** against `diffusers==0.40.0` /
   `transformers==5.16.1`. Then write failing tests for service contracts, fixed
   roots, disk estimates/reserve, prompt assembly, duration suggestion and override, commit updates,
   inventory, and history.
2. Unit-test CUDA/MPS/CPU safety, dtype/offload/quantization selection, both resource ceilings, dialogue-tag
   prompt assembly and token measurement, motion-prompt truncation overrides,
   duration suggestion from script length and per-language speaking rate including clamping and the
   missing-language fallback, operator override acceptance and out-of-range refusal, **absence** of any
   length-based rejection path, multi-image reference sets with per-image face validation and
   profile-limit enforcement, video-reference rejection, the different-words UI rule, consent binding and
   reset on audio change, language validation, metadata redaction, plaintext disclosure, and errors using
   fakes.
3. Catalog contracts cover unambiguous URL/tracking-ref parsing, fixed Hub endpoint, remote-code/kernel
   disablement, no license fields/calls even when fake metadata contains them, full pinned dependency
   closures/digests, disk preflight, interrupted download/retry/discard, explicit update check, separate
   revisions, offline-only inference, active/dependency protection, download refusal while a generation is
   active, and measured deletion.
4. History contracts scan only fixed `outputs/`, expose no mutation action, compute advisory dependencies,
   and reconcile missing/corrupt/symlinked bundles without changing retained neighbors. Path tests cover
   Windows junction/reparse points, UNC/drive/alternate-stream traversal, external deletion races, and
   bounded orphan-staging cleanup.
5. Offline end-to-end tests retain every successful artifact including the assembled prompt, verify
   MP4/audio duration agreement, exercise history and filesystem re-upload with fresh consent, and clean
   only failed/cancelled staging bundles.
6. **Profile-measurement tests assert nothing about a specific model's numbers.** Duration, FPS,
   resolution, sample rate, languages, reference limits, and token capacity are read from the profile
   under test. A fixture profile with deliberately different values must pass the same suite, which is the
   regression guard against re-baking one model's constants into the architecture.
7. CUDA acceptance runs one composite generation (speech then video) end to end, measures the real supported duration
   ceiling on this card rather than assuming 15 s, gates peak allocator-reserved accelerator memory at
   13.5 GiB and resident host memory at the configured ceiling, records allocated/reserved/free and host
   RSS, preserves the disk reserve, and verifies final and retained artifacts. Windows release also
   requires a clean install with zero hidden model downloads during inference.
8. **Long-runtime behaviour is tested, not bounded.** A generation under layer-wise offload must emit a
   monotonic completion fraction at least every few seconds, honour cancellation within the documented
   interval, and survive without a latency assertion of any kind. No test may fail a run for taking too long.
9. Negative tests cover malformed/gated/incompatible models, missing consent/face/language/audio, a
   duration override outside the profile's supported range, a video reference, more images than the
   profile allows, an image failing the face check inside an otherwise valid set, out-of-range reference
   clip counts or durations,
   an unsupported language, host-RAM-ceiling breach, low disk at preflight and mid-write, ambiguous
   providers, silent generated speech, export/codec failure, externally deleted origins, plaintext warning,
   and cancellation.

## Delivery Phases

**Status**: phases 1 and 3 are complete against the stub profile; phase 6's adapter (T040) is the
next unit of work and is blocked on hardware measurement. Phases 2, 4, 5, 7, and 8 are not started.
See [tasks.md](tasks.md) for the task-level state.


1. **Dependency/architecture gate**: verify published wheels, resolve and install the exact environment,
   prove the model-stack classes load with `trust_remote_code=False`, measure resident footprint per precision
   against both ceilings, then freeze contracts. Follow with failing foundation tests, domain/errors/
   settings, fixed roots, reserve, manifests, and service protocols.
2. **Model catalog**: URL inspection, immutable download, inventory, manual update-as-new-revision, leases,
   confirmed model deletion, reclaimed size, download refusal during an active generation, no license
   handling.
3. **Offline single-generation slice**: stubs, consent validation, multi-image per-image face validation,
   reference and language validation, dialogue-tag prompt assembly and token measurement, motion-prompt
   truncation overrides, duration suggestion and operator override, retained bundle publication, final
   MP4, complete successful artifacts, and failed staging cleanup.
4. **Read-only history**: scanner, schema, preview/artifact/size/dependency views, external-deletion
   reconciliation, filesystem voice re-upload and fresh consent.
5. **Device/memory/disk**: capability resolution, CUDA/MPS/CPU safety, layer-wise and sequential offload,
   reviewed quantization, dual-ceiling gating, allocated/reserved/free and host-RSS metrics, two-stage disk
   preflight with mid-write monitoring, OOM and disk recovery.
6. **Production adapter**: the `wan-s2v` profile with measured duration/FPS/resolution/sample-rate/
   language/reference/token fields, locally built prompt structuring per the published Prompting Guidance,
   immutable fingerprints, and the profile-driven test suite that keeps those values out of the architecture.
7. **Hardening/platform acceptance**: malicious paths/repos, secrets, interruption and cancellation of a
   multi-hour run, macOS stub/MPS, Windows PyTorch 2.13/CUDA 13.0 RTX 5080 clean-install, dual-ceiling
   memory tests, and measurement of the real duration ceiling on this hardware.
8. **Documentation/release**: bounded requirements, quickstart, plaintext/retention/filesystem-history
   disclosure, the timbre-anchor and different-words rule, operator-only license responsibility,
   troubleshooting, and a measured baseline recorded without any SLA.

## Complexity Tracking

No constitutional violations require justification. Collapsing three heavyweight providers into one joint
audio/video adapter removes substantial complexity: the local worker-process boundary, the separately
locked worker dependency stack, the cross-provider timebase bridge, the generated-face preflight, and the
entire loop architecture are all deleted. What remains separate -- catalog, bundle publisher, history
scanner, adapter registry -- stays separate because each owns an independent state machine and failure mode.

Two deliberate tensions are recorded rather than waived:

- **Unbounded inference time** versus the constitution's bounded-inference principle. The constitution
  requires bounded *allocation* and reproducible parameters, both of which the dual-ceiling gates and the
  recorded effective profile preserve. Wall-clock time is explicitly out of scope, by product decision.
- **Locally built prompt structuring** replacing any hosted structuring service. The model card attributes
  meaningful output quality to that module, and it is not open-sourced. This is an accepted quality risk
  with a measurable owner: the assembled prompt is retained per request so results stay explainable and
  the structuring can be improved without changing the contract.
