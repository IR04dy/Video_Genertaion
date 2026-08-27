# Implementation Plan: Generate Image-Conditioned Lip-Synced Video

**Branch**: `001-generate-image-video` | **Date**: 2026-08-27 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-generate-image-video/spec.md`

## Summary

Build a local, single-user Python 3.11 application that accepts a still image, motion prompt, speech
script, reference-voice file, language, and per-request voice-consent attestation. It resolves a
validated set of immutable Hugging Face model revisions, synthesizes the full speech first, derives a
compatible video frame count/FPS from the speech duration, generates motion, performs lip
synchronization, and returns a browser-playable MP4.

The reference CUDA stack is CogVideoX-5B-I2V, Qwen3-TTS-12Hz-1.7B-Base, and LatentSync 1.5 behind
reviewed capability adapters. Native voice/lip video adapters are supported only when they expose a
speech-first two-phase contract. Successful requests retain every input, derived voice artifact,
speech file, intermediate video, final MP4, and metadata as ordinary unencrypted files beneath fixed
`outputs/<request-id>/`. A read-only Request History scans that directory; reuse and deletion remain
external filesystem actions. The application neither inspects nor records model-license information.

## Technical Context

**Language/Version**: Python 3.11 (`>=3.11,<3.13`)

**Primary Dependencies**: PyTorch 2.x, Diffusers, Transformers 4.57.x-compatible stack, Accelerate,
Hugging Face Hub, Safetensors, Qwen-TTS, Gradio, Pillow, NumPy, SoundFile/librosa, imageio,
imageio-ffmpeg, Pydantic 2.x, psutil, filelock; adapter-specific face preprocessing and optional
TorchAO/compatible quantization extras

**Storage**: Fixed project `outputs/` for complete successful request bundles and read-only history;
fixed application-owned model cache/inventory separate from the global Hub cache; atomic JSON manifests
and locks; ordinary unencrypted files; no database and no configurable bundle root in v1

**Testing**: pytest, pytest-cov, fake Hub/model/media adapters for offline tests; opt-in `mps`, `cuda`,
and `model_download` markers for real backends

**Target Platform**: Development/control-path testing on macOS 13+; production on 64-bit Windows 11
with NVIDIA RTX 5080 16 GB, a current driver, and official PyTorch CUDA 12.8-or-newer wheels

**Project Type**: Local browser-based Python application with in-process generation, model catalog,
and read-only request-history services

**Performance Goals**: Stub UI available within 5 seconds; at least one status event per phase; model
download status at least every 2 seconds or completed chunk; each default heavy CUDA stage below
15.5 GiB peak allocated memory; final audio/video duration differs by no more than one frame; preserve
a configurable disk reserve defaulting to 10 GiB; measure target latency instead of inventing an SLA

**Constraints**: One active generation; speech duration authoritative; no speech trimming/time-
stretching; immutable commit-pinned models and manual update checks; no remote code; no runtime model-
license handling; complete role coverage and explicit Native/Dedicated choices; exactly one face;
provider-specific language/audio validation; fixed `outputs/`; full successful-bundle retention;
filesystem-only bundle reuse/deletion; read-only advisory dependency history; plaintext artifacts;
loopback binding; no automated lip-sync quality gate; offline stub tests download no weights

**Scale/Scope**: One local operator, one generation at a time, three model roles, multiple immutable
model revisions, model-specific frame/FPS ranges derived from speech, one speech stream and one complete
retained bundle per success

## Constitution Check

*GATE: Passed before Phase 0 and re-checked after Phase 1 design.*

| Gate | Pre-research | Post-design evidence |
|------|--------------|----------------------|
| Cross-platform/device safety | PASS | `pathlib.Path`, fixed project-relative roots, CUDA/MPS/CPU resolver, adapter device declarations, platform quickstart |
| Inference/UI separation | PASS | `GenerationService`, `ModelCatalogService`, and `RequestHistoryService` isolate Gradio from tensors, Hub, and storage scans |
| Bounded/reproducible inference | PASS | Commit-pinned models, speech-first effective parameters, seeded plans, sequential stages, leases, memory/disk preflight |
| Test-first reliability | PASS | Failing offline unit/contract tests precede adapters; real network/MPS/CUDA tests are opt-in |
| Observable/local operation | PASS | Progress, redacted logs, CUDA metrics, fixed local bundles, explicit plaintext/retention disclosure, no public sharing |

Plaintext storage is an explicit product decision, not a constitutional exception: the constitution
requires local handling and disclosure but does not mandate application-layer encryption. No
constitutional exceptions are required.

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
└── tasks.md                         # Regenerated later by $speckit-tasks
```

### Source Code (repository root)

```text
app.py                               # Gradio Blocks, read-only history, event mapping
pipeline.py                          # Public VideoGenerationEngine facade
config.py                            # Bounds, fixed roots, reserve, runtime settings
domain.py                            # Pydantic requests, models, plans, bundles, events/results
devices.py                           # CUDA/MPS/CPU resolution, dtype, memory snapshots
errors.py                            # Stable safe domain errors
logging_config.py                    # Redacted structured logging
media.py                             # Image/audio inspection, face preflight, retained-path detection
export.py                            # Frame export, AAC mux, stream/duration validation
execution.py                         # Speech-first artifact DAG and sequential provider planner
model_catalog.py                     # URL inspection, immutable downloads/updates, inventory/deletion
model_registry.py                    # Reviewed adapter fingerprints and capability resolution
request_history.py                   # Read-only outputs scan, manifests, advisory dependency graph
storage.py                           # Fixed roots, atomic manifests, disk estimation, bundle publication
adapters/
├── __init__.py
├── base.py                          # Video, voice, lip-sync, native two-phase protocols
├── cogvideox.py
├── qwen3_tts.py
├── latentsync.py
├── native.py
└── stub.py
requirements.txt
requirements-dev.txt
.env.example                        # Non-secret runtime options; no bundle-root override
outputs/.gitkeep                     # Successful bundles ignored except placeholder
tests/
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
│   ├── test_cuda_smoke.py
│   └── test_mps_smoke.py
└── unit/
    ├── test_model_urls.py
    ├── test_model_registry.py
    ├── test_model_updates.py
    ├── test_inventory.py
    ├── test_execution_plan.py
    ├── test_duration_planning.py
    ├── test_disk_preflight.py
    ├── test_request_history.py
    ├── test_devices.py
    ├── test_validation.py
    ├── test_consent.py
    ├── test_media.py
    ├── test_export.py
    └── test_errors.py
```

**Structure Decision**: Root-level `app.py` and `pipeline.py` remain as requested. Catalog, immutable
updates, speech-first planning, fixed bundle publication, and read-only history have separate modules
because they have independent state machines and failure modes. There is intentionally no request-
bundle deletion service or in-app voice library.

## Reviewed Adapter Profiles

| Adapter key | Reference model | Validated roles | Runtime | Key constraints |
|-------------|-----------------|-----------------|---------|-----------------|
| `cogvideox-i2v` | `zai-org/CogVideoX-5b-I2V` | video | CUDA BF16 | 720x480 default; adapter-supported temporal values; offload and VAE slicing/tiling |
| `qwen3-tts-base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | voice | CUDA BF16; CPU smoke | Provider language/audio rules; x-vector default avoids required reference transcript |
| `latentsync-1.5` | `ByteDance/LatentSync-1.5` | lip sync | CUDA FP16/BF16 as validated | Exactly one usable face; 256px profile; documented 8 GB inference minimum |
| `native-composite` | Reviewed registered model | declared video/voice/lip subset | Adapter-specific | Must expose speech artifact before video inference and validate duration/device/memory contracts |
| `stub` | Local fixtures | all roles | CPU | No network/weights; deterministic control-path tests |

LatentSync 1.6 remains incompatible with the 16 GB production profile because its documented minimum
is 18 GB. The application deliberately does not inspect, show, persist, acknowledge, or enforce model-
license terms. Repository authentication/access errors remain supported; all license responsibility
belongs to the operator outside the application.

## Device and Memory Profiles

| Profile | Device | Precision | Memory policy | Intended use |
|---------|--------|-----------|---------------|--------------|
| `production` | CUDA | BF16 preferred; validated FP16 fallback | one provider resident, model CPU offload, VAE slicing/tiling, unload between stages | RTX 5080 |
| `cuda-quantized` | CUDA | BF16/FP16 compute | reviewed component quantization plus offload | Explicit recovery/headroom |
| `mps-experimental` | MPS | Adapter-declared | no CUDA calls, constrained supported adapter | Opt-in Mac smoke |
| `cpu-experimental` | CPU | FP32 | bounded small supported adapter | Correctness smoke |
| `stub` | CPU | N/A | fixture frames/waveform | Default tests/UI |

GGUF is supported only through a reviewed adapter/component configuration. A GGUF link never becomes
ready solely because of its extension. No profile silently changes model, provider, quantization,
precision, seed, or effective media parameters.

## Model Catalog, Update, and Deletion Flow

1. Accept only canonical HTTPS Hugging Face repository URLs and approved revision forms; reject
   credentials, queries, fragments, file/blob paths, alternate hosts, and ambiguous routes.
2. Resolve repository metadata to a commit SHA, match a reviewed adapter without executing repository
   code, and ignore all license metadata.
3. Estimate snapshot bytes and block before network transfer unless completion preserves the configured
   disk reserve (10 GiB default). Report required/available space and manual cleanup guidance.
4. Download the immutable commit into the application's dedicated cache. Emit progress and mark ready
   only after required-file, format, revision, checksum, adapter, device, and memory validation.
5. Persist/reconcile inventory atomically; ready snapshots remain selectable offline.
6. Never poll/update automatically. `Check for updates` explicitly resolves the current upstream commit;
   downloading it creates a separate inventory entry and never replaces/selects/deletes the old one.
7. Model deletion remains an in-app confirmed action. Reject active/leased entries, use the Hub cache
   revision strategy for shared blobs, and report measured reclaimed bytes.

## Fixed Bundle Storage and Read-Only History

- Successful work is staged under `outputs/.work/<request-id>/` and atomically renamed to
  `outputs/<request-id>/` after verification. No setting/environment/request may redirect this root.
- The published bundle retains original image/reference audio, plaintext derived voice representation,
  synthesized speech, pre/post-lip intermediate media, final MP4, and sanitized manifest/metadata.
- Failed/cancelled `.work` directories are removed. Successful bundle artifacts are never cleaned or
  expired by the application, even if redundant.
- Before inference, estimate the worst-case complete bundle plus current model operation and preserve
  10 GiB by default. After speech synthesis, refine the estimate before video inference.
- Request History scans only `outputs/<request-id>/manifest.json`, validates paths without following
  symlinks outside `outputs/`, and exposes preview, artifacts, size, voice origin/dependents, and state.
- History has no delete/reuse actions. Users re-upload retained audio through the regular filesystem
  picker and delete bundle directories externally.
- If a re-upload path resolves beneath a known bundle, record an advisory origin dependency. Refresh
  detects externally missing/corrupt origins and disables affected reuse; it never repairs, deletes, or
  modifies later bundles automatically.
- The first retention view discloses that reference and derived voice artifacts are ordinary
  unencrypted files and may be visible to filesystem permissions, backups, sync tools, or device users.

## Speech-First Execution Flow

1. Gradio refreshes model inventory/history and builds a request from uploads, prompts, language,
   consent, preferred video controls, model set, and explicit provider choices.
2. Validate consent, fixed roots, disk reserve, still image, exactly one face/mouth, prompts, selected
   providers, reference audio/language constraints, and device feasibility before any inference.
3. Acquire model leases. Synthesize the complete speech first (dedicated or native voice phase), verify
   finite non-silent audio, and persist it in the staging bundle.
4. Use exact speech duration and video-adapter temporal constraints to select effective frame count/FPS
   within one-frame tolerance. Requested values are preferences. If no valid combination exists, fail
   before video inference without trimming, stretching, or omitting speech.
5. Recheck disk space with exact duration, load/generate video, unload, then load/run lip sync. A native
   adapter may combine later stages but must still expose speech before video allocation.
6. Mux/verify one video plus one non-silent AAC speech stream, publish the complete bundle atomically,
   write immutable model/provider/consent/effective-parameter/memory metadata, and release leases.
7. Technically valid output is previewable/downloadable without automated lip-sync scoring.
8. On failure/cancellation, clean the unpublished staging bundle, release leases/resources, and return
   one safe terminal error. Successful bundles remain untouched.

## Validation and Error Policy

- Motion prompt/speech script: trimmed, required, maximum 2,000 characters.
- Consent: true per request with server timestamp; prior consent never carries to a re-upload.
- Reference voice: selected through the filesystem picker; provider-specific format/duration/sample-
  rate/channels/speaker/transcript/quality constraints. A path under a valid bundle adds advisory origin.
- Image/face: still image, bounded decode, EXIF/RGB normalization, exactly one usable face before voice
  or video inference.
- Language: explicit member of effective voice provider's supported set.
- Seed: `0..2^63-1`; blank generates and records a secure random effective seed.
- Video controls: preferred frame count/FPS and guidance validated; exact speech duration selects the
  effective adapter-compatible temporal combination.
- Models: ready immutable commits, complete role coverage, compatible interfaces/language/device/memory,
  explicit Native/Dedicated choice for overlaps; license data intentionally excluded.
- Storage: fixed non-symlinked project `outputs/`, application-owned model cache, required bytes plus
  configurable 10 GiB reserve; request bundle root cannot be overridden.
- Errors: stable input/consent/face/language, URL/auth/download/update/inventory, disk, history,
  incompatibility, load/backend/OOM, speech/duration/lip-sync, mux/codec, cancellation, filesystem, and
  internal codes. Logs redact prompts, tokens, absolute upload paths, and voice data.

## Testing Strategy

1. Write failing tests first for service contracts, fixed roots, disk estimates/reserve, speech-first
   ordering, duration derivation, commit updates, inventory leases/deletion, and read-only history.
2. Unit-test CUDA/MPS/CPU safety, dtype/offload, image/audio/face/consent/language validation, provider
   coverage, native two-phase adapters, metadata redaction, plaintext disclosure, and errors using fakes.
3. Catalog contracts cover URL/revision parsing, no license fields/calls, interrupted download/retry,
   explicit update check, separate revisions, offline restart, active-model protection, and deletion.
4. History contracts scan only fixed `outputs/`, expose no mutation action, compute advisory dependencies,
   and reconcile missing/corrupt/symlinked bundles without changing retained neighbors.
5. Offline end-to-end tests retain every successful artifact, verify MP4/audio duration, history and
   filesystem re-upload with fresh consent, and clean only failed/cancelled staging bundles.
6. CUDA acceptance runs Qwen3-TTS -> duration planning -> CogVideoX -> LatentSync sequentially, records
   each peak below 15.5 GiB, preserves disk reserve, and verifies final/retained artifacts.
7. Negative tests cover malformed/gated/incompatible models, missing consent/face/language/audio,
   overlong speech, low disk, ambiguous providers, silent speech, technical lip failure, codec/path
   failure, externally deleted origins, plaintext warning, and cancellation.

## Delivery Phases

1. **Foundation/contracts**: failing tests, domain/errors/settings, fixed roots, reserve, manifests,
   structured redacted logs, service protocols.
2. **Model catalog**: URL inspection, immutable download, inventory, manual update-as-new-revision,
   leases, confirmed model deletion, reclaimed size, no license handling.
3. **Offline speech-first slice**: stubs, consent/face/audio validation, duration planner, retained bundle
   publication, final MP4, complete successful artifacts, failed staging cleanup.
4. **Read-only history**: scanner, schema, preview/artifact/size/dependency views, external-deletion
   reconciliation, filesystem voice re-upload and fresh consent.
5. **Device/memory/disk**: capability resolution, CUDA/MPS/CPU safety, sequential residency, metrics,
   two-stage disk preflight, OOM/disk recovery.
6. **Production adapters**: Qwen3-TTS, CogVideoX, LatentSync 1.5, native two-phase protocol, immutable
   fingerprints, adapter-specific input/temporal constraints.
7. **Hardening/platform acceptance**: malicious paths/repos, secrets, interruption/cancellation,
   macOS stub/MPS, Windows CUDA 12.8+ RTX 5080 memory and clean-install tests.
8. **Documentation/release**: bounded requirements, quickstart, plaintext/retention/filesystem-history
   disclosure, operator-only license responsibility, troubleshooting and measured baseline.

## Complexity Tracking

No constitutional violations require justification. Adapter, catalog, speech-first planner, bundle
publisher, and history scanner boundaries are necessary for immutable multi-role models, duration-
dependent execution, full retained artifacts, and external filesystem mutation.
