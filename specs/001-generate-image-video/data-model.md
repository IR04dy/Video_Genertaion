# Data Model: Generate Image-Conditioned Lip-Synced Video

The application uses Pydantic records in memory and versioned atomic JSON files on disk. It has no
database in v1. All paths are `pathlib.Path` values internally and serialize as app-root-relative
paths when persisted.

## Enumerations

| Name | Values |
|------|--------|
| `ModelRole` | `video`, `voice`, `lip_sync` |
| `ProviderMode` | `native`, `dedicated` |
| `ModelState` | `inspecting`, `downloading`, `verifying`, `ready`, `incompatible`, `failed`, `deleting` |
| `DownloadState` | `queued`, `inspecting`, `downloading`, `verifying`, `ready`, `failed`, `cancelled` |
| `RequestState` | `queued`, `validating`, `planning`, `running`, `exporting`, `complete`, `failed`, `cancelled` |
| `StageKind` | `video`, `voice`, `lip_sync`, `native_composite`, `mux`, `verify`, `metadata` |
| `DeviceKind` | `cuda`, `mps`, `cpu` |

## ModelSource

Normalized model input submitted through the UI.

| Field | Type | Validation / meaning |
|-------|------|----------------------|
| `url` | string | HTTPS `huggingface.co/<owner>/<repo>` or approved revision form; no credentials/query/fragment |
| `repo_id` | string | Exactly two safe path components after normalization |
| `requested_revision` | string/null | Optional branch, tag, or commit supplied by user |
| `resolved_commit` | 40-character SHA | Required before download/inventory identity is finalized |
| `requested_role` | `ModelRole` | Role under which the user submitted the link |

Identity is `(repo_id, resolved_commit)`, not a mutable branch name.

## ModelProfile

Validated capability manifest produced by an installed reviewed adapter.

| Field | Type | Validation / meaning |
|-------|------|----------------------|
| `adapter_key` | string | Installed unique adapter identifier |
| `roles` | non-empty set[`ModelRole`] | Capabilities proven by the adapter fingerprint |
| `native_capabilities` | set[`ModelRole`] | Voice/lip roles supplied by the same video provider |
| `pipeline_class` | string | Reviewed class/interface, never arbitrary repository code |
| `supported_devices` | set[`DeviceKind`] | At least one runtime |
| `dtype_policy` | mapping | Preferred and allowed dtype per device |
| `memory_profiles` | list | Offload, VAE, quantization, expected peak, minimum VRAM |
| `video_constraints` | record/null | Dimensions, frame predicate/range/default, FPS, guidance |
| `speech_constraints` | record/null | Languages and reference-audio constraints |
| `input_contract` | record | Required artifacts and optional reference transcript |
| `output_contract` | record | Produced video/audio/alignment artifacts |
| `weight_policy` | record | Allowed extensions, required files, hashes/exceptions |
| `license` | record | License identifier, access/gating, user notice |

`native_capabilities` must be a subset of `roles`, and a profile claiming native voice or lip sync
must also claim `video`.

## DownloadedModel

Persistent inventory entry for one immutable model revision.

| Field | Type | Validation / meaning |
|-------|------|----------------------|
| `model_id` | UUID | Stable application identity |
| `source` | `ModelSource` | Repository and resolved commit |
| `profile` | `ModelProfile`/null | Present after successful inspection |
| `state` | `ModelState` | Lifecycle state |
| `snapshot_path` | Path/null | Must remain beneath application cache root |
| `size_bytes` | integer | Non-negative measured logical snapshot size |
| `owned_cache` | boolean | Must be true for app deletion eligibility |
| `active` | boolean | Selected in current UI/model set |
| `lease_count` | integer | Number of active runtime leases; non-negative |
| `created_at`, `updated_at` | UTC datetime | Lifecycle timestamps |
| `last_used_at` | UTC datetime/null | Updated when a lease is acquired |
| `failure` | `ErrorDetail`/null | Sanitized incompatibility/download/deletion detail |

Ready requires a profile, existing verified snapshot path, matching commit, and complete required
files. A model is deletable only when ready/failed/incompatible, inactive, `lease_count == 0`, and
its cache is application-owned.

## ModelDownload

Ephemeral/persisted operation record used for progress and restart reconciliation.

| Field | Type | Meaning |
|-------|------|---------|
| `download_id` | UUID | Operation identity |
| `model_id` | UUID | Target inventory entry |
| `state` | `DownloadState` | Current operation state |
| `files_total`, `files_complete` | integer/null | File progress when known |
| `bytes_total`, `bytes_received` | integer/null | Byte progress when known |
| `retry_count` | integer | Non-negative attempts |
| `started_at`, `updated_at`, `finished_at` | UTC datetime/null | Operation timing |
| `error` | `ErrorDetail`/null | Terminal safe detail |

Fractions cannot decrease within one attempt. Only `ready`, `failed`, and `cancelled` are terminal.

## ModelLease

Protects an immutable revision while selected for active inference.

| Field | Type | Meaning |
|-------|------|---------|
| `lease_id` | UUID | Unique lease |
| `model_id` | UUID | Leased model |
| `request_id` | UUID | Owning generation |
| `acquired_at` | UTC datetime | Audit time |
| `released_at` | UTC datetime/null | Set exactly once |

Acquisition increments `lease_count` atomically; release decrements it even after failure/cancellation.

## ModelSet

Validated provider selection for one generation.

| Field | Type | Validation / meaning |
|-------|------|----------------------|
| `video_model_id` | UUID | Required ready model with video role |
| `voice_model_id` | UUID/null | Dedicated voice model when selected/required |
| `lip_sync_model_id` | UUID/null | Dedicated lip model when selected/required |
| `voice_provider` | `ProviderMode`/null | Required only when native and dedicated voice overlap |
| `lip_sync_provider` | `ProviderMode`/null | Required only when native and dedicated lip overlap |
| `resolved_providers` | mapping role -> model UUID | Complete unambiguous coverage after validation |
| `compatibility` | record | Language, artifact, device, license, memory checks |

Every accepted set resolves video, voice, and lip-sync exactly once. Dedicated IDs may be absent only
when the video model supplies and the effective choice uses the corresponding native capability.

## GenerationRequest

Validated UI input for exactly one generation.

| Field | Type | Validation / meaning |
|-------|------|----------------------|
| `request_id` | UUID | Server-generated collision-safe identity |
| `image_path` | Path | Required local still image beneath upload workspace |
| `motion_prompt` | string | Trimmed, 1-2,000 characters |
| `speech_script` | string | Trimmed, 1-2,000 characters |
| `reference_audio_path` | Path | Required local file; provider-specific validation |
| `reference_transcript` | string/null | Used only when effective voice profile allows/requires it |
| `language` | string | Explicit member of effective voice profile language list |
| `voice_consent_confirmed` | boolean | Must be true before inference |
| `voice_consent_at` | UTC datetime | Server-assigned when the accepted request is created |
| `model_set` | `ModelSet` | Fully resolved before allocation |
| `seed` | integer/null | `0..2^63-1`; null generates and records effective value |
| `num_frames` | integer | Selected video profile predicate/range |
| `fps` | integer | 7-16 and selected profile range |
| `guidance_scale` | finite number | Selected profile bounds |
| `runtime_profile` | string | Compatible memory/device profile |

## VoiceReference

Request-scoped interpretation of uploaded reference audio.

| Field | Type | Meaning |
|-------|------|---------|
| `request_id` | UUID | Owning request |
| `path` | Path | Temporary application-owned path |
| `format`, `sample_rate`, `channels`, `duration_seconds` | scalar | Measured media properties |
| `speaker_result` | record | Provider-specific single-speaker/quality result |
| `transcript` | string/null | Optional model-specific reference text |
| `consent_confirmed`, `consent_at` | boolean/datetime | Per-request attestation |
| `cleanup_state` | enum | `retained`, `deleted`, `failed` |

## RuntimeProfile

| Field | Type | Meaning |
|-------|------|---------|
| `name` | string | Stable selected profile |
| `device` | `DeviceKind` | Capability-checked effective device |
| `dtype` | string | Effective precision for current stage |
| `offload`, `vae_slicing`, `vae_tiling` | boolean | Memory flags |
| `quantization` | string/null | Reviewed optional mode |
| `memory_limit_bytes` | integer/null | Acceptance ceiling |
| `warnings` | list[string] | Safe limitations shown to user |

## ExecutionPlan and ExecutionStage

Immutable stage graph compiled before any model allocation.

| Field | Type | Meaning |
|-------|------|---------|
| `ExecutionPlan.request_id` | UUID | Owning request |
| `effective_seed` | integer | Recorded seed |
| `resolved_models` | mapping | Role -> model ID, repo ID, commit, adapter |
| `provider_choices` | mapping | Effective Native/Dedicated decisions |
| `stages` | ordered list[`ExecutionStage`] | Topologically sorted heavy and media stages |
| `output_dir` | Path | `outputs/<request-id>` beneath configured root |

Each `ExecutionStage` stores `stage_id`, `kind`, `provider_model_id` or null, input artifact names,
output artifact names, runtime profile, and unload-after-stage flag. No two heavy model stages overlap.

## Media Artifacts

### VideoArtifact

Request ID, path, frame count, FPS, width/height, codec/container status, duration, source stage, and
cleanup/publication state.

### SpeechArtifact

Request ID, path/waveform metadata, script hash, language, voice model/revision, sample rate, channels,
duration, finite/non-silence checks, normalization result, source stage, and cleanup state.

### LipSyncAlignment

Request ID, provider/revision, face/mouth target, technical processing state, output video artifact,
user-review state (`pending`, `reviewed`), and technical failure detail. It contains no automated
quality acceptance score in v1.

## ProgressEvent and MemorySnapshot

| Field | Type | Meaning |
|-------|------|---------|
| `request_id` | UUID | Owning request/download |
| `phase` | string | Sanitized phase identifier |
| `fraction` | number/null | Monotonic `0.0..1.0` when measurable |
| `message` | string | Safe user-facing status |
| `memory` | `MemorySnapshot`/null | Device metrics |
| `timestamp` | UTC datetime | Event ordering |

`MemorySnapshot` contains availability, device name, allocated/reserved/peak bytes when CUDA is
active, and an unavailable reason otherwise.

## GenerationResult

| Field | Type | Meaning |
|-------|------|---------|
| `request_id` | UUID | Request identity |
| `state` | terminal `RequestState` | `complete`, `failed`, or `cancelled` |
| `video_path` | Path/null | Verified final MP4 only on success |
| `metadata_path` | Path | Sanitized terminal metadata |
| `execution_plan` | record | Effective providers, revisions, choices, and parameters |
| `duration_seconds` | number | Wall-clock duration |
| `memory_by_stage` | mapping | Stage peak snapshots |
| `error` | `ErrorDetail`/null | Safe failure data |

## ErrorDetail

| Field | Type | Meaning |
|-------|------|---------|
| `code` | enum | `validation`, `consent`, `face`, `language`, `model_url`, `model_access`, `model_download`, `model_incompatible`, `inventory`, `model_load`, `unsupported_backend`, `oom`, `speech`, `lip_sync`, `mux`, `codec`, `cancelled`, `filesystem`, `internal` |
| `message` | string | Actionable and sanitized |
| `retryable` | boolean | Whether a changed/repeated action may succeed |
| `suggestions` | list[string] | Ordered recovery actions |

## State Transitions

### Generation

```text
queued -> validating -> planning -> running -> exporting -> complete
   |          |            |          |           |
   +----------+------------+----------+-----------+-> failed
   +----------+------------+----------+-----------+-> cancelled
```

### Model download

```text
queued -> inspecting -> downloading -> verifying -> ready
   |          |              |             |
   +----------+--------------+-------------+-> failed
   +----------+--------------+-------------+-> cancelled
```

### Model deletion

```text
ready/incompatible/failed -> deletion preview -> confirmed -> deleting -> removed
                                      |             |
                                      +-> cancelled +-> prior state with failure detail
```

## Cross-Entity Invariants

- A generation has exactly one terminal state and never publishes a partial output.
- `video_path` is visible only after container, stream, non-silence, and duration verification.
- Every successful request records true consent, explicit language, immutable model commits, effective
  provider choices, seed, parameters, device/dtype, and per-stage memory.
- Every accepted model set covers all three roles once; overlaps require an explicit choice.
- A model cannot be deleted while active or leased. Every acquired lease is released in `finally`.
- Inventory and metadata writes use temporary files plus atomic replace under a cross-platform lock.
- Repository tokens, raw prompts, uploaded absolute paths, and derived voice representations never
  appear in normal logs or inventory records.
- Temporary image, voice, speech, silent video, and partial mux artifacts are deleted after success,
  failure, or cancellation according to the retention policy.
