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
| `StageKind` | `voice`, `duration_plan`, `video`, `video_resample`, `lip_sync`, `native_composite`, `mux`, `verify`, `metadata`, `publish` |
| `DeviceKind` | `cuda`, `mps`, `cpu` |
| `BundleAvailability` | `available`, `missing`, `corrupt`, `unsafe` |
| `ArtifactKind` | `original_image`, `reference_audio`, `derived_voice`, `speech`, `video_pre_lip`, `video_post_lip`, `final_mp4`, `metadata` |

## ModelSource

Normalized model input submitted through the UI.

| Field | Type | Validation / meaning |
|-------|------|----------------------|
| `url` | string | HTTPS `huggingface.co/<owner>/<repo>` root or `/tree/<40-sha>` form; no credentials/query/fragment |
| `repo_id` | string | Exactly two safe path components after normalization |
| `tracking_ref` | string/null | Optional mutable branch/tag used only for explicit update checks |
| `requested_commit` | 40-character SHA/null | Commit embedded in the URL; mutually exclusive with `tracking_ref` |
| `resolved_commit` | 40-character SHA | Required before download/inventory identity is finalized |
| `requested_role` | `ModelRole` | Role under which the user submitted the link |

Identity is unique on `(repo_id, resolved_commit)`, not a mutable ref. A duplicate inspection/download
returns the existing entry/operation. A commit-only source is not update-trackable until the user later
establishes a tracking ref explicitly.

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
| `auxiliary_sources` | list[`ModelDependency`] | Complete pinned VAE/tokenizer/Whisper/etc. closure |
| `worker_profile` | record/null | Versioned local worker/runtime/timebase contract when isolated |

`native_capabilities` must be a subset of `roles`, and a profile claiming native voice or lip sync
must also claim `video`. License identifiers, terms, URLs, acknowledgement state, and compatibility
judgments are intentionally absent from this model.

## ModelDependency and RequiredFile

`ModelDependency` stores dependency ID, purpose, repository ID, immutable commit, local snapshot path,
reference count, and required-file manifest. `RequiredFile` stores cache-relative path, byte size,
trusted Hub/LFS digest when available, locally computed SHA-256, allowed format, and whether it is the
exact reviewed tensor-only `.pt` exception. Dependencies are leased/ref-counted with their owning
models; inference loads every path locally with `local_files_only=True` and performs no Hub call.

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
| `physical_bytes` | integer | Measured cache bytes attributable after shared-blob accounting |
| `required_files` | list[`RequiredFile`] | Persisted verification manifest |
| `dependency_ids` | list[UUID] | Complete ready auxiliary closure |
| `owned_cache` | boolean | Must be true for app deletion eligibility |
| `active` | boolean | Selected in current UI/model set |
| `lease_count` | integer | Number of active runtime leases; non-negative |
| `created_at`, `updated_at` | UTC datetime | Lifecycle timestamps |
| `last_used_at` | UTC datetime/null | Updated when a lease is acquired |
| `failure` | `ErrorDetail`/null | Sanitized incompatibility/download/deletion detail |

Ready requires a profile, existing verified snapshot path, matching commit, and complete required
files/dependencies. A model is deletable only when ready/failed/incompatible, inactive,
`lease_count == 0`, no other model references its exclusive dependencies, and its cache is app-owned.

## ModelDownload

Ephemeral/persisted operation record used for progress and restart reconciliation.

| Field | Type | Meaning |
|-------|------|---------|
| `download_id` | UUID | Operation identity |
| `model_id` | UUID | Target inventory entry |
| `state` | `DownloadState` | Current operation state |
| `files_total`, `files_complete` | integer/null | File progress when known |
| `bytes_total` | integer | Missing content bytes fixed by accepted dry-run preflight |
| `bytes_received` | integer | Monotonic transferred bytes for the current attempt |
| `retry_count` | integer | Non-negative attempts |
| `disk_preflight` | `DiskPreflight` | Accepted before model-content transfer |
| `started_at`, `updated_at`, `finished_at` | UTC datetime/null | Operation timing |
| `error` | `ErrorDetail`/null | Terminal safe detail |

Fractions cannot decrease within one attempt. Only `ready`, `failed`, and `cancelled` are terminal.

## DiskPreflight

| Field | Type | Meaning |
|-------|------|---------|
| `operation` | enum | `model_download`, `generation_initial`, `generation_refined`, `partial_discard` |
| `filesystem_id` | string | Destination volume identity without exposing a full path |
| `bytes_to_download` | integer | Missing model content after bounded metadata/dry-run inspection |
| `bundle_or_staging_bytes` | integer | Conservative retained/staging estimate |
| `overhead_bytes` | integer | Manifest, filesystem, and atomic-staging allowance |
| `required_bytes` | integer | Bytes operation needs before reserve |
| `available_bytes` | integer | Current free bytes on destination filesystem |
| `reserve_bytes` | integer | Configured margin; default `10 * 1024^3` |
| `passes` | boolean | `available_bytes >= required_bytes + reserve_bytes` |
| `manual_guidance` | list[string] | No automatic deletion action |

When cache and `outputs/` share a filesystem, requirements are aggregated once; otherwise each volume
must pass separately. Inventory distinguishes logical snapshot size, physical cache size, incomplete
bytes, expected reclaimable bytes, and measured reclaimed bytes.

## ModelUpdateInspection

| Field | Type | Meaning |
|-------|------|---------|
| `model_id` | UUID | Installed pinned revision |
| `tracking_ref` | string | Explicit mutable ref; required for update checks |
| `installed_commit` | 40-char SHA | Current immutable entry |
| `candidate_commit` | 40-char SHA/null | Ref resolution result |
| `status` | enum | `no_change`, `available`, `incompatible`, `offline`, `access_denied`, `failed` |
| `checked_at` | UTC datetime | Explicit-action time |
| `inspection` | `ModelInspection`/null | Full compatible candidate inspection |
| `error` | `ErrorDetail`/null | Safe result; never changes installed entry |

Downloading an `available` candidate creates or reuses a separate `(repo_id, candidate_commit)` entry.
Startup, `refresh()`, and inventory listing never create this record or contact the network.

## Catalog Boundary Records

- `ModelSourceInput`: raw URL, requested role, and optional tracking ref before normalization.
- `ModelInspection`: normalized source, access state, reviewed profile, dependency closure, expected
  missing bytes, compatibility, and safe warnings; no license/card fields.
- `DownloadEvent`: sanitized monotonic file/byte progress and accepted `DiskPreflight` summary.
- `DeletePreview`: eligibility plus a short-lived token bound to model/repo/commit, dependency closure,
  deletion-strategy fingerprint, expected bytes, and expiry.
- `DeleteResult`: terminal state, verified snapshot absence/presence, measured before/after physical
  cache bytes, actual reclaimed bytes, and safe partial-failure detail.
- `PartialDiscardPreview/Result`: separate confirmed records for app-owned incomplete content only.

## ModelLease

Protects an immutable revision while selected for active inference.

| Field | Type | Meaning |
|-------|------|---------|
| `lease_id` | UUID | Unique lease |
| `model_id` | UUID | Leased model |
| `dependency_ids` | list[UUID] | Auxiliary snapshots protected by the same atomic acquisition |
| `request_id` | UUID | Owning generation |
| `acquired_at` | UTC datetime | Audit time |
| `released_at` | UTC datetime/null | Set exactly once |

Acquisition increments model/dependency lease counts atomically; release decrements them even after
failure/cancellation. UI-active protection becomes request leases under the same catalog lock.

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
| `compatibility` | record | Language, artifact-interface, device, and memory checks |

Every accepted set resolves video, voice, and lip-sync exactly once. Dedicated IDs may be absent only
when the video model supplies and the effective choice uses the corresponding native capability.

## GenerationRequest

Validated UI input for exactly one generation.

| Field | Type | Validation / meaning |
|-------|------|----------------------|
| `request_id` | UUID | Server-generated collision-safe identity |
| `image_path` | Path | Required local still image from upload workspace or validated retained bundle; copied into staging |
| `motion_prompt` | string | Trimmed, 1-2,000 characters |
| `speech_script` | string | Trimmed, 1-2,000 characters |
| `reference_audio_path` | Path | Required upload or validated retained-bundle file; copied into staging before inference |
| `reference_transcript` | string/null | Used only when effective voice profile allows/requires it |
| `language` | string | Explicit member of effective voice profile language list |
| `consent` | `ConsentAttestation` | Fresh server-bound record created from a true current submission |
| `model_set` | `ModelSet` | Fully resolved before allocation |
| `seed` | integer/null | `0..2^63-1`; null generates and records effective value |
| `preferred_num_frames` | integer | UI preference; selected video profile validates and may override it |
| `preferred_fps` | number | UI preference; selected video profile validates and may override it |
| `guidance_scale` | finite number | Selected profile bounds |
| `runtime_profile` | string | Compatible memory/device profile |

## ConsentAttestation

| Field | Type | Meaning |
|-------|------|---------|
| `request_id` | UUID | Current request only |
| `reference_audio_sha256` | SHA-256 | Binds consent to the exact submitted bytes |
| `confirmed` | literal `true` | False/absent submissions never create this record |
| `confirmed_at` | UTC datetime | Server timestamp from the current submit event |

The UI checkbox defaults to false, resets after each submission, and resets whenever reference audio
changes. No bundle/history record can hydrate consent for a later request.

## VoiceReference

Request-scoped interpretation of uploaded reference audio.

| Field | Type | Meaning |
|-------|------|---------|
| `request_id` | UUID | Owning request |
| `source_path` | Path | Validated upload path or retained-bundle file selected by the user |
| `staged_path` | Path | Request-owned copy under `outputs/.work/<request-id>/inputs/` |
| `format`, `sample_rate`, `channels`, `duration_seconds` | scalar | Measured media properties |
| `speaker_result` | record | Provider-specific single-speaker/quality result |
| `transcript` | string/null | Optional model-specific reference text |
| `consent` | `ConsentAttestation` | Bound to staged audio digest and current request |
| `origin` | `VoiceOrigin`/null | Advisory origin when source resolves beneath an earlier valid bundle |
| `derived_artifact_path` | Path/null | Plaintext request-owned voice representation after synthesis |
| `lifecycle` | enum | `staged`, `retained`, `discarded_with_failed_stage` |

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

`WorkerProfile` records protocol version, interpreter identifier, dependency fingerprint, adapter/code
commit, PyTorch/CUDA versions, device capability, dtype, media timebase, allowed cache/staging roots,
and handshake status. The production LatentSync profile is local-only FP16 at 25 FPS/16 kHz; any
fingerprint mismatch fails before model load.

## ExecutionBlueprint, EffectiveExecutionPlan, and ExecutionStage

`ExecutionBlueprint` is compiled before any model inference. It fixes request ID, effective seed,
resolved providers/commits, provider choices, staging handle, runtime/worker profiles, static validation,
and the mandatory prefix `voice -> speech_verify -> duration_plan`, but it does not claim final temporal
values. After speech synthesis, `DurationPlan` finalizes the immutable `EffectiveExecutionPlan`.

| Field | Type | Meaning |
|-------|------|---------|
| `request_id` | UUID | Owning request |
| `effective_seed` | integer | Recorded seed |
| `resolved_models` | mapping | Role -> model ID, repo ID, commit, adapter |
| `provider_choices` | mapping | Effective Native/Dedicated decisions |
| `duration_plan` | `DurationPlan`/null | Null in blueprint; required in effective plan |
| `stages` | ordered list[`ExecutionStage`] | Blueprint prefix or finalized topological stages |
| `staging_dir` | Path | Fixed `outputs/.work/<request-id>` beneath the project root |
| `published_dir` | Path | Fixed `outputs/<request-id>` target; absent from history until atomic publication |

Each `ExecutionStage` stores `stage_id`, `kind`, `provider_model_id` or null, input/output artifact
names, runtime/worker profile, timebase, and unload-after-stage flag. The effective order is always
voice, speech verification, duration planning, video, optional duration-preserving resample, and lip
sync. A native composite may combine only the final video/lip stages after an accepted duration plan.
No two heavy model stages or worker processes overlap.

## DurationPlan

Immutable temporal decision created from the verified speech artifact before video allocation.

| Field | Type | Meaning |
|-------|------|---------|
| `speech_duration_seconds` | positive number | Authoritative decoded speech duration |
| `preferred_num_frames`, `preferred_fps` | scalar | User preferences retained for reporting |
| `effective_num_frames`, `effective_fps` | scalar | Adapter-supported combination selected by planner |
| `video_duration_seconds` | positive number | Duration implied by the effective combination |
| `lip_sync_fps`, `final_num_frames` | scalar/null | Declared bridge/final timebase when required |
| `tolerance_seconds` | positive number | Exactly one effective video frame |
| `overrides` | list[string] | Safe explanations for changed preferences |
| `adapter_constraint_id` | string | Versioned reviewed temporal rule used |
| `candidate_rank` | record | Deterministic delta/preference/profile-order tie-break evidence |

The plan is valid only when the full speech fits without trimming or time-stretching and the absolute
audio/video duration difference is no greater than `tolerance_seconds`. No valid combination is a
terminal `duration` error before video inference. Candidate duration is
`generated_frames / playback_fps`; selection sorts by duration delta, requested-frame distance,
requested-FPS distance, then stable profile order. The fixed CogVideoX profile contributes only
`(49, 8)`. The LatentSync bridge derives a 25-FPS final frame count while preserving clip duration and
uses `1 / 25` second as the final mux tolerance.

## Media Artifacts

### VideoArtifact

Request ID, path, frame count, FPS, width/height, codec/container status, duration, source stage, and
artifact kind (`video_pre_lip`, `video_post_lip`, or `final_mp4`), digest, byte size, and
staged/retained/discarded lifecycle state. Every successful variant is retained.

### SpeechArtifact

Request ID, path/waveform metadata, script hash, language, voice model/revision, sample rate, channels,
duration, finite/non-silence checks, normalization result, source stage, digest, byte size, and
staged/retained/discarded lifecycle state. The verified duration is authoritative for `DurationPlan`.

### LipSyncAlignment

Request ID, provider/revision, face/mouth target, technical processing state, output video artifact,
user-review state (`pending`, `reviewed`), and technical failure detail. It contains no automated
quality acceptance score in v1.

## ArtifactRecord

| Field | Type | Meaning |
|-------|------|---------|
| `kind` | `ArtifactKind` | Required retained artifact category |
| `relative_path` | Path | Relative to its bundle; absolute paths and `..` are forbidden |
| `media_type` | string | Sanitized MIME/media classification |
| `size_bytes` | non-negative integer | Measured retained size |
| `sha256` | 64-character hex | Integrity and origin-reuse check |
| `created_by_stage` | `StageKind` | Producer or `metadata` for copied inputs/manifests |

## VoiceOrigin

| Field | Type | Meaning |
|-------|------|---------|
| `bundle_id` | UUID | Earlier successful request bundle selected through the filesystem picker |
| `artifact_relative_path` | Path | Origin reference-audio path inside that bundle |
| `artifact_sha256` | SHA-256 | Digest observed when the new request was validated |

Origins are advisory. They must refer to an earlier successful bundle and cannot form a cycle at
creation time. Picker uploads are matched first by a validated request-ID-bearing retained filename and
digest, otherwise by a unique retained digest; ambiguous matches create no origin. History refresh may
later mark an origin missing/corrupt after external filesystem changes.

## RequestBundleManifest

Versioned immutable manifest written in staging and atomically published with a successful directory.

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | integer | Starts at `1`; unsupported versions are not trusted |
| `request_id` | UUID | Equals the directory name |
| `state` | literal `complete` | History never treats partial work as a bundle |
| `created_at`, `completed_at` | UTC datetime | Request and publication timestamps |
| `bundle_relative_path` | Path | Exactly `outputs/<request-id>` |
| `artifacts` | list[`ArtifactRecord`] | Contains every required artifact kind |
| `voice_origin` | `VoiceOrigin`/null | Advisory reuse edge |
| `consent` | record | `confirmed: true` and fresh server timestamp for this request |
| `language` | string | Effective explicitly selected speech language |
| `models` | mapping | Effective provider IDs, repositories, immutable commits, adapters |
| `parameters` | record | Requested preferences, effective seed, duration plan, runtime values |
| `memory_by_stage` | mapping | CUDA metrics or explicit unavailable records |
| `plaintext_sensitive_artifacts` | literal `true` | Disclosure invariant for reference/derived voice files |
| `disk_bytes` | non-negative integer | Sum of listed retained artifacts at publication (manifest excluded) |

The manifest contains no license data, token, raw prompt text, or source absolute path. Prompt content
may remain in a separately retained request input/metadata artifact when required for reproducibility,
but normal logs and history summaries expose only safe hashes/summaries.

## RequestHistoryEntry

Read-only projection produced by scanning the fixed `outputs/` directory.

| Field | Type | Meaning |
|-------|------|---------|
| `request_id` | UUID | Bundle identity |
| `bundle_path`, `preview_path` | Path/null | Validated contained paths; never followed outside `outputs/` |
| `availability` | `BundleAvailability` | Current reconciliation result |
| `artifacts` | list[`ArtifactRecord`] | Safe manifest inventory |
| `disk_bytes` | non-negative integer | Current measured size |
| `voice_origin` | `VoiceOrigin`/null | Persisted advisory origin |
| `dependent_bundle_ids` | list[UUID] | Computed in memory from current scan |
| `retained_voice_reusable` | boolean | False for missing/corrupt/unsafe origin or incompatible upload |
| `warnings` | list[string] | Safe reconciliation and plaintext-retention warnings |

The projection is never persisted back into a successful bundle and exposes no delete or reuse command.

`RequestHistorySnapshot` contains scan timestamp, ordered entries, total measured bytes, fixed-root
availability, and safe global warnings. It is an in-memory projection and is never written into bundles.

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
| `bundle_path` | Path/null | Atomically published `outputs/<request-id>` only on success |
| `manifest_path` | Path/null | Published request-bundle manifest only on success |
| `artifact_inventory` | list[`ArtifactRecord`] | Complete retained inventory on success; empty otherwise |
| `retained_bytes` | non-negative integer | Measured successful bundle size |
| `execution_plan` | record | Effective providers, revisions, choices, and parameters |
| `duration_seconds` | number | Wall-clock duration |
| `memory_by_stage` | mapping | Stage peak snapshots |
| `error` | `ErrorDetail`/null | Safe failure data |

## ErrorDetail

| Field | Type | Meaning |
|-------|------|---------|
| `code` | enum | `validation`, `consent`, `face`, `language`, `model_url`, `model_access`, `model_download`, `model_incompatible`, `inventory`, `model_load`, `unsupported_backend`, `oom`, `disk`, `speech`, `duration`, `lip_sync`, `mux`, `codec`, `history`, `cancelled`, `filesystem`, `internal` |
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

### Request-bundle publication

```text
absent -> outputs/.work/<request-id> -> verified staging -> outputs/<request-id>
                 |                              |
                 +-- failed/cancelled ----------+-> staging removed
```

Only the final atomic directory rename creates a history-visible bundle. Published bundles have no
application state transition to deleted; external filesystem removal is observed as `missing` on refresh.

## Cross-Entity Invariants

- A generation has exactly one terminal state and never publishes a partial output.
- `video_path` is visible only after container, stream, non-silence, and duration verification.
- Voice synthesis and speech verification precede duration planning; an accepted duration plan precedes
  every video allocation. Speech is never trimmed, stretched, or partially omitted.
- Every successful request records true consent, explicit language, immutable model commits, effective
  provider choices, seed, parameters, device/dtype, and per-stage memory.
- Every successful bundle contains all required `ArtifactKind` values and remains untouched by cleanup,
  expiry, history refresh, or cascade logic until the operator removes its directory externally.
- Failed/cancelled work has no published bundle and its `outputs/.work/<request-id>` directory is removed.
- Request History reads only the fixed `outputs/` root, does not follow escaping symlinks, and never
  exposes in-app bundle mutation. Missing voice origins disable reuse and warn dependents in memory.
- Output/staging names are server UUIDs. Root, `.work`, bundle, and artifact components reject symlinks
  and Windows reparse points; manifests reject absolute/drive/UNC/backslash/`..` paths. Failed staging
  cleanup receives only an internal verified handle and never a user path or published directory.
- Every accepted model set covers all three roles once; overlaps require an explicit choice.
- Inventory has one entry per `(repo_id, resolved_commit)`; inspections/downloads are idempotent, and
  startup/inference never resolve a mutable ref or contact the Hub for a ready local entry.
- A model cannot be deleted while active or leased. Every acquired lease is released in `finally`.
- Inventory and metadata writes use temporary files plus atomic replace under a cross-platform lock.
- Repository tokens, raw prompts, uploaded absolute paths, and derived voice representations never
  appear in normal logs or inventory records.
- Model-license data and acknowledgement state never appear in profiles, inventory, request metadata,
  history, validation decisions, or UI projections.
