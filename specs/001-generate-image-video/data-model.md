# Data Model: Generate Image-Conditioned Lip-Synced Video

The application uses Pydantic records in memory and versioned atomic JSON files on disk. It has no
database in v1. All paths are `pathlib.Path` values internally and serialize as app-root-relative
paths when persisted.

## Enumerations

| Name | Values |
|------|--------|
| `ModelRole` | `video`, `voice`, `lip_sync` — a single profile may declare all three natively |
| `ProviderMode` | `native`, `dedicated` |
| `ModelState` | `inspecting`, `downloading`, `verifying`, `ready`, `incompatible`, `failed`, `deleting` |
| `DownloadState` | `queued`, `inspecting`, `downloading`, `verifying`, `ready`, `failed`, `cancelled` |
| `RequestState` | `queued`, `validating`, `planning`, `running`, `exporting`, `complete`, `failed`, `cancelled` |
| `StageKind` | `validate`, `prepare_references`, `assemble_prompt`, `plan_duration`, `load_model`, `generate`, `decode`, `export`, `verify`, `metadata`, `publish` |
| `DeviceKind` | `cuda`, `mps`, `cpu` |
| `BundleAvailability` | `available`, `missing`, `corrupt`, `unsafe` |
| `ArtifactKind` | `original_image`, `reference_audio`, `derived_voice`, `assembled_prompt`, `decoded_video`, `decoded_audio`, `final_mp4`, `metadata` |

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
| `memory_profiles` | list | Offload mode, quantization, expected accelerator peak, expected host resident bytes |
| `duration_range_seconds` | record | Measured min/max/default supported output duration |
| `frame_rate` | positive number | Measured output frame rate |
| `resolutions` | list[record] | Measured supported output dimensions |
| `audio_output` | record | Measured sample rate and channel layout |
| `dialogue_languages` | non-empty list[string] | Measured stable language set |
| `speaking_rates` | mapping language -> positive number | Measured characters (or syllables) per second used to suggest a duration; a language may be absent |
| `reference_limits` | record | Accepted reference kinds with max counts and per-clip duration bounds; rejected kinds with reasons |
| `prompt_capacity_tokens` | integer | Measured prompt/token capacity |
| `dialogue_tag_form` | string | Documented tag syntax used to carry script content and language |
| `input_contract` | record | Required references and optional reference transcript |
| `output_contract` | record | Produced video/audio artifacts and whether they are generated jointly |
| `weight_policy` | record | Allowed extensions, required files, hashes/exceptions |
| `auxiliary_sources` | list[`ModelDependency`] | Complete pinned VAE/tokenizer/Whisper/etc. closure |
| `resource_profile` | record | Declared offload mode, quantization, accelerator ceiling, and host system-memory ceiling |

`native_capabilities` must be a subset of `roles`, and a profile claiming native voice or lip sync must
also claim `video`. License identifiers, terms, URLs, acknowledgement state, and compatibility judgments
are intentionally absent from this model.

**Every field in this table is a measured value belonging to one adapter.** No duration, frame rate,
resolution, sample rate, language list, reference limit, or token capacity may appear as an
application-level constant, a shared default, or a shared test assertion. The offline suite runs a second
time against a fixture profile whose values all differ, which fails if any of them have leaked into
architecture-level code.

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
| `image_paths` | non-empty list[Path] | One or more local still images from upload workspace or validated retained bundle; copied into staging. No application maximum; bounded by profile reference limits |
| `motion_prompt` | string | Trimmed, non-empty, no maximum length; truncated to the video adapter's text-encoder capacity with a recorded override |
| `speech_script` | string | Trimmed, non-empty, no maximum length; synthesized in full with provider segmentation when required |
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

## RuntimeProfile

| Field | Type | Meaning |
|-------|------|---------|
| `name` | string | Stable selected profile |
| `device` | `DeviceKind` | Capability-checked effective device |
| `dtype` | string | Effective precision for current stage |
| `offload_mode` | enum | `none`, `model_cpu`, `sequential_cpu`, `layer_wise`; declared, not discovered |
| `quantization` | string/null | Reviewed checkpoint/component mode; expected for the production profile |
| `max_reserved_bytes` | integer/null | Accelerator ceiling; production CUDA default is 13.5 GiB |
| `max_host_resident_bytes` | integer/null | **Host system-memory ceiling**, measured against installed RAM |
| `minimum_free_headroom_bytes` | integer/null | Required device free-memory margin around the heavy stage |
| `warnings` | list[string] | Safe limitations shown to user |

Both ceilings gate readiness. A profile satisfying `max_reserved_bytes` but breaching
`max_host_resident_bytes` is not selectable, and vice versa. Because layer-wise offload keeps the resident
model in host memory and streams it to the device, host RAM is a first-class budget rather than an
incidental one, and the required offload mode and quantization are recorded in the profile rather than
discovered at run time.

## ExecutionBlueprint, EffectiveExecutionPlan, and ExecutionStage

`ExecutionBlueprint` is compiled before any model inference. It fixes request ID, effective seed,
resolved providers/commits, provider choices, staging handle, runtime profile, static validation,
and static validation, but it does not claim a final duration. `DurationDecision` and `AssembledPrompt`
finalize the immutable `EffectiveExecutionPlan` before any model allocation.

| Field | Type | Meaning |
|-------|------|---------|
| `request_id` | UUID | Owning request |
| `effective_seed` | integer | Recorded seed |
| `resolved_models` | mapping | Role -> model ID, repo ID, commit, adapter |
| `provider_choices` | mapping | Effective Native/Dedicated decisions where a role overlaps |
| `duration_decision` | `DurationDecision`/null | Null in blueprint; required in effective plan |
| `assembled_prompt` | `AssembledPrompt`/null | Null in blueprint; required in effective plan |
| `stages` | ordered list[`ExecutionStage`] | Blueprint prefix or finalized topological stages |
| `staging_dir` | Path | Fixed `outputs/.work/<request-id>` beneath the project root |
| `published_dir` | Path | Fixed `outputs/<request-id>` target; absent from history until atomic publication |

Each `ExecutionStage` stores `stage_id`, `kind`, `provider_model_id` or null, input/output artifact
names, runtime profile, and unload-after-stage flag. The effective order is always validate, prepare
references, assemble prompt, plan duration, load model, generate, decode, export, verify, metadata, and
publish. Exactly one `generate` stage runs per request and it produces video and audio jointly, so no two
heavy model stages exist to overlap.

## DurationDecision

Immutable temporal decision recorded before any model allocation. There is no looping, repetition, or
padding: one generation produces the whole output, so this record selects and justifies the duration
handed to the adapter.

| Field | Type | Meaning |
|-------|------|---------|
| `suggested_duration_seconds` | positive number | Derived from trimmed script length and the profile's per-language speaking rate, clamped to the supported range |
| `speaking_rate_used` | record/null | Language and rate field the suggestion came from; null when the language has no entry |
| `requested_duration_seconds` | number/null | Operator override; null means the suggestion was accepted |
| `operator_overrode` | boolean | Whether the operator changed the suggested value |
| `effective_duration_seconds` | positive number | Value used, within the profile's supported range |
| `effective_num_frames` | integer | `effective_duration_seconds * profile.frame_rate` |
| `frame_rate` | positive number | From the profile, not a constant |
| `resolution` | record | Width/height from the profile's supported set |
| `audio_sample_rate` | integer | From the profile |
| `overrides` | list[string] | Safe explanations for any changed preference |
| `profile_id` | string | Adapter profile identity and revision the decision was made against |

Every numeric field above is derived from the effective `ModelProfile`, never from an application-level
constant. Because audio and video are generated jointly, duration is an **input** to generation and
cannot be measured from synthesized speech beforehand. There is therefore no pre-generation script-fit
check and no rejection for script length: the suggestion is advisory, the operator may override it
anywhere in the supported range, and rushed or clipped delivery is corrected by raising the duration and
regenerating.

## ReferenceSet

One or more image references plus exactly one audio reference. Video references are rejected.

| Field | Type | Validation / meaning |
|-------|------|----------------------|
| `request_id` | UUID | Owning request |
| `image_paths` | non-empty list[Path] | Staged still images anchoring subject identity and appearance, all of the same subject |
| `audio_path` | Path | Staged recording anchoring voice timbre only |
| `audio_role` | const `timbre_anchor` | Never played back, never mixed into output, never treated as content |
| `source_paths` | mapping staged -> Path | Validated upload path or retained-bundle file each staged copy came from |
| `measured` | record | Format, dimensions, duration, sample rate, channels, digests per reference |
| `speaker_result` | record | Provider-specific single-speaker/quality result for the audio reference |
| `transcript` | string/null | Optional model-specific reference text when the profile allows one |
| `consent` | `ConsentAttestation` | Bound to the staged audio digest and the current request |
| `origin` | `VoiceOrigin`/null | Advisory origin when the audio source resolves beneath an earlier valid bundle |
| `derived_artifact_path` | Path/null | Plaintext request-owned voice representation produced during generation |
| `profile_limits` | record | Accepted kinds, max counts, per-clip duration bounds from the profile |
| `rejected_kinds` | list[string] | Kinds refused with reason; video is always refused on token cost |
| `lifecycle` | enum | `staged`, `retained`, `discarded_with_failed_stage` |

The application imposes **no maximum** on `image_paths`; the only bound is the profile's measured
reference limit, and exceeding it is a `reference` error. Every image must independently contain exactly
one usable face and mouth region. Subject consistency across images is the operator's responsibility.

The UI states that the recording should say **different words** from the speech script. A recording whose
path resolves beneath a valid bundle adds an advisory `VoiceOrigin`.

## AssembledPrompt

The text actually submitted to the adapter, retained in every successful bundle.

| Field | Type | Meaning |
|-------|------|---------|
| `motion_text` | string | Motion description after any truncation |
| `dialogue_segments` | list[record] | Script content with the selected language, in the profile's dialogue-tag form |
| `rendered` | string | Final assembled prompt string submitted to the adapter |
| `token_count` | integer | Measured against the profile's capacity; may exceed it |
| `token_capacity` | integer | From the profile |
| `over_capacity` | boolean | Derived: whether `token_count` exceeds `token_capacity` |
| `motion_truncation` | record/null | `original_length`, `retained_length`, `discarded_length` |
| `structuring_version` | string | Version of the locally built prompt structuring |

Only `motion_text` may be truncated, and never silently. Dialogue segments are never truncated, dropped,
or reordered.

An over-capacity prompt is **recorded, never refused**. Motion gives way first, all of it if necessary;
if the script alone still exceeds capacity, `over_capacity` is set, the script is carried in full, and
the adapter decides. Refusing here would terminate a request because of its script length, which the
duration invariants forbid.

## GeneratedOutput

The single joint audio/video result of one generation. There is no separate speech artifact and no
lip-sync alignment record, because one adapter invocation produces mouth movement and voice together.

Request ID, decoded video path, decoded audio path, final MP4 path, frame count, frame rate,
width/height, audio sample rate and channel layout, container/stream summary, measured video and audio
duration, script hash, language, adapter profile and revision, a non-silence check over the spoken
region, digests, byte sizes, and staged/retained/discarded lifecycle state.

There is no pre-lip or post-lip variant, no intermediate timebase, no separate mux input, and no
automated lip-sync quality score. Synchronization is judged visually by the operator, per the
publication policy; the application records no review state because it exposes no review workflow.

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
| `parameters` | record | Requested preferences, effective seed, duration decision, assembled-prompt token counts and truncation override, runtime values |
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

`MemorySnapshot` contains availability, device name, current/peak allocated bytes, current/peak
allocator-reserved bytes, driver-reported free/total bytes when CUDA is active, gate result, and an
unavailable reason otherwise. Production acceptance gates peak reserved, not peak allocated.

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
| `code` | enum | `validation`, `consent`, `face`, `reference`, `language`, `duration`, `model_url`, `model_access`, `model_download`, `model_incompatible`, `inventory`, `model_load`, `unsupported_backend`, `oom`, `host_memory`, `disk`, `generation`, `export`, `codec`, `history`, `cancelled`, `filesystem`, `internal` |
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
- An accepted `DurationDecision` and `AssembledPrompt` precede every model allocation. The model is
  invoked exactly once per request and produces video and audio jointly. Speech is never trimmed,
  time-stretched, truncated, or partially omitted, and no request terminates because of script length.
  The `duration` error is reserved for an operator override falling outside the profile's supported
  range.
- No output is looped, repeated, reversed, padded, resampled across timebases, or concatenated. Mouth
  movement comes from the same invocation as the audio, so no separate lip-sync or face-validation stage
  exists.
- One or more image references and exactly one audio timbre anchor are accepted, bounded only by the
  profile's measured reference limits. Video references are rejected. The reference recording is never
  played back, mixed into the output, or treated as content.
- Duration is an input to generation, never a measurement. No request is rejected for script length.
- Every profile is validated against both the accelerator-memory and host system-memory ceilings before
  becoming selectable. Inference time is unbounded and carries no estimate or confirmation gate.
- Every model-specific value -- duration range, frame rate, resolution, audio sample rate, languages,
  reference limits, token capacity -- lives in the adapter profile. A value of this kind appearing in
  shared code, a shared constant, or a shared test assertion is an invariant violation.
- A truncated motion prompt is always recorded as an explicit override with original, retained, and
  discarded lengths; silent truncation is an invariant violation.
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
