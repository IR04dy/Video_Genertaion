# Feature Specification: Generate Image-Conditioned Video

**Feature Branch**: `001-generate-image-video`

**Created**: 2026-08-27

**Status**: Ready for Planning

**Input**: User description: "Build a production-ready local web application that turns an image
and motion prompt into a downloadable MP4 across macOS development and Windows NVIDIA production."

**Additional Input**: The user chooses compatible video, voice-cloning, and lip-sync models by
supplying Hugging Face model URLs and can view compatible models already downloaded locally.

## Clarifications

### Session 2026-08-27

- Q: What audio style should the app produce? → A: Speech only, with synchronized lip movement.
- Q: How are script and uploaded audio used? → A: Synthesize the script in the uploaded reference voice, then lip-sync the generated video.
- Q: What voice-cloning consent safeguard is required? → A: Require per-request confirmation of voice ownership or explicit consent and record it in metadata.
- Q: How should the app choose a lip-sync target? → A: Require exactly one clearly visible face.
- Q: What speech-language scope is required? → A: Multilingual speech with an explicit user-selected language.

### Session 2026-08-27

- Q: Which model roles may users configure with Hugging Face links? → A: Video, voice-cloning, and lip-sync; separate voice/lip-sync models are optional when the video model provides validated native capabilities.
- Q: How should users manage downloaded models? → A: Allow confirmed manual deletion of inactive models, protect active or in-use models, and report reclaimed disk space.
- Q: When both native and dedicated voice/lip-sync providers are available, which should be used? → A: Require the user to explicitly choose Native or Dedicated for each overlapping role and reject ambiguous configurations.
- Q: What validation policy applies to the uploaded reference voice? → A: Use the effective voice provider's own formats and limits, display them in the UI, and validate them before inference; do not impose an application-wide format or duration policy.
- Q: How should lip-sync quality determine whether an output is published? → A: Do not apply an automated quality gate; export every technically successful result and let the user judge lip synchronization visually.
- Q: What should happen when synthesized speech duration differs from the selected video duration? → A: Speech duration overrides the requested video controls; automatically choose compatible frame count/FPS, or reject before video generation if the model cannot represent the full speech without trimming or time-stretching it.
- Q: What should happen when a downloaded Hugging Face repository publishes a newer revision? → A: Keep the downloaded commit pinned; check for updates only on explicit user request and store a downloaded update as a separate revision without replacing the existing one.
- Q: What should the default cleanup behavior be after a request finishes? → A: For every successful request, retain the final MP4 and all inputs, derived voice data, speech, intermediates, and metadata locally until the user manually deletes the request bundle.
- Q: May a user reuse a retained reference voice in a later generation? → A: Yes; allow re-uploading a retained reference-voice file through the filesystem picker, but require a new ownership/permission consent attestation for every generation.
- Q: What should happen when disk space is insufficient for a model download or retained request bundle? → A: Block before download/inference unless estimated required space plus a configurable 10 GiB safety reserve is available, and require the user to free space manually.
- Q: How should retained reference audio and derived voice data be protected on disk? → A: Store them as ordinary unencrypted files in the fixed project `outputs/` directory and clearly disclose that the application provides no at-rest encryption.
- Q: How should the application handle Hugging Face model license terms? → A: Do not display, record, acknowledge, or enforce model-license information; compliance is entirely the local operator's responsibility.
- Q: What should happen when deleting a request bundle whose retained voice was reused by later successful requests? → A: Record and display dependencies as advisory, but allow filesystem deletion; on refresh, detect missing origins and invalidate affected retained-voice reuse.
- Q: How should users access and manage retained request bundles? → A: Provide a read-only in-app Request History; voice reuse uses the filesystem upload picker, bundle deletion is filesystem-only, dependency warnings are advisory, and missing bundles are reconciled on refresh.
- Q: How should the request-bundle directory be selected? → A: Always use the fixed project `outputs/` directory with no UI setting, environment override, or per-request destination selection.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate a Lip-Synced Video (Priority: P1)

A creator uploads a local image and a reference-voice recording, which may be chosen from a retained
bundle through the filesystem upload picker, provides separate motion and speech scripts, optionally
adjusts generation parameters, and receives a playable and downloadable MP4 whose generated speech
uses the reference voice with synchronized lip movement.

**Why this priority**: This is the complete value-producing journey and the minimum viable product.

**Independent Test**: Submit a valid image, motion prompt, speech script, and reference recording
through the local interface with a controlled test engine and verify that the returned MP4 speaks
the script in the reference voice with synchronized lip movement.

**Acceptance Scenarios**:

1. **Given** a valid local image, motion prompt, speech script, and reference-voice recording,
   **When** generation completes, **Then** the
   user can preview and download a uniquely named MP4 containing video, intelligible speech, and
   lip movement synchronized to that speech.
2. **Given** advanced parameters are unchanged, **When** generation starts, **Then** documented safe
   defaults are used as initial preferences and duration-adjusted effective values are reported.
3. **Given** a fixed seed and identical effective inputs, **When** generation is repeated on the
   same supported runtime, **Then** the request metadata is identical and output is reproducible
   within the underlying model's deterministic limits.
4. **Given** a speech script and valid reference-voice recording, **When** generation completes,
   **Then** the spoken audio matches the script in the reference voice and visible mouth movement
   follows the synthesized speech.
5. **Given** the user has not confirmed ownership or explicit permission for the reference voice,
   **When** generation is requested, **Then** the request is rejected before model inference.
6. **Given** the input contains zero faces, multiple faces, or no clearly visible mouth region,
   **When** generation is requested, **Then** the request is rejected before video generation.
7. **Given** a supported language is selected, **When** generation completes, **Then** the speech
   follows that language's pronunciation rules and the recorded metadata identifies the language.
8. **Given** synthesized speech has a duration supported by the selected video model, **When** video
   generation is planned, **Then** the system automatically chooses compatible effective frame count
   and FPS values that preserve the complete speech within one frame of the video duration.
9. **Given** synthesized speech is longer than every duration supported by the selected video model,
   **When** duration planning runs, **Then** the request stops before video generation and asks the user
   to shorten the script or select a compatible model without trimming or time-stretching the speech.
10. **Given** a generation completes successfully, **When** the user later views it in Request History,
    **Then** the read-only entry exposes its preview, artifact inventory, disk size, retained-voice and
    dependency status while its bundle remains on disk until the user removes it through the filesystem.
11. **Given** a retained reference voice file remains available and compatible, **When** the user
    chooses it through the filesystem upload picker for a new generation, **Then** the app requires a
    new ownership/permission attestation and performs zero inference if it is absent or false.
12. **Given** the application will retain reference audio or derived voice data, **When** the user
    submits a successful request, **Then** the UI discloses that these artifacts are stored unencrypted
    in the fixed project `outputs/` directory.
13. **Given** later retained request bundles reference a voice originating in an earlier bundle,
    **When** Request History is displayed, **Then** it lists the advisory dependencies; if the user
    deletes the earlier directory through the filesystem, the next refresh marks the origin missing and
    disables affected retained-voice reuse without deleting later bundles automatically.
14. **Given** any successful generation, **When** its request bundle is published, **Then** it is stored
    beneath the fixed project `outputs/` directory and appears in Request History without asking for or
    accepting an alternate destination.

---

### User Story 2 - Select and View Models (Priority: P2)

A creator supplies Hugging Face URLs for the required generation roles, downloads compatible models
with visible status, and can select, manually check updates for, or remove inactive entries from an
inventory of models already available locally. A video model with validated native voice-cloning or
lip-sync capabilities can cover those roles without separate model selections.

**Why this priority**: Model choice affects output quality and resource use, while local inventory
prevents unnecessary repeat downloads and makes offline reuse understandable.

**Independent Test**: Submit compatible Hugging Face repositories through a fake Hub client, verify
their detected roles/capabilities, confirm that a native-capability video model makes corresponding
voice/lip selections optional, restart, verify the inventory preserves models and revisions, then
delete an inactive model and verify confirmation, cache removal, and reclaimed-space reporting.

**Acceptance Scenarios**:

1. **Given** a canonical Hugging Face URL for a compatible video, voice-cloning, or lip-sync model,
   **When** the user downloads it, **Then** progress is displayed and the completed model appears with
   its validated roles/capabilities in the local inventory and appropriate selector.
2. **Given** a compatible model is already downloaded, **When** the application restarts without
   network access, **Then** the model remains visible and selectable without another download.
3. **Given** a malformed URL, unsupported host, incompatible pipeline, gated/private repository
   without credentials, or failed download, **When** validation runs, **Then** the model is not marked
   ready and the user receives an actionable message.
4. **Given** multiple downloaded revisions or models, **When** the inventory is displayed, **Then**
   each entry shows model ID, revision, roles/capabilities, compatibility, local size, and state.
5. **Given** the selected video model provides validated voice-cloning and lip-sync capabilities,
   **When** generation is configured, **Then** separate voice and lip-sync model inputs are optional;
   otherwise each uncovered role requires a compatible selected model.
6. **Given** a downloaded model is inactive and not used by a running request, **When** the user
   confirms its deletion, **Then** its application-managed files and inventory entry are removed and
   the reclaimed disk space is reported; active or in-use models cannot be deleted.
7. **Given** both the video model's native capability and a dedicated model can provide voice-cloning
   or lip-sync, **When** the user configures generation, **Then** the user must choose Native or
   Dedicated for that role and generation cannot start while the choice is ambiguous.
8. **Given** an effective voice provider has declared reference-audio constraints, **When** it is
   selected, **Then** the UI displays those constraints and rejects a reference recording that does
   not satisfy them before inference.
9. **Given** voice synthesis, lip-sync processing, and muxing complete without a technical failure,
   **When** export completes, **Then** the result is available for preview and download without an
   automated lip-sync quality score blocking publication, and the user judges sync quality visually.
10. **Given** a ready model's repository contains a newer commit, **When** the user explicitly checks
    for updates and downloads it, **Then** the new commit appears as a separate inventory revision and
    the existing pinned revision remains installed and selectable until explicitly deleted.
11. **Given** a model download would leave less than the configured disk reserve, **When** preflight
    runs, **Then** the download is blocked before network transfer and the user sees required, available,
    and manually reclaimable space without automatic deletion.
12. **Given** a repository declares model-license terms, **When** the user downloads or selects a
    compatible revision, **Then** the application neither blocks on nor displays/records those terms;
    gated/private access authentication remains independently enforced by the repository.

---

### User Story 3 - Understand Progress and Resource Use (Priority: P3)

A creator can see model loading, preprocessing, voice synthesis, duration planning, video inference,
lip synchronization, muxing, and export progress plus relevant memory information, so long operations
do not appear stalled.

**Why this priority**: Video diffusion is slow and memory intensive; usable feedback is essential.

**Independent Test**: Run a controlled generation whose phases emit known events and verify ordered
status updates, completion state, and device-appropriate memory information.

**Acceptance Scenarios**:

1. **Given** generation is running, **When** it changes phase, **Then** the interface identifies the
   current phase and updates progress without exposing secrets or full local paths.
2. **Given** an accelerated production device, **When** generation runs, **Then** allocated and peak
   accelerator memory are reported; on other devices, memory reporting is clearly marked unavailable.

---

### User Story 4 - Recover from Unsupported or Exhausted Environments (Priority: P4)

A developer can launch and exercise the interface on macOS or CPU-only hardware, while a production
user receives actionable recovery guidance for resource and dependency failures.

**Why this priority**: Cross-platform development must not be blocked by CUDA assumptions, and GPU
failures must not crash the web application.

**Independent Test**: Simulate unavailable accelerators, out-of-memory, invalid input, missing model,
and missing video encoder conditions and verify safe fallback or actionable error results.

**Acceptance Scenarios**:

1. **Given** CUDA is unavailable, **When** the app starts, **Then** it selects a supported fallback,
   warns about expected limitations, and does not invoke CUDA-only operations.
2. **Given** generation exceeds available accelerator memory, **When** the failure occurs, **Then**
   the request ends safely and suggests reducing frame count/resolution or using stronger offload.
3. **Given** video encoding cannot run, **When** export begins, **Then** the user is told which local
   dependency is missing or invalid and no false-success download is shown.
4. **Given** a generation cannot retain its estimated request bundle while preserving the configured
   disk reserve, **When** preflight runs, **Then** it performs no model inference and asks the user to
   manually delete request bundles or inactive models.

### Edge Cases

- Image is missing, corrupt, animated, extremely large, transparent, grayscale, or has EXIF rotation.
- Image contains no face, more than one face, a severely occluded mouth, or a face too small for the
  effective lip-sync provider's input constraints.
- Prompt is empty, whitespace-only, unusually long, or contains characters outside ASCII.
- Frame count is outside the selected model's valid temporal sequence or exceeds the memory profile.
- Synthesized speech duration cannot be represented by any supported frame-count/FPS combination for
  the selected video model.
- Model ID is unavailable, gated, incompatible, maliciously formatted, or requires authentication.
- Hugging Face URL uses a non-HTTPS scheme, unsupported host/path, missing repository, unsupported
  model role/task, unsupported pipeline class, unapproved remote code, or an incompatible revision.
- Selected models expose conflicting roles, incompatible revisions/interfaces, or leave voice-cloning
  or lip-sync capabilities uncovered.
- Both native and dedicated providers cover a role but the user has not explicitly selected which
  provider should execute it.
- Model download is interrupted, incomplete, corrupted, duplicated, or exceeds available local disk.
- Model deletion is requested for an active/in-use model, is cancelled at confirmation, encounters a
  shared cache entry, or removes only part of the application-managed files.
- A model update check finds no change, fails offline, resolves a gated revision, or discovers a newer
  revision that is incompatible with the installed adapter/runtime profile.
- Seed is negative or exceeds the supported integer range; FPS or guidance is outside safe bounds.
- The fixed project `outputs/` directory is missing, read-only, contains spaces, is a symlink outside
  the project, or is resolved incorrectly under Windows path semantics.
- Generation is cancelled, two requests overlap, or stale output from a prior request exists.
- Retained successful request bundles consume all available local disk, contain missing/corrupt files,
  or are requested for deletion while their final MP4 is being downloaded or previewed.
- The fixed project `outputs/` directory is readable by another local account, synchronized by an
  external backup/cloud tool, or located on removable/shared storage despite containing unencrypted
  voice artifacts.
- MPS reports availability but the chosen pipeline contains unsupported operations.
- Reference audio is missing, corrupt, too short, noisy, contains multiple speakers, or uses a
  format, sample rate, channel layout, duration, or speaker configuration unsupported by the
  effective voice provider.
- A retained reference voice was deleted, became corrupt, or is incompatible with the newly selected
  voice provider or language profile.
- A request-bundle dependency record is missing/corrupt, points to a nonexistent bundle, or would form
  a cycle despite voice reuse being allowed only from an already successful earlier request.
- Selected language is unsupported by the active voice-cloning profile, or the speech script uses a
  different language from the explicit selection.
- Voice ownership/consent confirmation is absent or false.
- Speech or lip-sync model access is gated, speech generation fails, no suitable face is detected,
  speech is silent/clipped, lip alignment fails, or muxing creates a duration mismatch.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST accept one local still image, one non-empty motion prompt, one separate
  non-empty speech script, and one local reference-voice recording selected through the filesystem
  upload picker, including a compatible recording located in a retained request bundle.
- **FR-002**: The system MUST expose seed, preferred frame count, preferred FPS, guidance scale,
  Hugging Face model URLs, and downloaded-model selection for each applicable role as advanced
  controls with documented defaults and ranges; effective frame count/FPS MAY be duration-adjusted.
- **FR-003**: The system MUST validate and normalize image orientation, color mode, dimensions, and
  model-specific temporal/spatial constraints before generation begins.
- **FR-004**: The system MUST select the best available supported execution device in the order
  CUDA, Apple acceleration, then CPU and display the effective device and limitations.
- **FR-005**: The system MUST support a production memory profile that stays within a 16 GB
  accelerator budget for the default model and parameter preset.
- **FR-006**: The system MUST provide opt-in offload or quantization profiles for models that cannot
  fit the production memory budget unmodified.
- **FR-007**: The system MUST report phase-level progress for preprocessing, voice synthesis, duration
  planning, video inference, lip synchronization, muxing, and export plus device-appropriate memory
  information.
- **FR-008**: The system MUST export successful results as uniquely named, browser-playable MP4 files
  containing video and intelligible speech synchronized with visible lip movement at the requested
  FPS and provide preview and download.
- **FR-009**: The system MUST record the effective model set and immutable revisions, provider choices,
  seed, device, precision, input parameters, output path, duration, and completion state for each
  request.
- **FR-010**: The system MUST translate invalid input, authentication, model loading, unsupported
  operation, accelerator OOM, cancellation, and encoding failures into actionable user messages.
- **FR-011**: The system MUST keep uploaded and generated media local unless the user explicitly
  configures a remote model credential or repository download.
- **FR-012**: The system MUST limit concurrent generations to prevent overlapping requests from
  exceeding the configured memory budget.
- **FR-013**: The system MUST allow the UI and complete control path to be tested without downloading
  multi-gigabyte model weights or requiring CUDA.
- **FR-014**: The system MUST document separate macOS development and Windows NVIDIA installation,
  verification, launch, and troubleshooting procedures.
- **FR-015**: The system MUST retain every successful request bundle until the user removes its
  directory through the filesystem and MUST immediately remove unpublishable partial artifacts from
  failed or cancelled requests; in-app deletion and automatic expiry are out of scope for v1.
- **FR-016**: The system MUST synthesize the speech script in the voice represented by the uploaded
  reference recording and MUST NOT add background music or ambient sound in v1.
- **FR-017**: The system MUST align visible mouth movement to the synthesized speech, keep audio and
  video duration within one frame, and mux the aligned speech into the final MP4.
- **FR-018**: The system MUST run video and audio model stages sequentially so the combined workflow
  remains within the configured accelerator-memory budget.
- **FR-019**: The system MUST treat speech/lip-sync model access, face detection, speech synthesis,
  alignment, and mux failures as actionable terminal errors and MUST NOT publish an output when a
  required processing stage fails or when the resulting speech track is silent.
- **FR-020**: The system MUST require a per-request attestation that the user owns the reference voice
  or has explicit permission to clone it, reject false or absent attestation before inference, and
  record the confirmation and timestamp in sanitized request metadata.
- **FR-021**: The system MUST require exactly one clearly visible face with a usable mouth region in
  the input image and MUST reject zero-face, multi-face, or insufficient-quality inputs before video
  generation begins.
- **FR-022**: The system MUST require the user to select a speech language from the active profile's
  supported language allowlist, validate support before inference, pass the selection through voice
  synthesis, and record it in result metadata; automatic language detection is out of scope for v1.
- **FR-023**: The system MUST accept only canonical HTTPS Hugging Face repository URLs for video,
  voice-cloning, and lip-sync model roles, normalize each to a repository ID and optional immutable
  revision, and reject unsupported hosts, routes, or ambiguous links before downloading weights.
- **FR-024**: The system MUST inspect repository metadata/configuration and accept only models that
  map to supported adapters/runtime profiles, recording every validated role and native capability;
  loading unreviewed remote code MUST remain disabled.
- **FR-025**: The system MUST show validation and download progress, support safe retry/resume where
  the model source permits it, and MUST NOT mark interrupted, corrupted, or incomplete downloads ready.
- **FR-026**: The system MUST persist and display a local model inventory containing repository ID,
  revision, adapter/pipeline type, roles/capabilities, compatibility, local size, download state, and
  last-used time, and MUST make ready entries selectable without network access.
- **FR-027**: The system MUST obtain gated/private repository credentials only from an approved local
  credential source and MUST NOT accept, display, log, or persist Hugging Face tokens in the model URL.
- **FR-028**: The system MUST allow a validated video model's native voice-cloning and/or lip-sync
  capabilities to satisfy those roles; separate voice or lip-sync model URLs MUST be optional only
  for roles covered by the video model and required for every uncovered role.
- **FR-029**: The system MUST reject a selected model set before inference when roles are uncovered or
  model interfaces, languages, revisions, devices, or memory profiles are incompatible.
- **FR-030**: The system MUST allow a user to manually delete an inactive downloaded model only after
  explicit confirmation, MUST prevent deletion of active or in-use models, MUST remove only files
  managed exclusively by the application, and MUST report the actual disk space reclaimed.
- **FR-031**: When both a video model's native capability and a dedicated selected model can provide
  voice-cloning or lip-sync, the system MUST require an explicit Native or Dedicated provider choice
  for each overlapping role, record the effective choice, and reject ambiguous configurations before
  inference; when exactly one compatible provider covers a role, the system MAY select it directly.
- **FR-032**: The system MUST obtain reference-audio formats, duration bounds, sample-rate/channel
  requirements, speaker constraints, and quality rules from the effective voice provider's validated
  profile, display them before upload/submission, and reject nonconforming audio before inference;
  v1 MUST NOT impose a conflicting application-wide format or duration policy.
- **FR-033**: The system MUST NOT use an automated lip-sync quality score as a publication gate in
  v1; after all required stages complete technically and container/audio validation passes, it MUST
  provide the MP4 for preview and download so the user can judge lip synchronization visually.
- **FR-034**: The system MUST treat synthesized speech duration as authoritative, automatically choose
  adapter-supported effective frame count and FPS values whose video duration differs by no more than
  one frame, and record/report any override of requested preferences; if no supported combination can
  preserve the complete speech, it MUST reject before video generation and MUST NOT trim, time-stretch,
  or partially omit the speech.
- **FR-035**: The system MUST keep every ready downloaded model pinned to its resolved immutable commit,
  MUST check for newer repository revisions only after an explicit user action, and MUST represent a
  downloaded update as a separate inventory revision without automatically replacing, selecting, or
  deleting an existing revision.
- **FR-036**: A successful request bundle MUST retain the original image, reference-voice recording,
  derived voice representation, synthesized speech artifact, intermediate video/media, final MP4, and
  sanitized metadata in application-owned local storage until its directory is removed through the
  filesystem; the UI MUST disclose this sensitive-data retention and local disk usage.
- **FR-037**: The system MUST allow an available retained reference-voice file to be chosen through the
  filesystem upload picker for a later request only when it satisfies the new effective voice provider's
  constraints, MUST require and record a fresh ownership/permission attestation for every reuse, and
  MUST NOT carry consent forward or infer without the new request's valid attestation.
- **FR-038**: Before a model download or generation inference, the system MUST estimate the operation's
  required local storage and verify that completion would preserve a configurable safety reserve whose
  default is 10 GiB; otherwise it MUST block before network transfer/inference, report required and
  available space with manual cleanup guidance, and MUST NOT automatically delete models or request
  bundles.
- **FR-039**: The system MUST store retained reference recordings and derived voice representations as
  ordinary unencrypted files in the fixed project `outputs/` directory, MUST clearly disclose before
  retention that it provides no application-level at-rest encryption, and MUST NOT describe the files
  as encrypted or protected beyond controls supplied by the host filesystem/storage environment.
- **FR-040**: The system MUST NOT inspect, display, record, request acknowledgment of, or enforce model-
  license terms for downloaded/selected repositories; repository authentication and gated/private
  access errors remain in scope, while determining and satisfying license obligations is solely the
  local operator's responsibility.
- **FR-041**: When a successful request reuses a retained voice from an earlier request bundle, the
  system MUST record and display an advisory dependency on the originating bundle; because deletion is
  filesystem-managed, the system MUST reconcile history on refresh, mark missing/corrupt origins, and
  disable affected retained-voice reuse without deleting or corrupting later bundles.
- **FR-042**: The system MUST provide a read-only Request History showing each discovered bundle's
  preview, artifact inventory, disk size, retained-voice origin/dependents, availability, and warnings;
  it MUST provide no in-app reuse or bundle-deletion action, with reuse performed through the filesystem
  upload picker and deletion performed through external filesystem operations.
- **FR-043**: The system MUST publish every successful request bundle beneath the fixed project
  `outputs/` directory, MUST use that directory as the sole Request History scan root, and MUST NOT
  expose a UI setting, environment override, or per-request destination for request-bundle storage.

### Key Entities

- **Generation Request**: Image and voice references, motion/speech scripts, requested parameters,
  selected model set/provider choices, consent, language, and request ID.
- **Runtime Profile**: Device, precision, memory strategy, quantization mode, and capability warnings.
- **Generation Result**: Retained request-bundle path, MP4 path, effective parameters, timing, memory
  metrics, state, retained-artifact inventory/disk size, and error detail.
- **Request Bundle**: Stable request identity, artifact inventory, local directory/disk size, originating
  voice bundle if reused, advisory dependent bundle identities, availability/reconciliation state,
  creation time, and externally observed filesystem-removal state.
- **Model Profile**: Compatible downloaded model identity, adapter type, supported parameter
  constraints, defaults, precision policy, and memory strategies.
- **Downloaded Model**: Normalized repository ID, resolved revision, adapter/pipeline type, validated
  roles/native capabilities, compatibility result, local cache path, disk size, download state,
  active/in-use state, ownership of cached files, optional newer-revision availability, timestamps,
  and failure or deletion detail.
- **Model Download**: Request identity, normalized model source, progress, expected/received content,
  retry state, terminal validation result, and relationship to a downloaded model.
- **Model Set**: Selected video model plus optional dedicated voice-cloning/lip-sync models, resolved
  capability coverage, explicit provider choice for overlapping roles, cross-model compatibility
  result, and rejection details.
- **Speech Profile**: Resolved voice-cloning and lip-sync providers, whether supplied natively by the
  video model or by dedicated selected models, access status, supported-language allowlist,
  provider-specific reference-audio formats and constraints, sample rate, duration bounds, speaker
  rules, and device/memory strategy.
- **Voice Reference**: User-provided recording, validated format/duration/speaker characteristics,
  stable reference identity and originating request, compatibility attributes, per-request consent
  attestations and timestamps, temporary/retained-bundle paths, derived voice representation,
  plaintext-at-rest status, dependent request-bundle identities, last-used time, availability, and
  cleanup state.
- **Speech Artifact**: Request-scoped waveform, spoken content, voice, authoritative duration, sample
  rate, normalization result, and lifecycle state before it drives video-duration planning and is
  aligned, muxed, and retained in the successful request bundle.
- **Lip-Sync Alignment**: Detected face/mouth target, processing state, technical completion result,
  user-review state, and failure detail associated with the generated video and speech artifact.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A first-time user can submit a valid image, motion prompt, speech script, reference
  voice, selected language, and consent attestation and obtain a previewable, downloadable MP4
  without editing source code.
- **SC-002**: 100% of invalid parameter combinations in the documented bounds are rejected before
  model inference with a corrective message.
- **SC-003**: The application launches and completes its mocked end-to-end workflow on macOS and on
  a CPU-only host with zero accelerator-specific startup failures.
- **SC-004**: The default production preset completes on the target 16 GB accelerator without an
  unhandled out-of-memory exception during acceptance testing.
- **SC-005**: Every accepted request displays ordered preprocessing, voice synthesis, duration
  planning, video inference, lip synchronization, mux, and export phases and reaches success, failure,
  or cancellation.
- **SC-006**: 100% of successful generated-video downloads are valid MP4 containers accepted by the
  embedded player and preserve the effective FPS within encoder tolerance.
- **SC-007**: Repeating a request records identical effective inputs and seed, and produces stable
  results within the selected model/runtime's documented determinism limits.
- **SC-008**: Automated tests cover all stated failure classes without network access or model-weight
  downloads; accelerator smoke tests remain separately selectable.
- **SC-009**: Every successful MP4 contains exactly one playable speech stream whose duration is
  within one video frame of the video stream and whose signal is not digital silence; every such MP4
  is available in the embedded player for user visual review of lip synchronization.
- **SC-010**: 100% of accepted generation requests contain a recorded per-request voice ownership or
  explicit-consent attestation; requests without it perform zero model inference.
- **SC-011**: 100% of zero-face, multi-face, and unusable-mouth test inputs are rejected before video
  generation, while valid single-face inputs identify exactly one lip-sync target.
- **SC-012**: 100% of unsupported language/profile combinations are rejected before inference, and
  every accepted request records and uses the explicit user-selected language.
- **SC-013**: 100% of ready downloaded-model entries remain visible/selectable after restart,
  while malformed, incompatible, incomplete, or corrupted repositories never appear as ready.
- **SC-014**: A user can submit compatible Hugging Face model URLs, observe download status and
  validated roles/capabilities, and select the completed models without editing configuration files.
- **SC-015**: 100% of accepted model sets cover video, voice-cloning, and lip-sync capabilities,
  while uncovered or incompatible sets perform zero generation inference.
- **SC-016**: 100% of downloaded-model deletion attempts require confirmation, active or in-use models
  remain intact, and successful model deletion reports the measured disk space reclaimed from
  application-owned files.
- **SC-017**: 100% of model sets with overlapping native and dedicated providers record an explicit
  provider choice for each overlapping role; ambiguous sets perform zero inference.
- **SC-018**: For every validated voice provider profile, the UI displays its reference-audio
  constraints and 100% of test recordings that violate those constraints are rejected before
  inference.
- **SC-019**: 100% of outputs that complete all required processing stages and pass container/audio
  validation are made available for preview and download without automated lip-sync scoring, while
  technical stage failures are never reported as successful outputs.
- **SC-020**: 100% of accepted requests use the complete synthesized speech duration to derive
  adapter-supported effective frame count/FPS within one-frame tolerance and report overrides; speech
  that exceeds the selected model's supported duration causes zero video inference.
- **SC-021**: 100% of ready models continue using their recorded immutable commit until the user
  explicitly selects another revision; update checks perform no automatic download/replacement, and a
  downloaded update coexists as a separately selectable inventory entry.
- **SC-022**: 100% of successful request bundles retain the final MP4, all original inputs, derived
  voice/speech data, intermediates, and metadata across application restarts until their directories are
  externally removed; failed/cancelled requests retain no unpublishable partial artifacts.
- **SC-023**: 100% of available compatible retained reference-voice files can be re-uploaded through the
  filesystem picker, and every reuse records a new per-request consent attestation; absent/false
  re-consent causes zero inference regardless of prior attestations.
- **SC-024**: 100% of model downloads and generation requests whose estimated completion would breach
  the configured disk reserve are rejected before network transfer or inference with required/available
  space reported, and zero retained data is deleted automatically.
- **SC-025**: Before the first successful retention, the UI explicitly states that reference audio and
  derived voice artifacts are stored as unencrypted files beneath the project `outputs/` directory; no
  application state or documentation claims at-rest encryption is enabled.
- **SC-026**: 100% of compatible repository revisions are evaluated for roles/runtime without a model-
  license acknowledgment gate, and inventory/request metadata contains no captured license terms or
  acknowledgment state; gated/private access failures remain actionable.
- **SC-027**: 100% of discovered voice-reuse dependencies appear as advisory links in Request History;
  after external deletion of an origin, the next refresh marks it missing and makes every affected
  retained-voice reuse unavailable without modifying later bundles.
- **SC-028**: Request History is read-only in 100% of UI contract tests: it exposes bundle preview,
  artifacts, disk size, voice dependencies, availability, and warnings but no reuse or deletion action.
- **SC-029**: 100% of successful request bundles are published beneath the project `outputs/` directory
  and discovered there after restart; UI, environment, and per-request inputs cannot redirect the
  request-bundle root.

## Assumptions

- Version 1 is a single-user, locally bound application with no authentication or public hosting.
- One generation runs at a time; multi-user scheduling and distributed inference are out of scope.
- The inputs are a still image, motion prompt, separate speech script, and reference-voice recording.
  The output uses cloned-voice speech with visible lip synchronization. Background music and ambient
  sound generation are out of scope for v1.
- Supported languages are declared by each voice-cloning profile and selected explicitly by the user;
  automatic language detection and silent fallback to another language are out of scope for v1.
- The operator is solely responsible for identifying and complying with every selected model's license;
  the application does not inspect, display, record, acknowledge, or enforce model-license terms.
  Gated repositories may still require a Hugging Face token supplied outside the UI.
- User-provided Hugging Face URLs may configure video, voice-cloning, and lip-sync roles. Dedicated
  voice/lip-sync models are optional only when the selected video model provides those validated
  native capabilities; repository access controls and application voice-consent safeguards still apply.
- If native and dedicated providers both cover a role, the application does not silently prioritize
  either provider; the user's explicit per-role choice determines the execution path.
- Reference-audio requirements are model-specific and come from the validated effective voice
  provider profile; the application does not define one universal file format or duration range.
- Lip-sync quality is evaluated visually by the user in v1; automated sync scoring, quality-based
  rejection, and automatic quality retries are out of scope.
- Synthesized speech duration determines the effective video duration. Requested frame count and FPS
  are preferences that may be adjusted to an adapter-supported combination; trimming, time-stretching,
  or omitting speech to satisfy those preferences is out of scope.
- Downloaded models are retained until the user explicitly deletes an inactive entry; v1 does not
  automatically evict models based on age or storage quotas.
- Model revisions never update automatically. The user explicitly checks for and downloads updates,
  and every resolved commit remains a separate selectable entry until explicitly deleted.
- CPU and Apple acceleration are intended for UI/control-path validation; production-quality video
  generation is expected on the Windows NVIDIA machine.
- Resolution is normalized to the selected model profile rather than exposed as an unrestricted v1
  control, while OOM guidance may advise selecting a lower preset.
- Every successful request retains its complete local bundle, including sensitive inputs, derived voice
  and speech data, intermediates, final MP4, and metadata, until the operator removes its directory
  through the filesystem; v1 has no in-app deletion or automatic expiry. Failed and cancelled requests
  clean unpublishable partial artifacts immediately.
- Retained reference-voice files may be re-uploaded through the filesystem picker, but no consent state
  is reusable: ownership or explicit permission is re-attested and recorded for every generation.
- Voice-reuse dependencies are advisory because the operator can delete bundle directories outside the
  application. History refresh detects missing origins and disables affected reuse without attempting
  automatic repair or deletion.
- The disk-space safety reserve is operator-configurable and defaults to 10 GiB. Insufficient space
  blocks downloads/generation until the user manually frees space; automatic eviction is out of scope.
- Reference recordings and derived voice representations are retained as unencrypted ordinary files in
  the fixed project `outputs/` directory; storage permissions, backups, synchronization, and physical
  device protection are the local operator's responsibility.
- Request-bundle storage is not configurable in v1. The fixed project `outputs/` directory is the sole
  publication and Request History root on every supported platform.
