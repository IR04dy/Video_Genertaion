# Feature Specification: Generate Image-Conditioned Video

**Feature Branch**: `001-generate-image-video`

**Created**: 2026-08-27

**Status**: Ready for Planning

**Input**: User description: "Build a production-ready local web application that turns an image
and motion prompt into a downloadable MP4 across macOS development and Windows NVIDIA production."

**Additional Input**: The user chooses compatible video, voice-cloning, and lip-sync models by
supplying Hugging Face model URLs and can view compatible models already downloaded locally.

## Clarifications

Decision log for this feature. Entries whose answers were later overturned outright — a fixed video
duration, ping-pong loop repetition, trailing-silence padding, a 110-character script cap, a
single-image reference rule, and the three-provider speech/lip-sync pipeline — have been removed
rather than annotated, so every entry below reflects a decision still in force. The full history
remains in git.

### Session 2026-08-27

- Q: What audio style should the app produce? → A: Speech only, with synchronized lip movement.
- Q: What voice-cloning consent safeguard is required? → A: Require per-request confirmation of voice ownership or explicit consent and record it in metadata.
- Q: What speech-language scope is required? → A: Multilingual speech with an explicit user-selected language.

### Session 2026-08-27

- Q: Which model roles may users configure with Hugging Face links? → A: Video, voice-cloning, and lip-sync; separate voice/lip-sync models are optional when the video model provides validated native capabilities.
- Q: How should users manage downloaded models? → A: Allow confirmed manual deletion of inactive models, protect active or in-use models, and report reclaimed disk space.
- Q: When both native and dedicated voice/lip-sync providers are available, which should be used? → A: Require the user to explicitly choose Native or Dedicated for each overlapping role and reject ambiguous configurations.
- Q: What validation policy applies to the uploaded reference voice? → A: Use the effective voice provider's own formats and limits, display them in the UI, and validate them before inference; do not impose an application-wide format or duration policy.
- Q: How should lip-sync quality determine whether an output is published? → A: Do not apply an automated quality gate; export every technically successful result and let the user judge lip synchronization visually.
- Q: What should happen when a downloaded Hugging Face repository publishes a newer revision? → A: Keep the downloaded commit pinned; check for updates only on explicit user request and store a downloaded update as a separate revision without replacing the existing one.
- Q: What should the default cleanup behavior be after a request finishes? → A: For every successful request, retain the final MP4 and all inputs, derived voice data, speech, intermediates, and metadata locally until the user manually deletes the request bundle.
- Q: May a user reuse a retained reference voice in a later generation? → A: Yes; allow re-uploading a retained reference-voice file through the filesystem picker, but require a new ownership/permission consent attestation for every generation.
- Q: What should happen when disk space is insufficient for a model download or retained request bundle? → A: Block before download/inference unless estimated required space plus a configurable 10 GiB safety reserve is available, and require the user to free space manually.
- Q: How should retained reference audio and derived voice data be protected on disk? → A: Store them as ordinary unencrypted files in the fixed project `outputs/` directory and clearly disclose that the application provides no at-rest encryption.
- Q: How should the application handle Hugging Face model license terms? → A: Do not display, record, acknowledge, or enforce model-license information; compliance is entirely the local operator's responsibility.
- Q: What should happen when deleting a request bundle whose retained voice was reused by later successful requests? → A: Record and display dependencies as advisory, but allow filesystem deletion; on refresh, detect missing origins and invalidate affected retained-voice reuse.
- Q: How should users access and manage retained request bundles? → A: Provide a read-only in-app Request History; voice reuse uses the filesystem upload picker, bundle deletion is filesystem-only, dependency warnings are advisory, and missing bundles are reconciled on refresh.
- Q: How should the request-bundle directory be selected? → A: Always use the fixed project `outputs/` directory with no UI setting, environment override, or per-request destination selection.
- Q: Which CUDA memory measurement should gate the 16 GB production profile? → A: Require no more than 13.5 GiB peak allocator-reserved memory per heavy stage on the display-attached Windows target; report allocated memory only as a diagnostic.

### Session 2026-08-27 (post-plan)

- Q: What happens when disk space runs out during generation rather than at preflight? → A: Monitor free space periodically during every streaming write stage; on reserve breach stop with the existing `disk` error, report required/available/reserve, remove the staging directory, and leave published bundles untouched.
- Q: How promptly must cancellation take effect in a stage that may run for hours? → A: Every duration-scaling stage checks for cancellation at bounded intervals no more than a few seconds apart; the worker receives a cancel message and is forcibly terminated after a stated grace period, then staging is cleaned and one terminal cancelled result returns.
- Q: What progress detail is required inside stages that can run for hours? → A: Report the current phase set including loop expansion and resampling, and require loop expansion, resampling, and lip synchronization to emit a monotonic completion fraction against total final frames at least every few seconds. (Phase names superseded by the model change; the fractional-progress requirement now applies to inference and decoding.)
- Q: May a model download run while a generation is in progress? → A: No; block starting any model download or update download while a generation is active, and report that the library is temporarily read-only until the run finishes.

### Session 2026-08-27 (model change)

- Q: Which model provides the default reviewed video profile? → A: `MiniMaxAI/MiniMax-H3` Ref2VA replaces CogVideoX-5B-I2V as the default. The adapter registry and Hugging Face URL selection are unchanged: H3 is the default profile, not the only one, and further reviewed profiles are added later.
- Q: How are speech and lip synchronization produced? → A: Natively and jointly by the video model. The dedicated text-to-speech stage, the dedicated lip-sync stage, the cross-provider timebase bridge, and the generated-frame face preflight are all removed.
- Q: Where does spoken content come from? → A: From the speech script, embedded in the prompt as `<d>[language]...</d>` dialogue tags. The reference recording is never played back and MUST say different words from the script; the UI surfaces this rule.
- Q: Is voice-cloning consent still required? → A: Yes, unchanged. This is still voice cloning; only the model performing it changed. Per-request attestation defaults and resets to false, resets whenever the reference audio changes, is bound server-side to request ID and reference-audio SHA-256, and is recorded in metadata.
- Q: How is output duration determined now? → A: By one generation at the adapter's own supported duration. The ping-pong loop architecture, loop plan, repetition count, streaming loop expansion, and the unbounded-output-duration requirement are all removed.
- Q: What bounds inference time? → A: Nothing. Inference time is explicitly unbounded, with no latency target or SLA. Layer-wise and sequential CPU offload and quantized checkpoints are permitted and expected, and a run taking hours is acceptable.
- Q: Which resource ceilings must every adapter satisfy? → A: Both accelerator memory and host system memory. A system-RAM budget sits alongside the VRAM gate, and every adapter is evaluated against both.

### Session 2026-08-27 (post-model-change)

- Q: How should the prohibition on background music and ambient sound be resolved now that audio and video are generated as one process? → A: Remove it. The application neither requests nor forbids non-speech audio; verification checks only that exactly one non-silent speech stream matching the script is present.
- Q: May a request supply more than one reference image? → A: Yes, with no application-level maximum. One or more images are accepted as identity/appearance references for the same subject, bounded solely by the effective adapter profile's measured reference limits. Exactly one audio timbre anchor is still required and video references are still rejected.
- Q: Duration is now an input to joint generation rather than a measurement of synthesized speech. How is it determined? → A: Suggest a default from script length and a per-language speaking-rate field in the adapter profile, clamped to the profile's supported range, and let the operator override within that range. There is no pre-generation fit check and no rejection for script length; a rushed delivery is corrected by raising the duration and regenerating.

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

1. **Given** a valid local image, motion prompt, speech script of any length, and reference-voice
   recording,
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
8. **Given** a speech script that fits the selected adapter's supported duration, **When** generation
   runs, **Then** the system performs exactly one joint audio/video generation and the resulting MP4
   carries the complete spoken script with natively synchronized mouth movement.
9. **Given** a long speech script, **When** the request is prepared, **Then** the system suggests a
   duration derived from the script and the profile's per-language speaking rate, clamped to the
   supported range, and the operator may override it within that range; the request is never rejected
   for script length.
10. **Given** a reference recording whose spoken words match the script, **When** the request is
    prepared, **Then** the UI states that the recording anchors timbre only and should say different
    words, and the recording is never played back or mixed into the output.
11. **Given** a generation completes successfully, **When** the user later views it in Request History,
    **Then** the read-only entry exposes its preview, artifact inventory, disk size, retained-voice and
    dependency status while its bundle remains on disk until the user removes it through the filesystem.
12. **Given** a retained reference voice file remains available and compatible, **When** the user
    chooses it through the filesystem upload picker for a new generation, **Then** the app requires a
    new ownership/permission attestation and performs zero inference if it is absent or false.
13. **Given** the application will retain reference audio or derived voice data, **When** the user
    submits a successful request, **Then** the UI discloses that these artifacts are stored unencrypted
    in the fixed project `outputs/` directory.
14. **Given** later retained request bundles reference a voice originating in an earlier bundle,
    **When** Request History is displayed, **Then** it lists the advisory dependencies; if the user
    deletes the earlier directory through the filesystem, the next refresh marks the origin missing and
    disables affected retained-voice reuse without deleting later bundles automatically.
15. **Given** any successful generation, **When** its request bundle is published, **Then** it is stored
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
9. **Given** joint audio/video generation, decoding, and export complete without a technical failure,
   **When** publication completes, **Then** the result is available for preview and download without an
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

A creator can see input validation, reference preparation, prompt assembly, model loading, joint
audio/video inference, decoding, export, and publication progress plus relevant accelerator and host
memory information, including a completion fraction during inference and decoding, so a run lasting
hours does not appear stalled.

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
   the request ends safely and suggests a shorter supported duration, a lower supported resolution, or
   stronger offload.
3. **Given** video encoding cannot run, **When** export begins, **Then** the user is told which local
   dependency is missing or invalid and no false-success download is shown.
4. **Given** a generation cannot retain its estimated request bundle while preserving the configured
   disk reserve, **When** preflight runs, **Then** it performs no model inference and asks the user to
   manually delete request bundles or inactive models.

### Edge Cases

- Image is missing, corrupt, animated, extremely large, transparent, grayscale, or has EXIF rotation.
- Image contains no face, more than one face, a severely occluded mouth, or a face too small for the
  effective lip-sync provider's input constraints.
- Prompt is empty, whitespace-only, or contains characters outside ASCII.
- Motion prompt exceeds the effective video adapter's text-encoder capacity and must be truncated with
  a recorded, displayed override rather than silently.
- Speech script is long enough to require provider-side segmentation, or contains multilingual text
  whose synthesized duration differs greatly from its character count.
- Requested duration is outside the profile's supported range, or a supported duration still exceeds the
  memory profile.
- The speech script is long relative to the suggested duration, so delivery may be rushed or clipped;
  the remedy is a longer duration and a regeneration, not a rejection.
- The selected language has no speaking-rate entry in the profile, so no duration can be suggested and
  the profile default is offered instead.
- The reference recording says the same words as the script, risking user confusion about whether it is
  played back.
- A reference recording falls outside the adapter's accepted clip-count or per-clip duration limits.
- More reference images are supplied than the effective profile's measured limit allows.
- One image in a multi-image reference set fails the face check while the others pass, or the images
  plainly depict different subjects.
- A video file is offered as a reference and must be rejected on token-cost grounds.
- The selected language is outside the adapter's stable dialogue-language set.
- A profile satisfies the accelerator-memory ceiling but breaches the host system-memory ceiling, or
  requires a quantized checkpoint or layer-wise offload to satisfy either.
- Requested speech is long enough that lip-sync processing and the retained bundle grow very large;
  the request still runs to completion unless a disk-reserve check blocks it.
- Free space falls below the configured reserve part-way through a long streaming write, either because
  the pre-inference estimate was low or because another process consumed the disk concurrently.
- A run under layer-wise offload occupies the machine for hours, during which the model library stays
  read-only and host memory remains largely committed.
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
- A model download is requested during a long-running generation and must be refused with a clear
  read-only explanation rather than a generic error.
- Cancellation arrives early in a multi-hour stage, or an isolated worker fails to yield within its
  grace period and must be forcibly terminated.
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

- **FR-001**: The system MUST accept, per request, one or more local still images anchoring subject
  identity and appearance, and exactly one local reference-voice recording used solely as a voice timbre
  anchor, together with one non-empty motion prompt and one separate non-empty speech script. It MUST NOT
  impose an application-level maximum on the number of reference images; the only bound is the effective
  adapter profile's measured reference limits, and exceeding those is refused as a reference error. All
  images are treated as references to the same subject, and subject consistency across them is the
  operator's responsibility. The recording is selected through the filesystem upload picker, including a
  compatible recording located in a retained request bundle. The system MUST reject video references. It
  MUST NOT impose an application-level maximum length on the motion prompt. The speech script is bounded
  by what the effective video adapter can represent in a single generation, as required by FR-034.
- **FR-002**: The system MUST expose seed, preferred duration, guidance scale, Hugging Face model URLs,
  and downloaded-model selection for each applicable role as advanced controls, with defaults and ranges
  read from the effective adapter profile rather than fixed by the application. Frame rate and resolution
  are profile-determined and reported, not user controls.
- **FR-003**: The system MUST validate and normalize image orientation, color mode, dimensions, and
  model-specific temporal/spatial constraints before generation begins. It MUST synthesize the
  complete speech script regardless of length, using the effective voice provider's own segmentation
  when the provider requires it, and MUST NOT drop, truncate, or reorder any part of the script. When
  a motion prompt exceeds the effective video adapter's text-encoder capacity, the system MUST
  truncate it to that capacity, MUST record the original length, retained length, and discarded
  remainder length as an explicit override in request metadata, and MUST display that override in the
  UI before and after generation; silent truncation is prohibited.
- **FR-004**: The system MUST select the best available supported execution device in the order
  CUDA, Apple acceleration, then CPU and display the effective device and limitations.
- **FR-005**: The system MUST support a production memory profile whose every heavy stage stays at or
  below 13.5 GiB peak CUDA allocator-reserved memory on the display-attached 16 GB Windows target;
  peak allocated memory is diagnostic and MUST NOT be the release gate.
- **FR-006**: The system MUST provide opt-in offload or quantization profiles for models that cannot
  fit the production memory budget unmodified.
- **FR-007**: The system MUST report phase-level progress for input validation and preprocessing,
  reference preparation, prompt assembly, model loading, joint audio/video inference, decoding,
  container export, verification, and publication, plus device-appropriate memory information including
  CUDA current/peak allocated, current/peak reserved, and free/total bytes, and host system-memory use.
  Because inference is unbounded in time and may run for hours under layer-wise offload, the inference
  and decoding phases MUST additionally report a monotonic completion fraction, refreshed at least every
  few seconds, so a long request is never indistinguishable from a stalled one.
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
  exceeding the configured memory budget. It MUST also refuse to start any model download or update
  download while a generation is active, MUST state that the model library is temporarily read-only for
  the duration of the run, and MUST allow the operator to cancel the generation if they need the library
  sooner. Inspection, listing, and update checks that transfer no model content remain available.
- **FR-012a**: Because a single stage may run for hours, the system MUST check for cancellation at
  bounded intervals no more than a few seconds apart inside every long-running stage, including model
  loading, joint audio/video inference, decoding, and export. It MUST propagate cancellation to any
  processing it has delegated, MUST forcibly reclaim delegated processing that does not yield within a
  documented grace period, and MUST then release accelerator resources and model leases, remove the
  unpublished staging directory, and return exactly one terminal cancelled result. Cancellation MUST NOT
  modify any published bundle or downloaded model.
- **FR-013**: The system MUST allow the UI and complete control path to be tested without downloading
  multi-gigabyte model weights or requiring CUDA.
- **FR-014**: The system MUST document separate macOS development and Windows NVIDIA installation,
  verification, launch, and troubleshooting procedures.
- **FR-015**: The system MUST retain every successful request bundle until the user removes its
  directory through the filesystem and MUST immediately remove unpublishable partial artifacts from
  failed or cancelled requests; in-app deletion and automatic expiry are out of scope for v1.
- **FR-016**: The system MUST produce the spoken audio in the voice represented by the uploaded
  reference recording, using the effective video adapter's native joint audio/video generation rather
  than a separate speech stage. Spoken content MUST come from the speech script, supplied to the adapter
  in its documented dialogue-tag form together with the selected language. The reference recording is a
  timbre anchor only: it MUST NOT be played back, mixed into the output, or transcribed as content, and
  the UI MUST state that it should say different words from the script.
- **FR-017**: The system MUST obtain visible mouth movement aligned to the spoken audio from the
  effective video adapter's native joint generation, MUST NOT run a separate lip-synchronization stage
  or cross-provider timebase conversion, MUST keep final audio and video duration within one frame, and
  MUST deliver a final MP4 carrying exactly one video stream and one non-silent speech stream.
- **FR-018**: The system MUST keep every request within both a configured accelerator-memory ceiling and
  a configured host system-memory ceiling, and MUST evaluate every adapter profile against both before
  it can be marked ready. Where a profile requires layer-wise or sequential CPU offload or a quantized
  checkpoint to satisfy those ceilings, that requirement MUST be recorded in the profile and applied
  deterministically rather than discovered at run time.
- **FR-019**: The system MUST treat model access, reference and face validation, prompt assembly,
  generation, decoding, and export failures as actionable terminal errors, and MUST NOT publish an
  output when a required stage fails or when the generated speech track is silent.
- **FR-020**: The system MUST require a per-request attestation that the user owns the reference voice
  or has explicit permission to clone it, reject false or absent attestation before inference, and
  record the confirmation and timestamp in sanitized request metadata.
- **FR-021**: The system MUST require exactly one clearly visible face with a usable mouth region in
  **each** supplied reference image, and MUST reject zero-face, multi-face, or insufficient-quality
  images before generation begins, naming which image failed. Because mouth movement is produced jointly
  with the audio by a single adapter, no separate post-generation face validation stage is required.
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
- **FR-028a**: A video adapter that natively provides voice and lip synchronization MUST declare those
  capabilities in its profile, MUST generate audio and video jointly in one invocation, and MUST satisfy
  the same validation, consent, language, reference, and resource requirements as any other profile.
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
- **FR-034**: The system MUST produce each request from exactly one generation at a duration the
  effective video adapter supports, and MUST NOT chain generations, repeat or loop output, re-generate
  per segment, or concatenate clips. Every video adapter MUST declare its supported duration range,
  frame rate, resolution, audio sample rate, dialogue languages, per-language speaking rate, reference
  limits, and prompt token capacity as measured profile fields; these MUST NOT be embedded in
  architecture-level invariants.
- **FR-034a**: Because audio and video are generated jointly, the output duration is an **input** to
  generation and cannot be measured from synthesized speech beforehand. The system MUST derive a
  suggested duration from the trimmed speech script and the profile's per-language speaking rate,
  clamp it to the profile's supported range, and present it as an editable default. The operator MAY
  override it anywhere within that range. The system MUST NOT perform a pre-generation script-fit
  check and MUST NOT reject a request because of script length. It MUST record the suggested duration,
  the effective duration, whether the operator overrode it, and the profile identity and speaking-rate
  field used. When delivery is rushed or clipped, the documented remedy is to raise the duration and
  regenerate.
- **FR-035**: The system MUST keep every ready downloaded model pinned to its resolved immutable commit,
  MUST check for newer repository revisions only after an explicit user action, and MUST represent a
  downloaded update as a separate inventory revision without automatically replacing, selecting, or
  deleting an existing revision.
- **FR-036**: A successful request bundle MUST retain the original image, reference-voice recording,
  derived voice representation, the decoded audio and video output of the generation, the assembled
  prompt actually submitted to the adapter, the final MP4, and sanitized metadata in application-owned
  local storage until its directory is removed through the filesystem; the UI MUST disclose this
  sensitive-data retention and local disk usage. Retention MUST NOT vary with output duration or bundle
  size: no artifact is dropped, downsampled, or expired because of a request's size.
- **FR-037**: The system MUST allow an available retained reference-voice file to be chosen through the
  filesystem upload picker for a later request only when it satisfies the new effective voice provider's
  constraints, MUST require and record a fresh ownership/permission attestation for every reuse, and
  MUST NOT carry consent forward or infer without the new request's valid attestation.
- **FR-038**: Before a model download or generation inference, the system MUST estimate the operation's
  required local storage and verify that completion would preserve a configurable safety reserve whose
  default is 10 GiB; otherwise it MUST block before network transfer/inference, report required and
  available space with manual cleanup guidance, and MUST NOT automatically delete models or request
  bundles. Because a run may take hours and write large decoded artifacts, the system MUST ALSO monitor
  free space periodically during every write stage. If the reserve would be breached mid-generation, it MUST stop the
  request with the same storage error, report required, available, and reserve values, remove the
  unpublished staging directory, and leave every published bundle and downloaded model untouched.
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

- **Generation Request**: Reference images and voice recording, motion prompt and speech script,
  requested parameters, selected model set and provider choices, consent, language, and request ID.
- **Reference Set**: The one-or-more still images anchoring identity and appearance plus the single
  recording anchoring voice timbre, with staged and source paths, measured properties and digests,
  speaker result, the bound consent attestation, any advisory voice origin, the derived voice
  representation, profile limits and refused kinds, and lifecycle state. The recording is a timbre
  anchor: never played back, never mixed into output, never treated as spoken content.
- **Assembled Prompt**: The motion description plus the speech script embedded as dialogue tags with the
  selected language, the token count actually submitted, any motion-prompt truncation override, and the
  adapter profile and prompt-capacity field it was validated against.
- **Duration Decision**: The suggested duration and the per-language speaking-rate entry it came from,
  any operator override, the effective duration and frame count, frame rate, resolution, audio sample
  rate, and the adapter profile the values were read from.
- **Generated Output**: The single joint audio/video result — decoded video and audio, final MP4,
  frame count, frame rate, dimensions, audio sample rate and channel layout, measured durations, script
  hash, language, non-silence result over the spoken region, digests, sizes, and lifecycle state.
- **Runtime Profile**: Device, precision, offload mode, quantization, accelerator peak-reserved ceiling,
  host system-memory ceiling, free-memory headroom, and capability warnings.
- **Generation Result**: Retained request-bundle path, MP4 path, effective parameters, timing, memory
  metrics, state, retained-artifact inventory and disk size, and error detail.
- **Request Bundle**: Stable request identity, artifact inventory, local directory and disk size,
  originating voice bundle if reused, advisory dependent bundle identities, availability and
  reconciliation state, creation time, and externally observed filesystem-removal state.
- **Model Profile**: Adapter identity and validated roles including native capabilities, plus every
  measured capability field — supported duration range, frame rate, resolutions, audio output, dialogue
  languages, per-language speaking rates, reference limits, prompt token capacity, precision policy,
  offload and quantization policy, and accelerator and host memory profiles.
- **Downloaded Model**: Normalized repository ID, resolved revision, adapter type, validated roles and
  native capabilities, compatibility result, local cache path, disk size, download state, active/in-use
  state, ownership of cached files, optional newer-revision availability, timestamps, and failure or
  deletion detail.
- **Model Download**: Request identity, normalized model source, progress, expected and received
  content, retry state, terminal validation result, and relationship to a downloaded model.
- **Model Set**: Selected video model plus optional dedicated voice-cloning and lip-sync models,
  resolved capability coverage, explicit provider choice for overlapping roles, cross-model
  compatibility result, and rejection details. A single profile may cover all three roles natively.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A first-time user can submit a valid image, motion prompt, speech script of any length,
  reference voice, selected language, and consent attestation and obtain a previewable, downloadable
  MP4 without editing source code.
- **SC-001a**: 100% of speech scripts are synthesized in full with no dropped or reordered content, and
  100% of motion-prompt truncations are recorded in metadata and displayed in the UI; zero truncations
  occur silently.
- **SC-002**: 100% of invalid parameter combinations in the documented bounds are rejected before
  model inference with a corrective message.
- **SC-003**: The application launches and completes its mocked end-to-end workflow on macOS and on
  a CPU-only host with zero accelerator-specific startup failures.
- **SC-004a**: 100% of adapter profiles are evaluated against both the accelerator-memory ceiling and the
  host system-memory ceiling before being marked ready; a profile breaching either is not selectable, and
  any required offload mode or quantized checkpoint is recorded in the profile rather than discovered at
  run time.
- **SC-004**: The default production preset completes on the display-attached target 16 GB accelerator
  without an unhandled OOM and with every heavy stage at or below 13.5 GiB peak allocator-reserved
  memory during acceptance testing.
- **SC-005a**: 100% of inference and decoding phases emit a monotonic completion fraction at least every
  few seconds, for outputs of every tested duration and under layer-wise offload.
- **SC-005**: Every accepted request displays ordered validation, reference preparation, prompt
  assembly, model loading, joint audio/video inference, decoding, export, verification, and publication
  phases and reaches success, failure, or cancellation.
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
- **SC-011a**: 100% of reference sets exceeding the effective profile's measured image limit are refused
  as reference errors before inference, and the application itself contributes no image-count maximum.
- **SC-011**: 100% of zero-face, multi-face, and unusable-mouth test inputs are rejected before video
  generation, while valid single-face inputs identify exactly one lip-sync target; 100% of generated
  videos that lose that target or introduce another face are rejected before lip-sync processing.
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
- **SC-020**: 100% of accepted requests carry the complete speech script, invoke the video model exactly
  once, use a duration within the effective profile's supported range, and report suggested duration,
  effective duration, whether it was overridden, frame rate, resolution, audio sample rate, and adapter
  profile identity. Zero requests are rejected for script length.
- **SC-021**: 100% of ready models continue using their recorded immutable commit until the user
  explicitly selects another revision; update checks perform no automatic download/replacement, and a
  downloaded update coexists as a separately selectable inventory entry.
- **SC-022**: 100% of successful request bundles retain the final MP4, all original inputs, derived
  voice/speech data, intermediates, and metadata across application restarts until their directories are
  externally removed; failed/cancelled requests retain no unpublishable partial artifacts.
- **SC-023**: 100% of available compatible retained reference-voice files can be re-uploaded through the
  filesystem picker, and every reuse records a new per-request consent attestation; absent/false
  re-consent causes zero inference regardless of prior attestations.
- **SC-024c**: 100% of model download and update-download attempts made while a generation is active are
  refused with a temporary read-only explanation, start no network transfer, and leave inventory
  unchanged; inspection, listing, and update checks remain available throughout.
- **SC-024b**: 100% of cancellations issued during a duration-scaling stage reach a terminal cancelled
  result within the documented bound, release every model lease and worker process, remove the staging
  directory, and modify zero published bundles.
- **SC-024a**: 100% of mid-generation reserve breaches stop the request with the storage error code,
  report required/available/reserve values, leave no unpublished staging directory behind, and modify
  zero published bundles or downloaded models.
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
- The inputs are one or more still images, a motion prompt, a separate speech script, and a
  reference-voice recording.
  The output uses cloned-voice speech with visible lip synchronization. Because the effective adapter
  generates audio and video as one process, any non-speech audio it produces alongside the speech is
  neither requested nor prohibited by the application.
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
- Neither the motion prompt nor the speech script has an application-imposed maximum length. The only
  length-sensitive behavior is truncating an over-capacity motion prompt at the video adapter's
  text-encoder limit, which is always reported as an override.
- The model library is read-only for the whole duration of an active generation. Because a run may take
  hours, an operator who needs to download a model during one is expected to cancel it first; v1 does not
  queue deferred downloads.
- Every successful bundle keeps the same artifact set regardless of size. Managing accumulated disk use
  remains the operator's responsibility through the filesystem.
- The target machine is one NVIDIA RTX 5080 with 16 GB of accelerator memory and 64 GB of host system
  memory. Both are hard ceilings that every adapter profile must satisfy, and a profile may rely on
  layer-wise or sequential CPU offload and a quantized checkpoint to do so.
- Output duration is bounded by the effective adapter profile and produced by exactly one joint
  audio/video generation. Inference time, by contrast, is explicitly unbounded: layer-wise and sequential
  CPU offload and quantized checkpoints are expected on the target hardware, no latency target or SLA
  applies, and a run taking hours is acceptable.
- The effective duration is chosen from the adapter profile's measured supported range so that the whole
  speech script fits. Requested duration is a preference; frame rate and resolution come from the profile.
  Trimming, time-stretching, or omitting speech to satisfy a preference is out of scope.
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
