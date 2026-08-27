# Tasks: Generate Image-Conditioned Video with Audio

**Input**: Design documents from `specs/001-generate-image-video/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Tests are required by the project constitution. Test tasks precede their corresponding
implementation tasks, ordinary tests remain offline, and real model/GPU tests are opt-in.

**Organization**: Tasks are grouped by user story so each story remains independently testable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: May run in parallel because it targets different files and has no incomplete dependency.
- **[Story]**: Maps the task to a user story from `spec.md`.
- Every task names the exact file or directory it changes.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the cross-platform Python project skeleton and dependency/test configuration.

- [ ] T001 Create the planned source, adapter, test, and runtime directories with package markers in `adapters/__init__.py`, `tests/contract/__init__.py`, `tests/integration/__init__.py`, `tests/unit/__init__.py`, and `outputs/.gitkeep`
- [ ] T002 Define bounded runtime dependencies for Python 3.11, including Diffusers, Gradio, image/audio processing, and platform-neutral PyTorch constraints in `requirements.txt`
- [ ] T003 [P] Define pytest, coverage, lint, formatting, and type-check dependencies in `requirements-dev.txt`
- [ ] T004 [P] Configure offline-by-default pytest discovery and `smoke`, `cuda`, and `mps` markers in `pytest.ini`
- [ ] T005 [P] Add Python caches, virtual environments, Hugging Face caches, `.env`, uploads, generated media, WAV files, and partial mux outputs to `.gitignore`
- [ ] T006 [P] Document non-secret runtime defaults for bind address, output root, video/audio profiles, retention, queue size, and offline stub mode in `.env.example`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared domain, configuration, device, error, logging, and test-double contracts.

**Critical**: No user-story implementation begins until this phase is complete.

### Foundational tests

- [ ] T007 [P] Write failing tests for generation records, seed derivation, state transitions, serialization, and audio metadata in `tests/unit/test_domain.py`
- [ ] T008 [P] Write failing CUDA→MPS→CPU selection, dtype, capability, and guarded memory-metric tests in `tests/unit/test_devices.py`
- [ ] T009 [P] Write failing settings and allowlisted video/audio profile validation tests, including gated-license acknowledgement, in `tests/unit/test_config.py`
- [ ] T010 [P] Write failing domain-error sanitization and stable error-code tests for model, OOM, audio, mux, codec, filesystem, and cancellation failures in `tests/unit/test_errors.py`

### Foundational implementation

- [ ] T011 Implement typed `ModelProfile`, `AudioProfile`, `GenerationRequest`, `EffectiveGenerationConfig`, `ProgressEvent`, `AudioArtifact`, `GenerationResult`, and metadata serialization in `domain.py`
- [ ] T012 [P] Implement the stable exception hierarchy, safe user messages, recovery suggestions, and framework-exception mapping helpers in `errors.py`
- [ ] T013 [P] Implement capability-checked device resolution, device/dtype compatibility, CUDA memory snapshots, peak reset, and safe cache cleanup in `devices.py`
- [ ] T014 Implement environment loading, `pathlib.Path` normalization, profile allowlists, frame rules, audio limits, output containment, and license acknowledgement in `config.py`
- [ ] T015 [P] Define injectable video/audio adapter protocols, progress callbacks, cancellation checks, and frame/waveform return contracts in `adapters/base.py`
- [ ] T016 [P] Configure redacted structured logging that omits tokens, prompts, and full local paths in `logging_config.py`
- [ ] T017 Create reusable fake video/audio adapters, sample images, waveform fixtures, temporary output roots, and CUDA/MPS capability fakes in `tests/conftest.py`

**Checkpoint**: Domain contracts and infrastructure pass offline tests; story work can start.

---

## Phase 3: User Story 1 — Generate a Video with Audio (Priority: P1) 🎯 MVP

**Goal**: Upload an image and prompts, generate video plus ambient soundtrack, and preview/download a
verified MP4 containing synchronized video and non-silent audio.

**Independent Test**: Run the stub profile from the Gradio handler with a valid image and prompts;
verify one playable video stream, one audible AAC audio stream within one frame of video duration,
effective seeds/parameters in metadata, and identical preview/download paths.

### Tests for User Story 1

- [ ] T018 [P] [US1] Write failing CogVideoX adapter tests for lazy loading, BF16/FP16 policy, model CPU offload, VAE slicing/tiling, seeded calls, and frame extraction in `tests/unit/test_cogvideox_adapter.py`
- [ ] T019 [P] [US1] Write failing audio tests for derived prompts/seeds, stereo waveform shape, finite samples, normalization, fades, exact duration, and non-silence detection in `tests/unit/test_audio.py`
- [ ] T020 [P] [US1] Write failing export tests for safe FFmpeg argument construction, silent-video staging, AAC muxing, dual-stream verification, atomic publication, and temporary cleanup in `tests/unit/test_export.py`
- [ ] T021 [P] [US1] Write the failing successful-generation service contract test with fake video/audio adapters and metadata assertions in `tests/contract/test_generation_service.py`
- [ ] T022 [P] [US1] Write failing Gradio handler contract tests for image/prompt controls, optional audio prompt, preview path, download path, and final status in `tests/contract/test_ui_contract.py`
- [ ] T023 [US1] Write the failing offline end-to-end stub test that creates and probes a synchronized MP4 with audible audio in `tests/integration/test_stub_end_to_end.py`

### Implementation for User Story 1

- [ ] T024 [P] [US1] Implement the deterministic bounded pan/zoom frame generator used for offline workflows in `adapters/stub.py`
- [ ] T025 [P] [US1] Implement the deterministic stereo tone/noise-envelope soundtrack generator used for offline workflows in `adapters/stub_audio.py`
- [ ] T026 [US1] Implement audio prompt fallback, deterministic audio-seed derivation, waveform normalization, trim/pad, fades, WAV staging, and non-silence validation in `audio.py`
- [ ] T027 [US1] Implement silent MP4 export, cross-platform bundled-FFmpeg lookup, safe AAC muxing, dual-stream/duration verification, atomic rename, and intermediate cleanup in `export.py`
- [ ] T028 [P] [US1] Implement the `zai-org/CogVideoX-5b-I2V` adapter with model-compatible image/frame arguments and CUDA memory controls in `adapters/cogvideox.py`
- [ ] T029 [P] [US1] Implement the gated `stabilityai/stable-audio-open-small` adapter with configurable replacement model, stereo 44.1 kHz output, seed, duration, and unload support in `adapters/stable_audio.py`
- [ ] T030 [US1] Implement `VideoGenerationEngine` orchestration for validation, sequential video/audio model residency, stub/real adapter selection, metadata, and terminal results in `pipeline.py`
- [ ] T031 [US1] Build the Gradio `Blocks` UI with image upload, motion prompt, optional audio prompt, advanced seed/frames/FPS/guidance/model controls, video player, and download button in `app.py`
- [ ] T032 [US1] Connect the Gradio submit handler to `VideoGenerationEngine`, map results to preview/download outputs, enforce loopback launch defaults, and disable public sharing in `app.py`

**Checkpoint**: The P1 workflow creates a synchronized MP4 with audio using stubs offline and exposes
the real video/audio profiles behind explicit model-access configuration.

---

## Phase 4: User Story 2 — Understand Progress and Resource Use (Priority: P2)

**Goal**: Show ordered loading, preprocessing, video inference, audio generation, mux, export, and
terminal status with device-appropriate memory information.

**Independent Test**: Drive the engine with event-emitting fakes and verify monotonic phase/progress
updates plus CUDA memory metrics or an explicit unavailable state on MPS/CPU.

### Tests for User Story 2

- [ ] T033 [P] [US2] Write failing ordered, monotonic, request-scoped progress-event contract tests covering video, audio, mux, and terminal phases in `tests/contract/test_generation_service.py`
- [ ] T034 [P] [US2] Write failing UI progress/status and CUDA-versus-unavailable memory rendering tests in `tests/contract/test_ui_contract.py`

### Implementation for User Story 2

- [ ] T035 [US2] Emit sanitized phase events, inference-step fractions, per-stage timing, and terminal progress from `VideoGenerationEngine` in `pipeline.py`
- [ ] T036 [P] [US2] Add request-scoped CUDA allocated/reserved/peak sampling and sequential-stage memory logging in `devices.py`
- [ ] T037 [US2] Render `gr.Progress`, phase status, elapsed time, effective device/profile, and memory summaries without prompts or full paths in `app.py`

**Checkpoint**: Long-running generation is visibly progressing through both ML stages and muxing,
with accurate device-aware diagnostics.

---

## Phase 5: User Story 3 — Recover from Unsupported or Exhausted Environments (Priority: P3)

**Goal**: Launch safely on macOS/CPU and return actionable, cleaned-up failures for invalid input,
unsupported backends, OOM, model/license access, audio generation, muxing, and cancellation.

**Independent Test**: Inject each documented failure into the offline engine and verify one safe
terminal result, no published MP4, no leaked temporary media, and corrective UI guidance.

### Tests for User Story 3

- [ ] T038 [P] [US3] Write failing prompt/audio-prompt, seed, frame/FPS/guidance, image orientation/mode/size, path-containment, and model-profile validation tests in `tests/unit/test_validation.py`
- [ ] T039 [P] [US3] Write failing OOM, model/auth/license, unsupported MPS operation, audio-generation, mux/codec, filesystem, and unexpected-error translation tests in `tests/unit/test_errors.py`
- [ ] T040 [P] [US3] Write failing cancellation, concurrency-limit, partial-output cleanup, and cached-model release contract tests in `tests/contract/test_generation_service.py`
- [ ] T041 [P] [US3] Write failing UI tests ensuring failures clear video/download outputs and display ordered recovery suggestions in `tests/contract/test_ui_contract.py`

### Implementation for User Story 3

- [ ] T042 [US3] Implement image preprocessing, decompression-bomb limits, request/profile preflight, duration limits, output containment, and license/access validation in `pipeline.py`
- [ ] T043 [US3] Implement stage-aware exception translation, OOM recovery, failed-artifact cleanup, and safe terminal metadata in `pipeline.py`
- [ ] T044 [US3] Implement cooperative cancellation and a process-wide single-generation lock compatible with Gradio queueing in `pipeline.py`
- [ ] T045 [US3] Configure a bounded Gradio queue with concurrency one, cancellation wiring, cleared failure outputs, and actionable recovery messages in `app.py`
- [ ] T046 [P] [US3] Add opt-in real-backend smoke tests for MPS compatibility and CUDA video/audio model loading without affecting the offline suite in `tests/integration/test_mps_smoke.py` and `tests/integration/test_cuda_smoke.py`

**Checkpoint**: All specified failure modes terminate safely and the app remains testable without
CUDA or model downloads.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Complete deployment documentation, security/license review, and cross-platform release
validation after the desired user stories are finished.

- [ ] T047 [P] Write architecture, configuration, model-license/access, artifact-retention, and local-security documentation in `README.md`
- [ ] T048 [P] Update executable macOS and Windows setup, CUDA verification, audio-model access, launch, test, and OOM/mux troubleshooting instructions in `specs/001-generate-image-video/quickstart.md`
- [ ] T049 [P] Add explicit Hub token, gated-model license acknowledgement, remote-code prohibition, loopback binding, and media-retention checks in `tests/unit/test_config.py`
- [ ] T050 Run and fix formatting, linting, type checking, offline tests, and coverage gates configured by `requirements-dev.txt` across `app.py`, `pipeline.py`, supporting modules, `adapters/`, and `tests/`
- [ ] T051 Run the RTX 5080 CUDA acceptance benchmark for 720x480, 49 frames, 8 FPS, sequential audio generation, dual-stream MP4, synchronization, and <15.5 GiB peak memory in `tests/integration/test_cuda_smoke.py`
- [ ] T052 Validate every command and expected output in `specs/001-generate-image-video/quickstart.md` on a clean macOS environment and record the Windows validation checklist in `README.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Starts immediately.
- **Foundational (Phase 2)**: Depends on Setup and blocks every user story.
- **US1 (Phase 3)**: Depends on Foundational; delivers the complete audio/video MVP.
- **US2 (Phase 4)**: Depends on Foundational and the P1 engine/UI extension points; its test design
  can begin in parallel with late P1 implementation.
- **US3 (Phase 5)**: Depends on Foundational and the P1 engine lifecycle; failure-test design can
  begin in parallel with late P1 implementation.
- **Polish (Phase 6)**: Depends on all user stories selected for release.

### User Story Dependency Graph

```text
Setup -> Foundation -> US1 (audio/video MVP) -> Polish
                         |-> US2 (progress/metrics) -|
                         |-> US3 (recovery/safety) --|
```

- **US1** is independently testable with stub video/audio adapters and is the required MVP.
- **US2** adds observable progress without changing US1 generation outputs.
- **US3** adds failure recovery without changing the successful US1 contract.
- After the P1 engine contract exists, US2 and US3 can proceed in parallel.

### Within Each User Story

- Write the story's tests and verify they fail for the intended missing behavior.
- Implement adapters/domain behavior before orchestration that consumes them.
- Implement engine behavior before UI integration.
- Publish a final MP4 only after video, audio, mux, and verification all succeed.
- Run the independent story test at the phase checkpoint before proceeding.

## Parallel Opportunities

- T003–T006 are independent setup-file tasks.
- T007–T010 are independent failing-test tasks; T012, T013, T015, and T016 target separate modules.
- T018–T022 can be written in parallel; T024, T025, T028, and T029 implement separate adapters.
- T033 and T034 can run in parallel; T036 is independent of the engine/UI implementations.
- T038–T041 can run in parallel; T046 is isolated behind opt-in markers.
- T047–T049 can run in parallel before the integrated quality/acceptance passes.

## Parallel Examples

### User Story 1

```text
T018: tests/unit/test_cogvideox_adapter.py
T019: tests/unit/test_audio.py
T020: tests/unit/test_export.py
T021: tests/contract/test_generation_service.py
T022: tests/contract/test_ui_contract.py

After tests fail as expected:
T024: adapters/stub.py
T025: adapters/stub_audio.py
T028: adapters/cogvideox.py
T029: adapters/stable_audio.py
```

### User Story 2

```text
T033: tests/contract/test_generation_service.py
T034: tests/contract/test_ui_contract.py
T036: devices.py
```

### User Story 3

```text
T038: tests/unit/test_validation.py
T039: tests/unit/test_errors.py
T040: tests/contract/test_generation_service.py
T041: tests/contract/test_ui_contract.py
T046: tests/integration/test_mps_smoke.py and tests/integration/test_cuda_smoke.py
```

## Implementation Strategy

### MVP First

1. Complete Setup and Foundational phases.
2. Complete US1 tests, offline stubs, audio normalization/mux, engine, and UI.
3. Stop and validate the synchronized audio/video MP4 independently.
4. Integrate real CogVideoX and Stable Audio profiles behind explicit access configuration.

### Incremental Delivery

1. **Foundation**: Typed, testable, cross-platform control plane.
2. **US1**: Complete image + prompt → video + generated audio → downloadable MP4 workflow.
3. **US2**: Detailed progress and memory visibility.
4. **US3**: Cross-platform fallback and production-grade failure recovery.
5. **Polish**: License/security review, clean-environment verification, and RTX 5080 acceptance.

## Completion Criteria

- All 52 tasks retain checkbox, sequential task ID, required story label, and exact file path.
- The P1 independent test proves both video and audible synchronized audio are present.
- Offline tests require neither network access nor accelerator hardware.
- CUDA/MPS/model tests run only through explicit pytest markers.
- No temporary silent video or WAV is published as a successful result.
