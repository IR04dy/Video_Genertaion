# Tasks: Generate Image-Conditioned Lip-Synced Video

**Input**: Design documents from `specs/001-generate-image-video/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Required. The project constitution makes test-first NON-NEGOTIABLE, so every test task
precedes its implementation task. Ordinary tests are offline and download no weights; real
model/GPU tests are opt-in behind markers.

**Organization**: Tasks are grouped by user story so each story remains independently testable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: May run in parallel — different files, no incomplete dependency.
- **[Story]**: Maps the task to a user story from `spec.md`.
- Every task names the exact file or directory it changes.

## Standing rules for every task

- Model-specific values (duration range, frame rate, resolution, audio sample rate, languages,
  speaking rates, reference limits, token capacity) live **only** in adapter profiles. A task that
  puts one in shared code, a shared constant, or a shared assertion is incorrectly implemented.
- No test may assert an upper bound on wall-clock time. Inference is unbounded by decision.
- `trust_remote_code` stays false everywhere. No test may enable it to make a load succeed.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the cross-platform Python skeleton and dependency/test configuration.

- [X] T001 Create the planned source, adapter, and test directories with package markers in `adapters/__init__.py`, `tests/__init__.py`, `tests/contract/__init__.py`, `tests/integration/__init__.py`, `tests/unit/__init__.py`, and `outputs/.gitkeep`
- [X] T002 Declare bounded Python 3.11 runtime dependencies — Diffusers and Transformers releases exporting the H3 classes, Accelerate, a reviewed quantization backend, Hugging Face Hub, Safetensors, Gradio, Pillow, NumPy, SoundFile/librosa, imageio, imageio-ffmpeg, Pydantic 2.x, psutil, filelock — with PyTorch deliberately excluded so the platform wheel is never replaced, in `requirements.txt`
- [X] T003 [P] Declare pytest, pytest-cov, lint, format, and type-check dependencies in `requirements-dev.txt`
- [X] T004 [P] Configure offline-by-default discovery and the `stack_compatibility`, `cuda`, `mps`, and `model_download` markers in `pytest.ini`
- [X] T005 [P] Ignore Python caches, virtual environments, `.model-cache/`, `outputs/` except its placeholder, `.env`, and generated media in `.gitignore`
- [X] T006 [P] Document non-secret runtime options — bind address, disk reserve, dual resource ceilings, runtime profile, cancellation grace period — with no bundle-root override, in `.env.example`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Prove the stack loads, then establish domain, config, device, error, logging, storage, and
adapter-protocol foundations.

**Critical**: No user-story work begins until this phase completes.

### Blocking dependency gate

> **Gate outcome (resolved).** The gate found that the `Ref2VA/` subfolder cannot be loaded with
> `trust_remote_code=False`: its `model_index.json` names classes no upstream release exports, and its
> `video_vae`/`audio_vae` configs carry `auto_map` entries pointing at bundled `.py` modules. The
> architecture therefore loads the repository **root** `modular_model_index.json` as a
> `MiniMaxH3ModularPipeline` with `load_components(workflow="ref2va")`, which uses only upstream classes
> and ships no Python beside its weights. Every downstream task naming an H3 class must use the root
> names. See `research.md` → "No remote code required — via the root modular layout".


- [X] T007 Write the clean-environment gate asserting every class named by the repository root's `modular_model_index.json` — `MiniMaxH3ModularPipeline`, `MiniMaxH3Blocks`, `MiniMaxH3Transformer3DModel`, `AutoencoderKLMiniMaxH3`, `AutoencoderKLMiniMaxH3Audio`, `MiniMaxH3Scheduler`, `Qwen3VLForConditionalGeneration`, `Qwen3VLProcessor`, `Qwen2TokenizerFast` — imports with `trust_remote_code=False`, and that no root component config declares an `auto_map`, in `tests/integration/test_stack_compatibility.py`
- [X] T008 Resolve and pin the exact Diffusers/Transformers releases that satisfy T007, recording the resolved versions in `requirements.txt`; if no release exports them, stop and mark the profile `incompatible` rather than enabling remote code — **resolved: `diffusers==0.40.0`, `transformers==5.16.1`, torch floor 2.5**. Verified 12/12 on the Windows RTX 5080 host (Python 3.11.9, torch 2.13.0+cu130); on macOS 7 of the 12 skip below the torch 2.5 floor, so the production run is the one that closes this task

### Foundational tests

- [X] T009 [P] Write failing tests for typed requests, profiles, plans, bundles, events, and results with serialization round-trips in `tests/unit/test_domain.py`
- [X] T010 [P] Write failing CUDA→MPS→CPU resolution, dtype, accelerator snapshot, and **host system-memory** measurement tests in `tests/unit/test_devices.py`
- [X] T011 [P] Write failing tests asserting a profile is rejected when it breaches either the accelerator ceiling or the host RAM ceiling, and that offload mode and quantization are read from the profile rather than discovered, in `tests/unit/test_resource_ceilings.py`
- [X] T012 [P] Write failing stable-error-code and sanitization tests covering `validation`, `consent`, `face`, `reference`, `language`, `duration`, `oom`, `host_memory`, `disk`, `generation`, `export`, `codec`, `cancelled`, and `internal` in `tests/unit/test_errors.py`
- [X] T013 [P] Write failing path-containment tests for fixed roots, symlink and Windows reparse-point refusal, UNC/drive/alternate-stream rejection, and server-UUID staging names in `tests/unit/test_paths.py`
- [X] T014 [P] Write failing settings tests for fixed `outputs/` and `.model-cache/` roots, the configurable 10 GiB reserve, both resource ceilings, and the absence of any bundle-root override in `tests/unit/test_config.py`

### Foundational implementation

- [X] T015 Implement the Pydantic domain records from `data-model.md` — `ModelProfile` with all measured capability fields, `ReferenceSet`, `AssembledPrompt`, `DurationDecision`, `GeneratedOutput`, `ProgressEvent`, `GenerationResult`, `ErrorDetail` — in `domain.py`
- [X] T016 [P] Implement the stable exception hierarchy, safe user messages, ordered recovery suggestions, and framework-exception mapping in `errors.py`
- [X] T017 [P] Implement capability-checked device resolution, dtype policy, accelerator allocated/reserved/free snapshots, and host resident-memory sampling in `devices.py`
- [X] T018 [P] Implement redacted structured logging that omits tokens, prompts, absolute upload paths, and voice data in `logging_config.py`
- [X] T019 Implement settings loading, `pathlib.Path` normalization, fixed roots, disk reserve, and dual resource ceilings in `config.py`
- [X] T020 Implement path containment, symlink and reparse-point refusal, atomic manifest writes under a cross-platform lock, and disk estimation in `storage.py`
- [X] T021 [P] Define the joint audio/video adapter protocol and the capability-profile contract, with progress callbacks and cooperative cancellation checks, in `adapters/base.py`
- [X] T022 Create reusable fixtures — fake adapters, sample images, waveform fixtures, temporary roots, capability fakes, and a **fixture profile whose measured values all differ from H3's** — in `tests/conftest.py`

**Checkpoint**: The stack is proven loadable, foundations pass offline, and story work can start.

---

## Phase 3: User Story 1 — Generate a Lip-Synced Video (Priority: P1) 🎯 MVP

**Goal**: Upload images and a reference voice, supply motion and speech text, and receive a verified
MP4 whose generated speech uses the reference voice with natively synchronized lip movement.

**Independent Test**: Run the stub profile end to end offline; verify one video stream and one
non-silent speech stream agreeing within one frame at the profile's frame rate, a complete retained
bundle, and identical preview/download paths.

### Tests for User Story 1

- [X] T023 [P] [US1] Write failing reference tests for one-or-more images with **no application maximum**, per-image exactly-one-face validation, profile-limit enforcement, single audio anchor, and video-reference rejection in `tests/unit/test_references.py`
- [X] T024 [P] [US1] Write failing consent tests for per-request attestation defaulting and resetting to false, reset on reference-audio change, and binding to request ID plus audio SHA-256 in `tests/unit/test_consent.py`
- [X] T025 [P] [US1] Write failing image/audio inspection, EXIF/RGB normalization, decompression-bomb bounds, and retained-path detection tests in `tests/unit/test_media.py`
- [X] T026 [P] [US1] Write failing prompt tests for dialogue-tag assembly with the selected language, token measurement against profile capacity, motion-prompt truncation recorded as an override, and **speech-script content never truncated, dropped, or reordered** in `tests/unit/test_prompting.py`
- [X] T027 [P] [US1] Write failing duration tests for suggestion from script length and per-language speaking rate, clamping to the supported range, missing-language fallback, operator override acceptance, out-of-range override refusal, and the **absence of any length-based rejection path** in `tests/unit/test_duration_selection.py`
- [X] T028 [P] [US1] Write failing validation tests for language membership, seed bounds, guidance against profile bounds, and request preflight ordering in `tests/unit/test_validation.py`
- [X] T029 [P] [US1] Write failing export tests for safe argument construction, container export, dual-stream verification, non-silence over the spoken region, duration agreement within one frame, atomic publication, and temporary cleanup in `tests/unit/test_export.py`
- [X] T030 [P] [US1] Write failing bundle tests validating a published manifest against `contracts/request-bundle.schema.json`, including multiple retained `original_image` artifacts and duration provenance fields, in `tests/unit/test_request_bundle.py`
- [X] T031 [P] [US1] Write the failing generation-service contract test for the successful path, preconditions, and result guarantees with fake adapters in `tests/contract/test_generation_service.py`
- [X] T032 [P] [US1] Write failing Gradio handler contract tests for multi-image upload, reference-audio upload with the timbre-anchor rule displayed, editable suggested duration, consent reset, preview path, and download path in `tests/contract/test_ui_contract.py`
- [X] T033 [US1] Write the failing offline end-to-end stub test that publishes a complete bundle and probes the final MP4 in `tests/integration/test_stub_end_to_end.py`
- [X] T034 [P] [US1] Write the failing profile-agnostic suite that re-runs reference, prompt, duration, and export assertions against the fixture profile from T022 in `tests/unit/test_profile_agnostic.py`

### Implementation for User Story 1

- [X] T035 [P] [US1] Implement image and audio inspection, EXIF/RGB normalization, bounded decode, per-image face and mouth detection, and retained-path detection in `media.py`
- [X] T036 [US1] Implement `ReferenceSet` staging — copying references into `outputs/.work/<request-id>/inputs/`, digest computation, profile-limit enforcement, video-reference refusal, and consent binding — in `execution.py`
- [X] T037 [US1] Implement dialogue-tag prompt assembly, token measurement against profile capacity, motion-prompt truncation with recorded override, and retention of the assembled prompt in `prompting.py`
- [X] T038 [US1] Implement duration suggestion from script length and per-language speaking rate, clamping, missing-language fallback, override handling, and `DurationDecision` construction in `execution.py`
- [X] T039 [P] [US1] **[implemented in Stage 1, hoisted after T021]** Implement the deterministic offline stub adapter producing fixture frames and a fixture waveform with a declared capability profile in `adapters/stub.py`
- [ ] T040 [P] [US1] Implement the `minimax-h3` adapter — loading the **repository-root modular pipeline** (`MiniMaxH3ModularPipeline` / `MiniMaxH3Blocks` via `modular_model_index.json`) with `trust_remote_code=False`, declared offload mode and quantization, and a capability profile carrying every measured field — in `adapters/minimax_h3.py`. **Not the `Ref2VA/` subfolder**: T007 proved its VAE configs carry `auto_map` and it names classes no Diffusers release exports, so that path requires remote code and is prohibited
- [X] T041 [US1] Implement reviewed adapter fingerprints and measured capability-profile resolution in `model_registry.py`
- [X] T042 [US1] Implement container export, dual-stream verification, non-silence checking over the spoken region, duration agreement within one frame, and atomic publication in `export.py`
- [X] T043 [US1] Implement bundle publication — staging, manifest writing, artifact inventory, and the atomic rename to `outputs/<request-id>/` — in `storage.py`
- [X] T044 [US1] Implement `VideoGenerationEngine` orchestration for validation, prompt assembly, duration decision, the single joint generation, decode, export, verify, and publish in `pipeline.py`
- [X] T045 [US1] Build the Gradio Blocks UI with multi-image upload, reference-audio upload displaying the timbre-anchor and different-words rule, motion prompt, speech script, language selector, editable suggested duration, consent checkbox, advanced controls, video player, and download button in `app.py`
- [X] T046 [US1] Wire the submit handler to `VideoGenerationEngine`, map results to preview and download outputs, reset consent after every submit and on audio change, and enforce loopback binding with sharing disabled in `app.py`

**Checkpoint**: The P1 workflow produces a verified MP4 offline with stubs and exposes the real H3
profile behind explicit model configuration.

---

## Phase 4: User Story 2 — Select and View Models (Priority: P2)

**Goal**: Supply Hugging Face URLs, download compatible models with visible status, and select, update,
or delete inventory entries.

**Independent Test**: With a fake Hub, inspect and download a model, restart, confirm it stays ready
and selectable offline, then preview and confirm its deletion with measured reclaimed bytes.

### Tests for User Story 2

- [ ] T047 [P] [US2] Write failing URL tests for canonical HTTPS `huggingface.co` roots, the commit-pinned versus `tracking_ref` `oneOf` in `contracts/model-source.schema.json`, and rejection of credentials, queries, fragments, blob paths, and alternate hosts in `tests/unit/test_model_urls.py`
- [ ] T048 [P] [US2] Write failing registry tests asserting inspection matches adapter fingerprints, records every measured profile field, and marks a profile `incompatible` when any field is missing or unmeasured in `tests/unit/test_model_registry.py`
- [ ] T049 [P] [US2] Write failing tests asserting **no license field is ever requested, parsed, persisted, or displayed**, even when fake Hub metadata contains one, in `tests/unit/test_model_license_exclusion.py`
- [ ] T050 [P] [US2] Write failing dependency-closure tests for pinned auxiliary snapshots, digests, and refusal of hidden downloads during inference in `tests/unit/test_model_dependencies.py`
- [ ] T051 [P] [US2] Write failing inventory tests for atomic replacement, restart reconciliation, lease protection, and deletion eligibility in `tests/unit/test_inventory.py`
- [ ] T052 [P] [US2] Write failing manual-update tests asserting `refresh()` never contacts the network, `check_for_update()` runs only on explicit action, and a downloaded update becomes a separate revision in `tests/unit/test_model_updates.py`
- [ ] T053 [P] [US2] Write failing disk-preflight tests for pre-transfer estimation, the configurable reserve, and reporting required/available/reserve without mutation in `tests/unit/test_disk_preflight.py`
- [ ] T054 [P] [US2] Write the failing model-catalog-service contract test covering inspect, download, retry, update, delete, discard-partial, and **refusal of downloads while a generation is active** in `tests/contract/test_model_catalog_service.py`
- [ ] T055 [P] [US2] Write the failing offline restart test proving a ready revision stays selectable with no network in `tests/integration/test_inventory_restart.py`
- [ ] T056 [P] [US2] Write the opt-in real-download test behind the `model_download` marker in `tests/integration/test_model_download.py`

### Implementation for User Story 2

- [ ] T057 [US2] Implement URL normalization, `validate_repo_id()` rules, tracking-ref versus commit resolution, and the pinned `https://huggingface.co` endpoint in `model_catalog.py`
- [ ] T058 [US2] Implement metadata inspection with `trust_remote_code=False`, remote-kernel disablement, adapter fingerprint matching, dependency-closure resolution, and dual-ceiling compatibility in `model_catalog.py`
- [ ] T059 [US2] Implement immutable snapshot download into `.model-cache/`, progress events, resume, digest and required-file verification, and ready-state transition in `model_catalog.py`
- [ ] T060 [US2] Implement atomic inventory persistence, startup reconciliation, stale-state recovery, and lease acquisition and release in `model_catalog.py`
- [ ] T061 [US2] Implement `check_for_update()` and `download_update()` as explicit user actions producing separate revisions in `model_catalog.py`
- [ ] T062 [US2] Implement deletion preview with a short-lived confirmation token, eligibility rechecks under lock, shared-blob-preserving revision deletion, and measured reclaimed bytes in `model_catalog.py`
- [ ] T063 [US2] Implement refusal of downloads and update downloads while a generation is active, keeping listing, inspection, and update checks available, in `model_catalog.py`
- [ ] T064 [US2] Build the Model Library UI — URL entry, role selection, inspection summary with measured profile fields and no license data, download progress, inventory table, update check, and confirmed deletion — in `app.py`

**Checkpoint**: Models can be acquired, inspected, updated, and removed entirely offline after download.

---

## Phase 5: User Story 3 — Understand Progress and Resource Use (Priority: P3)

**Goal**: Show ordered phase progress with a completion fraction during the long stages, plus
accelerator and host memory information.

**Independent Test**: Drive the engine with event-emitting fakes and verify ordered phases, monotonic
fractions during `generating` and `decoding`, and both memory readings, with no latency assertion.

### Tests for User Story 3

- [ ] T065 [P] [US3] Write failing progress-contract tests for the ordered phase set, request-scoped events, monotonic fractions within a stage, and exactly one terminal event in `tests/contract/test_generation_service.py`
- [ ] T066 [P] [US3] Write failing UI tests for phase status rendering, completion fraction display, and accelerator-versus-unavailable plus host memory summaries in `tests/contract/test_ui_contract.py`
- [ ] T067 [US3] Write the failing long-runtime test proving fractions keep arriving during a slow fake generation and that **no assertion bounds elapsed time**, in `tests/integration/test_long_runtime.py`

### Implementation for User Story 3

- [X] T068 [US3] Emit sanitized ordered phase events with per-stage timing and a monotonic completion fraction during generation and decoding from `pipeline.py`
- [X] T069 [P] [US3] Add request-scoped accelerator allocated/reserved/peak and host resident-memory sampling per stage in `devices.py`
- [ ] T070 [US3] Render progress, phase status, elapsed time, effective device and profile, and both memory summaries without prompts or absolute paths in `app.py`

**Checkpoint**: A multi-hour run is visibly progressing and never looks stalled.

---

## Phase 6: User Story 4 — Recover from Unsupported or Exhausted Environments (Priority: P4)

**Goal**: Launch safely on macOS and CPU, and return actionable cleaned-up failures for every
documented error class.

**Independent Test**: Inject each documented failure offline and verify one safe terminal result, no
published bundle, no leaked staging directory, and ordered recovery guidance.

### Tests for User Story 4

- [ ] T071 [P] [US4] Write failing translation tests for OOM, host-memory breach, model access, unsupported backend, generation, export, codec, and filesystem failures in `tests/unit/test_errors.py`
- [ ] T072 [P] [US4] Write failing cancellation tests asserting checks at bounded intervals inside long stages, released leases, removed staging, and one terminal cancelled result — with no elapsed-time assertion — in `tests/contract/test_generation_service.py`
- [ ] T073 [P] [US4] Write failing mid-write disk tests asserting the reserve is monitored during writes, the run stops with the storage error, staging is removed, and published bundles are untouched in `tests/unit/test_disk_preflight.py`
- [ ] T074 [P] [US4] Write failing UI tests ensuring failures clear video and download outputs and show ordered recovery suggestions in `tests/contract/test_ui_contract.py`
- [ ] T075 [P] [US4] Add the opt-in MPS and CUDA smoke tests in `tests/integration/test_mps_smoke.py` and `tests/integration/test_cuda_smoke.py`

### Implementation for User Story 4

- [ ] T076 [US4] Implement stage-aware exception translation, OOM and host-memory recovery guidance, failed-artifact cleanup, and safe terminal metadata in `pipeline.py`
- [ ] T077 [US4] Implement cooperative cancellation checked at bounded intervals in every long stage, with resource release and bounded staging cleanup, in `pipeline.py`
- [ ] T078 [US4] Implement periodic free-space monitoring during every write stage, stopping with the storage error while leaving published bundles untouched, in `storage.py`
- [ ] T079 [US4] Implement the process-wide single-generation lock and bounded Gradio queue with cancellation wiring in `pipeline.py` and `app.py`
- [ ] T080 [US4] Implement startup orphan-staging recovery using owner and lock markers before bounded cleanup in `storage.py`

**Checkpoint**: Every documented failure terminates safely and the app remains testable without CUDA.

---

## Phase 7: Read-Only Request History

**Goal**: Expose retained bundles read-only, with advisory voice-reuse dependencies.

**Independent Test**: Publish bundles offline, refresh history, verify preview, artifact inventory,
size, dependency links, and that no mutation control exists.

- [ ] T081 [P] Write failing history-service contract tests for the fixed scan root, absence of any mutation operation, and safe projections in `tests/contract/test_request_history_service.py`
- [ ] T082 [P] Write failing scanner tests for manifest validation, UUID directory filtering, `.work` exclusion, and containment checks in `tests/unit/test_request_history.py`
- [ ] T083 Write the failing reconciliation test for externally deleted and corrupt origins that never repairs or cascades in `tests/integration/test_history_reconciliation.py`
- [ ] T084 Implement the read-only scanner, manifest validation, advisory dependency graph, and safe projections in `request_history.py`
- [ ] T085 Implement the filesystem voice-reuse boundary — digest matching, origin recording, and a required fresh consent attestation — in `execution.py`
- [ ] T086 Build the read-only History panel with preview, artifact rows, dependency summary, disk summary, and warnings, wired to no mutation control, in `app.py`

---

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T087 [P] Write architecture, configuration, retention, and local-security documentation in `README.md`
- [ ] T088 [P] Validate every command and expected output in `specs/001-generate-image-video/quickstart.md` on a clean macOS environment
- [ ] T089 [P] Add explicit Hub-token, remote-code-prohibition, loopback-binding, and plaintext-disclosure checks in `tests/unit/test_config.py`
- [ ] T090 Run and fix formatting, linting, type checking, the offline suite, and coverage gates across all modules, `adapters/`, and `tests/`
- [ ] T091 **[hoisted to Stage 0 — blocked: needs the Windows RTX 5080 host]** Measure the **real supported duration ceiling** for the H3 profile on the RTX 5080 and record it as a profile field rather than assuming the card's stated 15 s, in `adapters/minimax_h3.py`. Source the value from `MiniMaxH3ModularPipeline.max_duration` and confirm it against an actual generation; run `spikes/h3_feasibility.py` first
- [ ] T092 **[hoisted to Stage 0 — blocked: needs the Windows RTX 5080 host]** Measure resident host footprint per precision against 64 GB and peak allocator-reserved memory against 13.5 GiB, recording both in the profile, in `tests/integration/test_cuda_smoke.py`. The `ref2va` working set is 134.12 GiB at BF16, so establish which quantization is actually resident before T040 declares one; run `spikes/h3_feasibility.py --stage load` first
- [ ] T093 Run the Windows RTX 5080 acceptance — clean install, one joint `ref2va`-workflow generation, zero hidden downloads during inference, dual-ceiling compliance, complete retained artifacts, and audio/video duration agreement — in `tests/integration/test_cuda_smoke.py`
- [ ] T094 Record measured wall-clock baselines in `README.md` **without introducing any SLA or latency gate**

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (Phase 1)**: starts immediately.
- **Foundational (Phase 2)**: depends on Setup. **T007–T008 gate everything** — if the H3 classes do
  not load with `trust_remote_code=False`, stop and resolve dependencies before any further work.
- **US1 (Phase 3)**: depends on Foundational. Delivers the MVP.
- **US2 (Phase 4)**: depends on Foundational. Independent of US1 except for shared domain records.
- **US3 (Phase 5)**: depends on the US1 engine extension points.
- **US4 (Phase 6)**: depends on the US1 engine lifecycle.
- **History (Phase 7)**: depends on US1 bundle publication.
- **Polish (Phase 8)**: depends on all selected stories.

### User story graph

```text
Setup → Foundational(gate) → US1 (MVP) → History → Polish
                           ├→ US2 (catalog) ─────────┤
                           ├→ US3 (progress) ────────┤
                           └→ US4 (recovery) ────────┘
```

### Within each story

- Write the story's tests and confirm they fail for the intended reason.
- Implement adapters and domain behavior before the orchestration that consumes them.
- Implement engine behavior before UI integration.
- Publish a bundle only after generation, decode, export, and verification all succeed.

## Parallel opportunities

- T003–T006 are independent setup files.
- T009–T014 are independent failing-test files; T016, T017, T018, and T021 target separate modules.
- T023–T032 and T034 can be written in parallel; T035, T039, and T040 implement separate modules.
- T047–T056 are independent; T065–T067 and T071–T075 likewise.
- T087–T089 can run in parallel before the integrated quality and acceptance passes.

## Implementation strategy

### MVP first

1. Complete Setup and Foundational, treating T007–T008 as a hard gate.
2. Complete US1 tests, then stubs, references, prompting, duration, export, engine, and UI.
3. Validate the offline stub bundle independently.
4. Integrate the real H3 Ref2VA profile behind explicit model configuration.

### Incremental delivery

1. **Foundation**: typed, testable, cross-platform control plane with a proven dependency stack.
2. **US1**: complete images + voice + prompts → verified MP4 workflow.
3. **US2**: model acquisition, inventory, updates, and deletion.
4. **US3**: progress and dual memory visibility for long runs.
5. **US4**: failure recovery, cancellation, and cross-platform safety.
6. **History**: read-only retained-bundle browsing and voice reuse.
7. **Polish**: measured hardware baselines and release validation.

## Completion criteria

- All 94 tasks retain a checkbox, sequential ID, required story label, and exact file path.
- The P1 independent test proves a verified MP4 with non-silent synchronized speech.
- Offline tests require neither network access nor accelerator hardware.
- The profile-agnostic suite passes against a fixture profile with entirely different values.
- No test asserts an upper bound on elapsed time.
- CUDA, MPS, model-download, and stack-compatibility tests run only through explicit markers.



---------------





The numbered order above is correct on *dependencies* but wrong on *risk*. It follows the spec-kit convention — setup → foundation → stories by priority → polish — which puts every hardware measurement (T091, T092, T093) in Phase 8. That schedules the one question that can invalidate the whole project **last**.

Here's the order I'd actually work in.

## Stage 0 — Prove it runs at all (before any product code)

```
T001, T003, T004, T005  →  T007 → T008  →  [SPIKE]  →  T091, T092
```

T002 stays a stub here; T008 is what fills in the real pins.

The spike is **not in tasks.md and should not be product code** — a throwaway script that `hf download`s the Ref2VA checkpoint, loads it under sequential/layer-wise offload plus quantization, and generates one clip at the minimum duration. It answers three things:

1. Does a 33B omni-model produce a frame at all on 16 GB VRAM?
2. Peak reserved bytes vs. the 13.5 GiB gate; peak host RSS vs. 64 GB.
3. Wall-clock for the shortest possible run — the number that tells you whether "unbounded" means 20 minutes or 9 hours.

You don't need US2 for this. Nothing in US1 touches `model_catalog.py` (I checked — the catalog appears only in T054, T057–T064), so hand-placing the snapshot in `.model-cache/` is entirely sufficient for months.

**This also fixes a real circular dependency in the current file.** T040 must declare "every measured field," but T091 — which produces the measurements — runs 51 tasks later and edits the same file. As written, T040 necessarily writes placeholder numbers that T091 comes back and corrects. Measuring first collapses that into one honest write.

## Stage 1 — Foundation

```
T002 (finalize)  →  T009–T014  →  T015–T021  →  T039  →  T022
```

One change: **hoist T039 (stub adapter) out of US1 to sit right after T021.** It's the reference implementation of the protocol you just defined, and it's what makes every downstream test runnable. Leaving it at T039 means writing twelve test files against a protocol nothing implements yet.

## Stage 2 — MVP on stubs

```
T023–T034  →  T035–T038  →  T041–T044  →  T068, T069  →  T045, T046
```

**Hoist T068 + T069 (progress emission) out of US3 into US1**, before the UI lands. They're rated P3 as a *user-facing* feature, but for a run measured in hours they're your only way to distinguish a slow generation from a hung one — you'll need them while debugging Stage 3, not after. Leave T070 (the rendered panel) at P3 where it belongs.

## Stage 3 — MVP on real H3

```
T040  →  T093
```

T040 now declares numbers you measured in Stage 0 instead of numbers you hoped for.

## Stage 4 — Robustness before catalog

```
T071–T080   (US4)
T047–T064   (US2)
```

**This inverts the spec's P2/P4 priority, deliberately.** After Stage 3 you have a real multi-hour run on a card with 16 GB. OOM is the most probable failure mode in this entire project, and T076 (OOM translation and recovery guidance) plus T077 (cancellation) are what make it survivable. The model catalog is 18 tasks of convenience over a command you can run by hand.

Keep the spec order instead if the Model Library is a demo requirement rather than a workflow one — that's the only reason to.

## Stage 5 — Finish

```
T065–T067, T070  (US3 UI)  →  T081–T086 (History)  →  T087–T090, T094
```

## On the `[P]` markers

51 tasks are marked parallel, but that's addressed to multiple developers or agents. Solo, treat `[P]` as "these can be written in one sitting without switching context" — batch T009–T014 and T023–T032 as single writing sessions rather than trying to interleave them.

---

The single highest-value change is Stage 0. Everything else is refinement; that one moves project-viability discovery from ~task 90 to ~task 10.

Want me to restructure tasks.md to this sequence — either renumbering, or adding a "Recommended execution order" section that leaves the IDs stable?