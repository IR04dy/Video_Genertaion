<!--
Sync Impact Report
- Version change: template -> 1.0.0
- Added principles:
  - I. Cross-Platform and Device-Safe by Default
  - II. Inference/UI Separation
  - III. Bounded, Reproducible Inference
  - IV. Test-First Reliability
  - V. Observable and Local-by-Default Operation
- Added sections:
  - Technical and Deployment Constraints
  - Development Workflow and Quality Gates
- Removed sections: none (template placeholders replaced)
- Templates reviewed:
  - ✅ .specify/templates/plan-template.md (constitution gates made explicit)
  - ✅ .specify/templates/spec-template.md (compatible; no structural change required)
  - ✅ .specify/templates/tasks-template.md (platform, failure, and test tasks required)
  - ✅ .specify/templates/commands/*.md (directory absent; no command templates to update)
- Runtime guidance reviewed:
  - ✅ AGENTS.md (managed by the planning workflow)
- Deferred items: none
-->
# Image-text-to-video Constitution

## Core Principles

### I. Cross-Platform and Device-Safe by Default
All filesystem access MUST use `pathlib.Path` and all runtime device selection MUST be explicit,
capability-checked, and ordered CUDA, Apple MPS, then CPU. Importing or launching the application
on macOS or a CPU-only host MUST NOT execute CUDA-only code. Platform-specific installation steps
MUST be isolated and documented. Rationale: development and production use different operating
systems and accelerators, so platform assumptions are correctness defects.

### II. Inference/UI Separation
Model loading, input validation, generation, memory policy, and video export MUST live behind a
typed inference-engine interface independent of Gradio. The web layer MUST translate user input,
progress, results, and domain errors without owning tensor or pipeline logic. Each module MUST have
a single clear responsibility and be independently testable. Rationale: a thin UI keeps the costly
ML core reusable, testable, and replaceable.

### III. Bounded, Reproducible Inference
Every generation request MUST validate model-compatible frame counts, dimensions, FPS, guidance,
and seed before allocating model memory. CUDA inference MUST use a supported reduced precision and
documented offload/VAE controls suitable for a 16 GB budget; heavyweight models MUST require an
explicit quantization/offload profile. The effective seed, model ID, device, dtype, and normalized
parameters MUST be recorded with each result. Rationale: predictable resource use and replayable
inputs are required for production diagnosis.

### IV. Test-First Reliability (NON-NEGOTIABLE)
Tests MUST be written before implementation for device selection, parameter validation, pipeline
adaptation, export, OOM translation, and the UI-to-engine contract. Unit tests MUST use fakes and
MUST NOT download model weights. At least one opt-in smoke test MUST cover real model loading on a
supported environment. OOM, missing codec, invalid image, unavailable model, and unsupported
device failures MUST produce actionable user messages and release recoverable resources.
Rationale: ML infrastructure failures are expensive and platform-dependent, so deterministic tests
must guard the control paths.

### V. Observable and Local-by-Default Operation
Generation MUST expose phase-level progress and structured logs without leaking prompts, tokens, or
local paths at normal log levels. CUDA memory metrics MUST be reported when CUDA is active and a
clear "unavailable" status MUST be shown elsewhere. Uploaded images and generated videos MUST stay
local by default, use collision-safe output names, and have an explicit retention/cleanup policy.
Secrets MUST come from environment or approved credential stores and MUST never be committed.
Rationale: useful diagnostics and safe local handling are both required for a trustworthy tool.

## Technical and Deployment Constraints

- Python 3.11 is the baseline runtime; dependencies MUST declare compatible bounded versions.
- PyTorch 2.x and Hugging Face Diffusers are the supported inference foundation; model-specific
  adapters MAY use concrete pipeline classes when a generic auto-pipeline cannot honor the model
  contract.
- The production Windows build MUST use an official CUDA 12+ PyTorch wheel that supports the RTX
  5080 architecture; installation docs MUST verify `torch.cuda.is_available()` and device identity.
- macOS is a functional control-path environment. MPS MAY run only a compatible small/smoke model;
  CPU/MPS users MUST be warned that production-quality generation may be impractically slow.
- Output MUST be a browser-playable MP4 with dimensions accepted by the selected encoder.
- Quantized/GGUF loading MUST be opt-in and adapter-based; absence of optional quantization
  dependencies MUST not break the default pipeline.

## Development Workflow and Quality Gates

1. Each feature MUST have a validated specification, research record, implementation plan, and
   dependency-ordered tasks before production code is written.
2. The plan MUST name the supported model/pipeline pair, parameter constraints, device matrix,
   memory profile, and fallback behavior using primary documentation.
3. Tests MUST fail for the intended reason before implementation, then pass on macOS/CPU without
   model downloads. CUDA smoke and memory tests MUST be separately marked and runnable on Windows.
4. Changes to generation parameters or adapters MUST update validation, metadata, contracts, tests,
   and user documentation together.
5. Reviews MUST verify no platform-specific path literals, hidden network exposure, unbounded GPU
   allocations, secrets, or generated media are committed.

## Governance

This constitution supersedes conflicting project practices. Amendments MUST include a rationale,
an impact assessment for specifications/templates/runtime behavior, and a semantic version change:
MAJOR for incompatible governance changes, MINOR for new or materially expanded rules, and PATCH
for clarifications. Every plan and pull request MUST record compliance or explicitly justify a
temporary exception with an owner and removal condition. Compliance is reviewed at planning,
post-design, and before release.

**Version**: 1.0.0 | **Ratified**: 2026-08-27 | **Last Amended**: 2026-08-27
