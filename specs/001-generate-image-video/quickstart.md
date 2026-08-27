# Quickstart: Generate Image-Conditioned Lip-Synced Video

This is the planned setup and verification contract. Commands become executable after implementation
creates the application and bounded requirement files.

## macOS development and offline workflow

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Run the deterministic providers first. They exercise model/history UI, input staging, fresh consent,
speech-first planning, duration decisions, worker messaging, full bundle publication, MP4 verification,
preview/download, and failure cleanup without CUDA, network access, or model weights.

```bash
I2V_RUNTIME_PROFILE=stub python app.py
python -m pytest -m "not mps and not cuda and not model_download"
```

Open `http://127.0.0.1:7860`. Public sharing remains disabled. An Apple Silicon backend smoke is opt-in:

```bash
python -m pytest -m mps
```

Only adapters declaring MPS support may run. Production CogVideoX/Qwen/LatentSync is not expected on
macOS; unsupported profiles fail before allocation and never import/execute CUDA-only worker code.

## Windows 11 RTX 5080 production setup

Use NVIDIA driver 580.88 or newer. Create the main environment in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The main lock includes `qwen-tts==0.1.1`, `transformers==4.57.3`, and `accelerate==1.12.0`.
Create the separately locked LatentSync worker because its provider stack conflicts with the main one:

```powershell
deactivate
py -3.11 -m venv .venv-latentsync
.\.venv-latentsync\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-latentsync-windows.txt
deactivate
.\.venv\Scripts\Activate.ps1
$env:I2V_LATENTSYNC_PYTHON = (Resolve-Path '.\.venv-latentsync\Scripts\python.exe').Path
```

Verify Blackwell support before model-file transfer:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NO CAPABILITY'); print(torch.cuda.get_arch_list() if torch.cuda.is_available() else [])"
```

Acceptance requires PyTorch 2.13.x, CUDA 13.0, RTX 5080, capability `(12, 0)`, compiled `sm_120`
support where exposed, and successful BF16/FP16 allocations. Then launch:

```powershell
$env:I2V_RUNTIME_PROFILE = 'production'
python app.py
```

If a repository is gated/private, authenticate outside the UI using `hf auth login` or `HF_TOKEN`.
Never put a token in a URL or form. The application pins all Hub traffic to `https://huggingface.co`,
does not inspect/display/record/acknowledge/enforce model-license terms, and treats access denial only
as an authentication/access error.

## Download and select the reference model set

Submit canonical repository URLs in Model Library:

```text
Video:    https://huggingface.co/zai-org/CogVideoX-5b-I2V
Voice:    https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
Lip sync: https://huggingface.co/ByteDance/LatentSync-1.5
```

The app shows normalized repository ID, tracking ref when provided, resolved immutable commit,
adapter/roles, access state, dependency closure, expected transfer bytes, disk preflight, device/memory
compatibility, and provider constraints. It never shows or stores license fields.

Before model-file content transfer, the complete missing dependency closure plus staging overhead must
leave the configured reserve (10 GiB by default). Ready entries include verified required-file digests
and remain selectable offline; inference loads only local commit-pinned snapshots.

After download:

1. Confirm all entries and auxiliary dependencies show `ready` with immutable commits.
2. Select CogVideoX as video, Qwen3-TTS as Dedicated voice, and LatentSync as Dedicated lip sync.
3. If native and dedicated providers overlap, explicitly choose Native or Dedicated for each role.
4. Review Qwen's ten-language allowlist and reference-audio rules. `Auto` language is disabled.
5. Note that transcript-free x-vector cloning is used by the reference profile and may be less faithful
   than Qwen's ICL mode with a matching reference transcript.

`Check for updates` is manual and uses an entry's stored tracking ref. A different commit downloads as
a separate entry; the old revision is never replaced, selected, or deleted automatically. A commit-only
entry has no update check until a tracking ref is established.

Model deletion requires confirmation, rejects active/leased/dependency-referenced revisions, preserves
shared blobs, and reports only measured reclaimed bytes after re-scan. App-owned incomplete downloads
have a separate confirmed discard action; no broad cache prune runs automatically.

## Generate a video

Provide one still image containing exactly one clear face/mouth, a motion prompt, a speech script, one
provider-compatible reference recording, an explicit supported language, and current confirmation that
you own the voice or have permission to clone it. Consent defaults/resets false, resets when the audio
changes, and is bound server-side to the new request ID and audio SHA-256.

The reference CogVideoX adapter is fixed at 720x480, 49 generated frames, 8 FPS, guidance 6, and an
English motion prompt. Frame/FPS controls are preferences for adapters that declare alternatives; they
do not make this fixed profile variable-duration. Speech is synthesized first. If its full duration
cannot fit the single CogVideoX candidate within the final one-frame tolerance, generation stops before
the video model and asks for a shorter/longer script near the supported duration or another adapter.
Speech is never trimmed or time-stretched.

The successful order is:

```text
validate and stage inputs
-> synthesize and verify complete speech
-> derive duration plan and recheck disk
-> generate 49-frame/8-FPS video
-> resample video to 25 FPS without changing duration
-> run isolated LatentSync 1.5 worker
-> mux and verify audio/video
-> write manifest and atomically publish the full bundle
```

The result reports requested/effective values, 8-FPS source and 25-FPS final timebases, seed, immutable
models/providers, stage memory, and retained bytes. A technically valid MP4 appears in the player and
download control without an automated lip-sync quality gate; review synchronization visually.

## Successful retention and Request History

Before the first submission, the UI states that reference audio and derived voice data are ordinary
unencrypted files. Every successful request is published only under fixed project
`outputs/<request-id>/` and retains:

- copied original image and reference audio;
- derived voice representation and synthesized speech;
- pre-lip, post-lip, and final MP4 media;
- request/effective metadata and the validated manifest.

There is no output-root setting or environment override, no in-app bundle deletion, and no automatic
expiry/cleanup of successful artifacts. Remove a bundle only through the filesystem. Request History is
read-only: refresh scans `outputs/`, shows preview/inventory/bytes/origin/dependents/warnings, and never
repairs or mutates a bundle.

To reuse retained audio, choose the retained file through the ordinary filesystem picker and confirm
voice consent again. The server matches a validated request-ID filename/digest or a unique retained
digest, copies the bytes into the new staging bundle, and records an advisory origin. Ambiguous digest
matches create no origin. If the origin later disappears/corrupts, refresh warns dependents and disables
origin-based reuse without changing later bundles.

Failed/cancelled requests remove only their unpublished `outputs/.work/<request-id>/`. Startup may
reconcile orphan staging after lock/owner checks; it never cleans a published bundle.

## Acceptance commands

```text
python -m pytest -m "not mps and not cuda and not model_download"
python -m pytest -m model_download
python -m pytest -m cuda
```

The offline suite downloads no weights and uses a fake worker. Model-download tests verify fixed Hub
endpoint, no remote code/license handling, immutable dependencies, disk reserve, offline restart,
manual update coexistence, deletion races, and partial-download handling. Windows CUDA acceptance
verifies the full Qwen -> duration -> CogVideoX -> 25-FPS bridge -> isolated LatentSync path, zero hidden
downloads during inference, each heavy peak below 15.5 GiB, complete retained artifacts, and final
audio/video duration within one 25-FPS frame.

Record measured target-machine latency; do not invent an SLA.

## Recovery order

1. Confirm driver, PyTorch/CUDA/capability, and worker handshake versions.
2. Confirm no other process materially consumes GPU memory.
3. Retry after the application releases the failed provider/worker.
4. Enable a stronger reviewed offload/quantized profile that passed the same adapter tests.
5. Select another adapter whose documented temporal/memory profile supports the complete speech.
6. Free model/bundle space manually when the reserve preflight fails.
7. Never silently change model, provider, seed, speech, or effective media values.

LatentSync 1.6 is not enabled because its documented minimum inference VRAM is 18 GB. Lowering the
fixed CogVideoX frame count is not a valid recovery action for this profile.

## Runtime files and privacy

- Application-owned model snapshots/inventory live beneath fixed project `.model-cache/`, separate
  from the user's global Hugging Face cache.
- Successful bundles live only under fixed project `outputs/`; unpublished work uses `outputs/.work/`.
- The default reserve is exactly 10 GiB (`10 * 1024^3` bytes) and nothing is auto-evicted.
- Reference audio, derived voice data, prompts, speech, intermediates, and final media are local plaintext
  files. Host permissions, backup/sync tools, removable storage, and physical access remain operator risks.
- Tokens, virtual environments, model caches, uploads, output bundles, and derived voice data remain
  untracked. Normal logs omit prompts, tokens, absolute upload paths, and voice data.
- The server binds only to `127.0.0.1`; Gradio sharing is disabled.
