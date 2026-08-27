# Quickstart: Generate Image-Conditioned Lip-Synced Video

This is the planned setup and verification contract. Commands become executable after implementation
tasks create the application and requirement files.

## macOS development and offline UI testing

```bash
python3 --version
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Use the deterministic providers first. This exercises model inventory UI, image/audio upload,
consent and provider validation, progress, sequential video/voice/lip stages, MP4 mux/verification,
preview, download, metadata, and cleanup without CUDA, network access, or model weights.

```bash
I2V_RUNTIME_PROFILE=stub python app.py
```

Open `http://127.0.0.1:7860`. Run the offline suite:

```bash
python -m pytest
```

An Apple Silicon real-backend test is opt-in:

```bash
python -m pytest -m mps
```

Only adapters explicitly declaring MPS support may run there. Production-quality CogVideoX/Qwen3-TTS/
LatentSync generation is not expected on the Mac control-path environment; an unsupported profile must
fail before allocation rather than call CUDA or silently fall back after a failed model load.

## Windows RTX 5080 production setup

In PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

CUDA 12.8 is the validated baseline for the RTX 5080. If the official PyTorch selector provides a
newer stable CUDA 12+ wheel tested by this project, use that command consistently. CUDA 12.4 is retained
only as a migration example for older compatible NVIDIA environments and is not the RTX 5080 acceptance
baseline:

```powershell
# Legacy compatibility example; do not use for RTX 5080 acceptance.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify the installed wheel and GPU before downloading weights:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

The command must report CUDA available and identify the RTX 5080. If a selected model is gated/private,
authenticate through the Hugging Face CLI or approved environment credential source. Never paste a
token into a model URL or the Gradio form.

```powershell
$env:I2V_RUNTIME_PROFILE = "production"
python app.py
```

## Download and select the reference model set

In the Model Library section, submit each canonical repository URL under the matching role:

```text
Video:    https://huggingface.co/zai-org/CogVideoX-5b-I2V
Voice:    https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
Lip sync: https://huggingface.co/ByteDance/LatentSync-1.5
```

Before download, the app shows the resolved immutable commit, detected adapter/roles, access/license,
expected size, device and memory compatibility, and provider constraints. Accept each repository's
terms separately. Compatibility validation does not grant commercial rights.

After download:

1. Confirm all three entries show `ready` and their immutable commits.
2. Select CogVideoX as video, Qwen3-TTS as dedicated voice, and LatentSync as dedicated lip sync.
3. If a future validated video adapter provides native voice or lip sync and a dedicated provider is
   also selected, explicitly choose Native or Dedicated for each overlapping role.
4. Review the effective voice provider's languages and reference-audio rules before upload.

Downloaded ready models remain selectable offline. To remove one, deselect it, ensure no generation
uses it, choose Delete, review expected reclaimed space, and confirm. The app protects active/in-use
models and reports actual reclaimed bytes after deletion.

## Generate a video

Provide:

- one still image containing exactly one clearly visible face and usable mouth region;
- a motion prompt;
- a separate speech script;
- one reference-voice recording satisfying the selected voice provider's displayed rules;
- an explicit supported language;
- per-request confirmation that you own the voice or have explicit permission to clone it.

The default CogVideoX controls are 49 frames, 8 FPS, guidance 6.0, and a generated seed when blank.
The app shows the effective seed and parameters. It runs heavy providers sequentially and displays
loading, preprocessing, video, voice, lip-sync, mux, verification, export, and memory status.

A technically successful MP4 appears in both the embedded player and Download button. v1 does not
use an automated lip-sync quality score; review synchronization visually before using the result.

## Acceptance commands

```text
python -m pytest
python -m pytest -m model_download
python -m pytest -m cuda
```

The offline suite must not download weights. The model-download suite uses explicitly configured test
repositories. The CUDA acceptance run verifies:

- resolved commits and adapter capability coverage;
- consent/language/reference-audio validation before inference;
- sequential CogVideoX, Qwen3-TTS, and LatentSync 1.5 stages;
- peak allocated CUDA memory below 15.5 GiB for every default heavy stage;
- one video stream and one non-silent speech stream within one-frame duration tolerance;
- preview/download publication, sanitized metadata, and temporary-file cleanup.

Record actual target-machine generation time as the baseline; do not fail release against an invented
latency target.

## OOM recovery order

1. Confirm no other process is consuming material GPU memory.
2. Retry after the application unloads the cached provider and releases recoverable CUDA cache.
3. Lower the frame count to the next adapter-valid value or choose a supported lower-resolution preset.
4. Enable the stronger reviewed offload or quantized profile.
5. Select a model set whose validated memory profiles fit 16 GB.
6. Never silently change the model, provider choice, seed, or effective parameters.

LatentSync 1.6 is not enabled for this target because its official minimum inference VRAM is 18 GB.

## Runtime files and privacy

- The application uses an exclusive model cache beneath the configured `I2V_DATA_ROOT`; it does not
  delete revisions from the user's global Hugging Face cache.
- Inventory JSON and lock files live beneath the data root; completed media and metadata live under
  `outputs/<request-id>/`.
- Uploaded image/audio, derived voice data, silent video, WAV, and partial mux files are temporary and
  removed according to the request cleanup policy.
- Completed outputs remain until explicit deletion or configured expiry. Downloaded models are never
  automatically evicted in v1.
- Tokens, `.env`, caches, uploaded media, generated media, and derived voice representations remain
  untracked.
- The server binds to `127.0.0.1`; Gradio public sharing is disabled.
