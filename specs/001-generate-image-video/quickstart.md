# Quickstart: Generate Image-Conditioned Lip-Synced Video

This is the planned setup and verification contract. Commands become executable after implementation
creates the application and bounded requirement files.

## macOS development and offline workflow

For the offline suite, the control plane is all you need — no torch, no model
stack, no accelerator:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest
```

`requirements-dev.txt` includes `requirements-core.txt`, so this installs on any
platform and runs every offline test. Add the model stack only when you need to
load real weights:

```bash
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
```

On an Intel Mac the second step installs but cannot load a model: PyTorch ships
no x86_64 macOS wheel above 2.2.2, and the pinned libraries need 2.5. The offline
suite is unaffected because it never imports Diffusers.

Run the deterministic stub profile first. It exercises model/history UI, input staging, fresh consent,
two-reference validation, dialogue-tag prompt assembly, duration decisions, full bundle publication, MP4
verification, preview/download, and failure cleanup without CUDA, network access, or model weights.

```bash
python app.py          # stub profile is the default until a model is downloaded
python -m pytest       # offline: pytest.ini deselects every opt-in marker
```

Do not pass `-m` to get the offline run. A command-line `-m` *replaces* the
expression in `pytest.ini` rather than narrowing it, so
`-m "not mps and not cuda and not model_download"` silently re-enables the
network-dependent stack gate. The bare `python -m pytest` is the offline suite.

Open `http://127.0.0.1:7860`. Public sharing remains disabled. An Apple Silicon backend smoke is opt-in:

```bash
python -m pytest -m mps
```

Only adapters declaring MPS support may run. The production Wan2.2-S2V profile is not expected on macOS;
unsupported profiles fail before allocation and never import or execute CUDA-only code.

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

**The torch line must come first.** Diffusers and Transformers both depend on
torch, so installing `requirements.txt` into an environment without it pulls a
generic CPU wheel — and the application then runs on the CPU with nothing
obviously wrong in the install log. `python -m pytest -m stack_compatibility`
catches it: below torch 2.5 the gate skips and says so.

There is a **single environment**. Collapsing to one joint audio/video provider removed the previous
second virtual environment and its separately locked worker stack, because there is no longer a second
heavyweight provider with conflicting dependency pins.

Wan2.2-S2V is loaded through **DiffSynth**, not Diffusers: neither `WanSpeechToVideoPipeline` nor
`WanS2VTransformer3DModel` exists in any Diffusers release, so that route would mean running a fork of
the core library. The lock must supply `diffsynth==2.1.5` and `transformers==5.16.1`. Verify before
anything else, because the profile stays `incompatible` rather than falling back to remote repository
code:

```powershell
python -m pytest -m stack_compatibility
```

That gate is authoritative. For a quick manual check:

```powershell
python -c "from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig; import diffsynth.models.wan_video_dit_s2v; import torch, inspect; print('stack available; torch.load weights_only default =', inspect.signature(torch.load).parameters['weights_only'].default)"
```

The gate needs **torch >= 2.6**: Transformers 5.x disables its model classes below 2.5, and 2.6 is
where `torch.load` defaults to `weights_only=True` — the gate on Wan's `.pth` pickles. PyTorch ships no
x86_64 macOS wheel above 2.2.2, so this check cannot run on an Intel Mac at all — the offline suite still
can, because it never imports Diffusers.

Verify Blackwell support and host memory before model-file transfer:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NO CAPABILITY'); print(torch.cuda.get_arch_list() if torch.cuda.is_available() else [])"
```

Acceptance also records installed and available host system memory; the target machine has 64 GB, and
layer-wise offload makes host RAM a gating resource alongside the 16 GB of VRAM.

Acceptance requires PyTorch 2.13.x, CUDA 13.0, RTX 5080, capability `(12, 0)`, compiled `sm_120`
support where exposed, and successful BF16/FP16 allocations. Then launch:

```powershell
$env:APP_RUNTIME_PROFILE = 'production'
python app.py
```

If a repository is gated/private, authenticate outside the UI using `hf auth login` or `HF_TOKEN`.
Never put a token in a URL or form. The application pins all Hub traffic to `https://huggingface.co`,
does not inspect/display/record/acknowledge/enforce model-license terms, and treats access denial only
as an authentication/access error.

## Download and select the reference model set

In the Model Library section, submit the default reviewed profile's repository URL:

```text
Video + native lip sync:  https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
Voice cloning:            (TTS repository — pending the packaging decision)
```

Two models cover the three roles, behind one adapter. Wan covers video **and** lip sync, because audio
conditions its denoiser directly rather than being applied as a later pass. There is no separate
lip-sync model to download. The working set is **42.60 GiB across 15 files**.

Before download the app shows the resolved immutable commit, detected adapter and roles, expected size,
and compatibility against **both** the accelerator-memory and host system-memory ceilings, along with the
measured profile fields it will enforce: supported duration range, frame rate, resolutions, audio output,
dialogue languages, accepted reference kinds and limits, and prompt token capacity. It shows no license
information; that responsibility is entirely yours, outside this application.

After download:

1. Confirm the entry shows `ready` with its immutable commit.
2. Select it as the video profile. Voice and lip-sync roles resolve to it natively, so no dedicated
   provider selection is required.
3. Review the displayed reference rules and dialogue-language list before uploading anything.

Ready models remain selectable offline. **While a generation is running the model library is read-only:**
downloads and update downloads are refused until the run finishes or you cancel it. Listing, inspection,
and update checks stay available. To remove a model, ensure no generation uses it, choose Delete, review
expected reclaimed space, and confirm.

## Generate a video

Provide:

- **one or more still images** of the same subject, each containing exactly one clear face and mouth
  region. More views generally anchor identity better. There is no app-imposed limit — only the profile's;
- **one reference recording** as a *voice timbre anchor*;
- a **motion prompt** describing the scene and movement;
- a **speech script** — what the subject actually says;
- an explicit **language** from the profile's supported list;
- current confirmation that you own the voice or have permission to clone it.

**The reference recording must say different words from the speech script.** It is never played back and
never mixed into the output — it conditions voice timbre only. Spoken content comes exclusively from the
script, which the app embeds in the prompt as `<d>[language]...</d>` dialogue tags. The UI states this
rule at the point of upload.

**Video files are not accepted as references.** A 15-second reference clip costs roughly 86,000 tokens on
its own, which breaks the memory ceiling before generation starts. Image and audio only.

Consent still applies in full. It defaults and resets to false, resets whenever the reference audio
changes, and is bound server-side to the request ID and the audio's SHA-256. This is still voice cloning;
only the model performing it has changed.

The app then runs **one** generation that produces video and stereo audio together. There is no separate
speech step, no lip-sync pass, and no timebase conversion — mouth movement and voice come out of the same
invocation.

```text
validate inputs, references, language, consent
-> assemble prompt with dialogue tags and measure tokens
-> suggest a duration from script length and speaking rate; operator may override
-> recheck disk
-> ONE joint audio/video generation
-> decode video and audio
-> export container and verify streams
-> write manifest and atomically publish the full bundle
```

Duration is an **input** to the generation, not something measured from the speech — audio and video come
out together, so there is no separate speech stage to measure first. The app suggests a duration from your
script length and the profile's per-language speaking rate, clamped to the supported range, and you can
override it. Nothing is rejected for script length. If the delivery sounds rushed, raise the duration and
regenerate.

A motion prompt longer than the profile's token capacity is truncated to fit, and the truncation is
reported as an explicit override. The speech script is never truncated.

**Expect a long run.** On a single 16 GB card the model runs under layer-wise CPU offload from a
quantized checkpoint, trading time for capacity. There is no time limit, no runtime estimate, and no
confirmation prompt — a run measured in hours is the expected operating point, not a fault. Progress
shows a completion fraction during inference and decoding so you can tell it is working, and cancelling
takes effect within seconds.

Local output resolution is a measured profile field; DiffSynth's low-VRAM path uses 448x832. Frame
count must be **4n+1** at 16 fps, so the duration grid is 0.25 s and the effective duration is rounded
up to the next legal count with the audio tail padded to match.

## Successful retention and Request History

Before the first submission, the UI states that reference audio and derived voice data are ordinary
unencrypted files. Every successful request is published only under fixed project
`outputs/<request-id>/` and retains:

- copied original images and reference audio;
- derived voice representation;
- the assembled prompt actually submitted;
- decoded video, decoded audio, and the final MP4;
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
python -m pytest -m stack_compatibility
python -m pytest -m model_download
python -m pytest -m cuda
```

The offline suite downloads no weights and uses a stub adapter. It also runs a second time against a
fixture profile whose duration, frame rate, resolution, language set, reference limits, and token
capacity all differ from the production profile's — that pass fails if any model-specific value has leaked into shared code. Model-download tests verify fixed Hub
endpoint, no remote code/license handling, immutable dependencies, disk reserve, offline restart,
manual update coexistence, deletion races, and partial-download handling. Windows CUDA acceptance
The blocking stack spike verifies that the model-stack classes load with `trust_remote_code=False` and measures
resident footprint per precision against both ceilings; failure leaves the profile `incompatible` before
architecture is frozen. Windows CUDA acceptance verifies one composite generation (speech then video)
end to end, measures the real supported duration ceiling on this card, zero hidden downloads
during inference, peak at or below 13.5 GiB allocator-reserved accelerator memory on the display-attached
target, host resident memory at or below the configured ceiling, complete retained artifacts, and final
audio and video duration agreeing within one frame at the profile's frame rate. A long-runtime run proves
that progress keeps reporting a completion fraction and that cancellation takes effect within the
documented interval; it asserts nothing about elapsed time.

Record measured target-machine wall-clock time as a baseline only. **Do not invent an SLA, and never fail
a run for taking too long** — unbounded inference time is the accepted trade for running a 20B-effective
model on a single 16 GB card.

## Recovery order

1. Confirm driver, PyTorch/CUDA/capability versions, and that the pinned DiffSynth and Transformers
   releases actually export the model-stack classes.
2. Confirm no other process materially consumes GPU or system memory.
3. Retry after the application releases the failed provider.
4. Select a profile variant with stronger offload or heavier quantization that passed the same adapter
   tests; both the accelerator and host ceilings must still hold.
5. Shorten the speech script when a request is refused for duration, or choose a shorter supported
   duration.
6. Free model/bundle space manually when a disk reserve check fails, at preflight or mid-write.
7. Never silently change model, profile, seed, speech, or effective media values.

A slow run is not a fault condition. Do not treat elapsed time as a failure signal; there is no timeout
to recover from.

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
