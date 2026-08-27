# Research: Generate Image-Conditioned Lip-Synced Video

All technical unknowns from the clarified specification are resolved below. Links point to primary
project or vendor documentation reviewed on 2026-08-27.

## Capability registry for user-provided model links

**Decision**: Accept canonical Hugging Face repository links, resolve them to immutable commit SHAs,
and mark a model ready only when repository metadata matches a reviewed in-process adapter fingerprint.
The adapter declares roles (`video`, `voice`, `lip_sync`), native capabilities, inputs/outputs,
language/audio constraints, devices, precision, memory profile, and allowed weight formats.

**Rationale**: User-selected repositories cannot safely or reliably map to one generic call signature.
Diffusers auto-pipelines select classes from repository configuration, but image-to-video architectures
still expose model-specific parameters. The registry satisfies model choice without executing arbitrary
Hub code. Source: [Diffusers pipeline overview](https://huggingface.co/docs/diffusers/en/api/pipelines/overview).

**Alternatives considered**: `trust_remote_code=True` was rejected as incompatible with the security
requirement. Accepting any model tagged image-to-video was rejected because tags do not prove interface,
memory, voice, or lip-sync compatibility.

## Reference image-to-video adapter

**Decision**: Use `zai-org/CogVideoX-5b-I2V` with
`diffusers.CogVideoXImageToVideoPipeline` as the reviewed default video adapter.

**Rationale**: Official Diffusers examples cover image plus text prompt, BF16, seeded generation,
guidance, 49 frames, 720x480 defaults, VAE slicing/tiling, and 8-FPS export. This aligns with the v1
controls and fits the 16 GB design when offloaded. Sources:
[CogVideoX pipeline API](https://huggingface.co/docs/diffusers/main/api/pipelines/cogvideox) and
[CogVideoX-5B-I2V model card](https://huggingface.co/zai-org/CogVideoX-5b-I2V).

**Alternatives considered**: Stable Video Diffusion lacks the required motion text prompt. Newer or
larger I2V models may be added as adapters, but are not the default until they pass the same memory and
interface tests. Generic auto-loading remains an implementation helper only after registry matching.

## Reference voice-cloning adapter

**Decision**: Use `Qwen/Qwen3-TTS-12Hz-1.7B-Base` with the official `qwen-tts` package. Use its
x-vector-only voice-cloning path as the default so the specified reference audio does not require a
transcript; permit a reference transcript only when a validated profile requests it.

**Rationale**: The official model supports voice cloning from reference audio, exposes a supported
language list, covers ten major languages, accepts local paths or waveform/sample-rate tuples, uses
safetensors, and is Apache-2.0 licensed. The 1.7B repository is about 4.5 GB and can run sequentially
after video-model release. Sources:
[Qwen3-TTS Base model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) and
[Qwen3-TTS official repository](https://github.com/QwenLM/Qwen3-TTS).

**Alternatives considered**: XTTS-v2 has a more restrictive model license and a different package
stack. Qwen3-TTS 0.6B Base is a lower-memory fallback adapter candidate, but the 1.7B model is the
quality-oriented reference profile. CustomVoice and VoiceDesign checkpoints were rejected because
they do not satisfy user-provided reference-voice cloning in the same way.

## Reference lip-sync adapter

**Decision**: Use a reviewed `ByteDance/LatentSync-1.5` adapter for the target card. Do not enable
LatentSync 1.6 in the 16 GB production profile.

**Rationale**: The official LatentSync project documents 8 GB minimum inference VRAM for 1.5 and
18 GB for 1.6. Version 1.5 therefore leaves room for the application and encoding on a 16 GB card,
provided other heavy models are unloaded first. Sources:
[LatentSync official repository](https://github.com/bytedance/LatentSync) and
[LatentSync 1.5 model repository](https://huggingface.co/ByteDance/LatentSync-1.5).

**Alternatives considered**: LatentSync 1.6 exceeds the target VRAM. Wav2Lip's common pretrained
weights carry use restrictions unsuitable as the default. Native audio-driven video models remain
supported through the composite adapter protocol when their manifests pass validation.

## Native and dedicated provider resolution

**Decision**: Compile the selected model set into an artifact dependency graph. A model may provide
one or more roles. If native and dedicated providers overlap, require the user's explicit selection
for that role. Schedule voice first when a native video/lip provider consumes synthesized audio;
otherwise use video, voice, then lip sync. A native all-in-one model may execute a single stage.

**Rationale**: A fixed video-first sequence cannot support audio-conditioned native video models,
while loading all providers together violates the memory budget. An explicit graph preserves user
choice, validates complete coverage, and still guarantees sequential heavy-model residency.

**Alternatives considered**: Native-first and dedicated-first implicit priorities were rejected by
clarification. Automatic benchmark-based selection was rejected because it changes behavior and
reproducibility without user intent.

## CUDA memory and quantization policy

**Decision**: Prefer BF16 on CUDA, allow FP16 only per validated adapter, enable CogVideoX model CPU
offload and VAE slicing/tiling, maintain one loaded heavy provider, and release/offload between stages.
Treat peak allocated memory below 15.5 GiB for each default stage as a release gate. Offer stronger
offload and reviewed component quantization as explicit profiles.

**Rationale**: Diffusers documents VAE slicing/tiling and model/sequential offload as memory controls;
its CogVideoX guidance reports a material reduction with offload and tiling. Source:
[Diffusers memory optimization](https://huggingface.co/docs/diffusers/optimization/memory).

**Alternatives considered**: Whole-pipeline CUDA residency risks OOM. Sequential CPU offload is a
slower recovery profile, not a hidden default. Compilation and FlashAttention remain opt-in until the
base RTX 5080 path is measured and compatibility-tested.

## GGUF and heavyweight models

**Decision**: Represent GGUF/quantized support as adapter-specific component loading. A link is not
ready merely because it contains a GGUF file; the registry must provide architecture/config mapping,
supported roles, and a tested memory profile.

**Rationale**: Quantized component formats do not define an entire generation pipeline. Treating
GGUF as a generic pipeline would obscure tokenizer, scheduler, VAE, and custom component requirements.
Source: [Diffusers model formats](https://huggingface.co/docs/diffusers/using-diffusers/other-formats).

**Alternatives considered**: Arbitrary GGUF auto-loading was rejected. Maintaining no hook was also
rejected because the specification explicitly anticipates 14B+ models.

## Hugging Face download, inventory, and deletion

**Decision**: Use `HfApi` for metadata/commit resolution, `snapshot_download()` for immutable
snapshots, and a dedicated application cache. Persist an atomic inventory that references commit-
specific snapshot paths. Use `scan_cache_dir().delete_revisions()` to preview and execute deletion,
protect active/in-use leases, and report measured bytes before and after.

**Rationale**: Hugging Face snapshots resolve files through commit-based cache paths and its cache
API accounts for blobs shared across revisions when constructing deletion strategies. Sources:
[Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download) and
[Hub cache management](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

**Alternatives considered**: Deleting repository directories directly risks shared blobs and corrupt
inventory. The global user cache was rejected because the app could delete files owned by other tools.
SQLite is unnecessary for one local process; atomic JSON plus a cross-platform file lock is sufficient.

## Repository and credential security

**Decision**: Permit only HTTPS `huggingface.co` repository roots and approved immutable revision
forms. Reject embedded credentials, query tokens, blob/file URLs, alternate hosts, unreviewed code,
and unexpected executable/pickle artifacts. Credentials come from Hugging Face login or an environment
source and are never accepted in the UI or stored in inventory/logs.

**Rationale**: Model loading can cross a code-execution boundary. Even trusted attention kernels are
opt-in in Diffusers, so the application must keep remote execution disabled by default. Source:
[Diffusers attention backend security note](https://huggingface.co/docs/diffusers/main/optimization/attention_backends).

**Alternatives considered**: Tokens in URLs leak through history and logs. Allowing arbitrary hosts
would require a broader downloader threat model and is outside v1.

## Device resolution and Windows CUDA baseline

**Decision**: Resolve CUDA, then MPS, then CPU, intersected with selected adapter capabilities. Use
an official stable PyTorch CUDA 12.8-or-newer Windows wheel for RTX 5080 production and verify
`torch.cuda.is_available()`, device name, CUDA runtime, and a small allocation before model download.

**Rationale**: PyTorch publishes platform-specific wheels and its selector currently includes CUDA
12.8. CUDA 12.4 predates the desired Blackwell-safe baseline. Source:
[PyTorch local install selector](https://pytorch.org/get-started/locally/).

**Alternatives considered**: A single requirements file cannot choose both macOS and Windows CUDA
wheels safely. Nightly wheels are a documented fallback only when no tested stable wheel supports the
target. Automatic MPS-to-CPU fallback after allocation was rejected because it can multiply latency.

## Reference-audio, language, face, and consent validation

**Decision**: Store audio formats, duration, sample rate/channels, speaker conditions, optional
transcript rule, quality bounds, and languages in each voice profile. Display those constraints before
submission and validate locally. Require an explicit consent checkbox for every request and exactly
one face/mouth target before video generation, using the effective lip provider's preflight analyzer.

**Rationale**: The user chose provider-specific audio limits and explicit language. Performing consent,
face, and audio validation before loading models prevents expensive invalid work and records the safety
decision at the request boundary.

**Alternatives considered**: One universal audio limit was rejected by clarification. Automatic
language detection and multi-face selection are explicitly out of scope.

## Lip-sync publication policy

**Decision**: Treat adapter exceptions, absent face output, invalid media, silent speech, or failed
muxing as technical failures. Do not compute or enforce a lip-sync quality score in v1; every
technically valid MP4 is previewable/downloadable for visual user review.

**Rationale**: This directly implements the clarification and avoids presenting model-specific sync
metrics as a universal quality truth.

**Alternatives considered**: Global and per-model automated thresholds were rejected by the user.
Automatic retries were rejected because they add unpredictable time and GPU cost.

## MP4 export and verification

**Decision**: Write a request-scoped silent/intermediate MP4, mux synthesized speech as AAC using an
argument-list FFmpeg invocation, verify exactly one video stream and one non-silent audio stream with
duration tolerance of one frame, and atomically rename the final MP4. Use Diffusers/imageio utilities
for frame encoding where compatible.

**Rationale**: Diffusers `export_to_video` accepts PIL/NumPy frames, FPS, quality/bitrate, and
macroblock constraints. Explicit mux and verification are required because the pipeline produces
speech separately. Source: [Diffusers export utilities](https://huggingface.co/docs/diffusers/main/api/utilities).

**Alternatives considered**: Returning the lip-sync adapter's raw file without stream validation can
publish silent or mismatched media. Shell-form FFmpeg commands were rejected for path and injection
safety.

## Gradio UI, progress, and concurrency

**Decision**: Use Gradio Blocks with image/audio uploads, motion and speech text areas, consent and
language controls, advanced parameters, model URL/download controls, inventory table, provider
selectors, progress/status/memory panels, Video, and DownloadButton. Queue generation with concurrency
one; model downloads also use bounded catalog operations and cannot delete leased models.

**Rationale**: DownloadButton accepts `Path` output and Gradio supports progress/queue integration,
while a single active heavy request matches the GPU memory constraint. Source:
[Gradio DownloadButton](https://www.gradio.app/main/docs/gradio/downloadbutton).

**Alternatives considered**: Parallel generations were rejected due to OOM risk. A separate web API,
database, and worker queue are unnecessary for a local single-user v1.

## Test and packaging strategy

**Decision**: Keep ordinary tests fully offline with fake Hub and model adapters. Mark real model
download, MPS, and CUDA tests separately. Install the platform PyTorch wheel before bounded application
requirements; document current official CUDA selection rather than allowing pip to replace it. Keep
optional quantization acceleration out of the mandatory control path.

**Rationale**: This meets the constitution's test-first and cross-platform requirements while avoiding
multi-gigabyte downloads in CI and on macOS. The Qwen package currently pins Transformers and
Accelerate, so the final requirements set must be resolved and tested as one compatible environment.
Source: [Qwen-TTS package metadata](https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml).

**Alternatives considered**: Exact transitive hashes are deferred until both OS lock files can be
generated from tested environments. Conda and container-only deployment were rejected because the
requested path is standard virtual environments and pip.
