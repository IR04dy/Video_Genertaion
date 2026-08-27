# Research: Generate Image-Conditioned Lip-Synced Video

All technical unknowns from the clarified specification are resolved below. Links point to primary
project or vendor documentation reviewed on 2026-08-27.

## Capability registry for user-provided model links

**Decision**: Accept canonical Hugging Face repository links, resolve them to immutable commit SHAs,
and mark a model ready only when repository metadata matches a reviewed adapter/worker fingerprint.
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

**Decision**: Use `zai-org/CogVideoX-5b-I2V` with the Diffusers 0.40.x
`CogVideoXImageToVideoPipeline` only as a fixed reviewed video profile: 720x480, 49 generated frames,
8 FPS, guidance 6, BF16, and English motion prompts. Do not advertise arbitrary temporal combinations.

**Rationale**: The official card and tagged pipeline source cover image plus English text prompt, BF16,
seeded generation, 49 frames, 720x480, VAE slicing/tiling, and 8-FPS export. The card's lowest memory
figure assumes sequential CPU offload plus all listed VAE optimizations and was measured on A100/H100,
so RTX 5080 fit remains an empirical release gate below 15.5 GiB rather than an assumed fact. Sources:
[Diffusers 0.40 pipeline source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py) and
[CogVideoX-5B-I2V model card](https://huggingface.co/zai-org/CogVideoX-5b-I2V).

**Alternatives considered**: Re-exporting 49 frames at arbitrary FPS was rejected as proof of model-
supported duration because it merely retimes the trained motion. CogVideoX1.5's discrete 81/161-frame
profiles and other I2V models may be added as separate adapters after memory/interface tests. Generic
auto-loading remains an implementation helper only after registry matching.

## Reference voice-cloning adapter

**Decision**: Use `Qwen/Qwen3-TTS-12Hz-1.7B-Base` with `qwen-tts==0.1.1`,
`transformers==4.57.3`, and `accelerate==1.12.0`. Use its x-vector-only voice-cloning path because the
specified UI has no required reference transcript, but disclose in provider constraints and metadata
that the official ICL path with a matching transcript may clone more faithfully.

**Rationale**: The official model supports voice cloning from reference audio, exposes a supported
language list, covers ten major languages, accepts local paths or waveform/sample-rate tuples, uses
safetensors, and can create an x-vector-only prompt without a reference transcript. The official
`qwen-tts` 0.1.1 package pins Transformers 4.57.3 and Accelerate 1.12.0, so the application must test
and lock the complete Diffusers/Qwen environment rather than independently upgrading either library.
The 1.7B provider is loaded only for speech synthesis and unloaded before video inference. Sources:
[Qwen3-TTS Base model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) and
[Qwen3-TTS official repository](https://github.com/QwenLM/Qwen3-TTS), including its
[package metadata](https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml).

**Alternatives considered**: Qwen3-TTS 0.6B Base is a lower-memory fallback adapter candidate, but the
1.7B model is the quality-oriented reference profile. CustomVoice and VoiceDesign checkpoints were
rejected because they do not satisfy user-provided reference-voice cloning in the same way. Other
voice stacks remain eligible only through reviewed adapters with the same speech-artifact contract.

## Reference lip-sync adapter

**Decision**: Use a reviewed, commit-pinned `ByteDance/LatentSync-1.5` adapter in a separately locked
local worker. Pin the 1.5-compatible code/config and every auxiliary VAE/Whisper source; use the official
FP16, 256px, 25-FPS/16-kHz inference profile. Resample CogVideoX's complete 8-FPS video to 25 FPS while
preserving duration before the worker. Do not enable LatentSync 1.6 on the 16 GB target.

**Rationale**: The project documents 8 GB minimum inference VRAM for 1.5 and 18 GB for 1.6, but the
8 GB value is only a lower bound. Its official environment pins older, mutually incompatible Torch,
Diffusers, Transformers, and Accelerate versions and contains CUDA/Linux assumptions. Process isolation
keeps Qwen's exact pins coherent, while the worker's Blackwell port and peak memory remain Windows
release gates. LatentSync does not infer input FPS reliably, so the explicit 25-FPS bridge prevents it
from treating a 49-frame/8-FPS clip as a short 25-FPS clip. The adapter dependency manifest includes
the exact `stabilityai/sd-vae-ft-mse` commit and the selected local Whisper checkpoint/digest instead of
allowing the upstream repo-ID load to contact the network. Sources:
[LatentSync official repository](https://github.com/bytedance/LatentSync) and
[LatentSync 1.5 model repository](https://huggingface.co/ByteDance/LatentSync-1.5).

**Alternatives considered**: A single Python environment was rejected because provider pins conflict.
An unmodified upstream environment was rejected because its legacy CUDA wheel is not the production
Blackwell baseline. LatentSync 1.6 exceeds the target VRAM. Other lip-sync providers remain
eligible through reviewed adapters after their interfaces, face requirements, artifact contract, and
target memory profile are validated. Native audio-driven video models remain supported through the
composite adapter protocol when their manifests pass validation.

## Native and dedicated provider resolution

**Decision**: Compile the selected model set into a speech-first artifact dependency graph. A model
may provide one or more roles. If native and dedicated providers overlap, require the user's explicit
selection for that role. Every accepted plan produces and validates the complete speech artifact
first, derives a supported frame-count/FPS combination from its exact duration, and only then runs
video generation and lip synchronization. A native composite adapter may fuse the later stages, but
it must expose speech as a completed first phase before allocating the video stage.

**Rationale**: Speech duration is authoritative and may make the selected video adapter infeasible.
Resolving duration before video allocation prevents trimming/time-stretching and avoids expensive
generation that cannot contain the full speech. An explicit graph preserves user choice, validates
complete coverage, and guarantees sequential heavy-model residency.

**Alternatives considered**: Video-first ordering was rejected because duration cannot be planned from
the finished speech. Native-first and dedicated-first implicit priorities were rejected by
clarification. Automatic benchmark-based selection was rejected because it changes behavior and
reproducibility without user intent.

## Duration planning and cross-provider timebases

**Decision**: Separate a static `ExecutionBlueprint` from the post-speech `EffectiveExecutionPlan`.
For each reviewed video profile, enumerate only documented `(generated_frames, playback_fps)` candidates
and compute clip duration as `generated_frames / playback_fps`. A candidate is accepted only when the
full verified speech differs by no more than one effective final frame after any declared lip-provider
bridge. Tie-break deterministically by smallest duration delta, then closest requested frame preference,
then closest requested FPS, then the profile's stable candidate order.

The reference CogVideoX profile has the single candidate `(49, 8)`. Its video is resampled to LatentSync
1.5's 25-FPS processing/final timebase with duration preserved and no frame deletion; the effective
final frame count is derived and recorded. If the complete speech cannot meet the final one-frame
tolerance, stop before CogVideoX inference. Never change speech rate, trim it, or omit content.

**Rationale**: FPS is not an inference argument for the fixed CogVideoX model, while LatentSync assumes
its supplied video FPS. Modeling both timebases prevents an export preference from being mistaken for
model capability and makes the final mux tolerance testable.

**Alternatives considered**: Treating arbitrary export FPS as a CogVideoX duration range, generating
video before speech, or silently truncating/looping frames in LatentSync were rejected.

## CUDA memory and quantization policy

**Decision**: Prefer BF16 in the main CUDA runtime and use LatentSync's reviewed FP16 worker. Enable
CogVideoX sequential CPU offload plus VAE slicing/tiling as the conservative 16 GB reference path,
maintain one loaded heavy provider/process, and release between stages. Treat peak allocated memory
below 15.5 GiB for every stage as a measured release gate. A faster model-offload profile or reviewed
component quantization is enabled only after separate RTX 5080 acceptance.

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

**Decision**: Use a fixed-endpoint `HfApi` for metadata/commit resolution and
`snapshot_download()`/dry-run for immutable snapshots in a dedicated application cache. Persist
`tracking_ref` separately from `resolved_commit`, a required-file digest manifest, and the complete
commit-pinned auxiliary dependency closure. Use `scan_cache_dir().delete_revisions()` for confirmed
revision deletion, protect active/in-use/dependency leases, and rescan physical bytes afterward.
Interrupted `.incomplete` content uses a separate confirmed app-owned discard operation.

**Rationale**: Hugging Face snapshots resolve files through commit-based cache paths and its cache
API accounts for blobs shared across revisions when constructing deletion strategies. Sources:
[Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download) and
[Hub cache management](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

**Alternatives considered**: Deleting repository directories directly risks shared blobs and corrupt
inventory. Broad automatic cache pruning was rejected because Hub revision strategies do not own every
incomplete/corrupt file. The global user cache was rejected because the app could delete files owned by
other tools. SQLite is unnecessary; atomic JSON plus a cross-platform file lock is sufficient.

## Repository and credential security

**Decision**: Permit only HTTPS `huggingface.co` repository roots and unambiguous revision inputs. Pin
the client/download endpoint to `https://huggingface.co` and ignore/reject `HF_ENDPOINT` overrides.
Reject embedded credentials, query tokens, blob/file URLs, alternate hosts, community custom pipelines,
and remote attention kernels; set `trust_remote_code=False` on every loader and
`DIFFUSERS_DISABLE_REMOTE_CODE=true` before importing Diffusers. Credentials come only from `HF_TOKEN`
or the local `hf auth login` store and never enter UI/domain/inventory/log fields.

LatentSync 1.5's reviewed `.pt` weight is an explicit exception to the general safetensors policy only
when repository commit and SHA-256 match the adapter fingerprint and the safest supported tensor-only
loader is used. All other pickle-bearing/unreviewed executable artifacts fail closed.

**Rationale**: Model loading can cross a code-execution boundary. Even trusted attention kernels are
opt-in in Diffusers, so the application must keep remote execution disabled by default. Source:
[Diffusers attention backend security note](https://huggingface.co/docs/diffusers/main/optimization/attention_backends).

**Alternatives considered**: Tokens in URLs leak through history and logs. Allowing arbitrary hosts
would require a broader downloader threat model and is outside v1.

## Model-license non-handling boundary

**Decision**: Exclude model-license fields and workflow states from Hub inspection, adapter profiles,
inventory records, request metadata, UI rows, validation, and tests. Authentication and gated/private
repository failures are treated strictly as access failures. The application presents no license text,
checkbox, acknowledgement, compatibility verdict, or policy decision.

**Rationale**: This is an explicit product boundary in FR-040, separate from credential enforcement.
Fetching the configuration, sibling filenames, immutable commit, file sizes, and access status is
sufficient to match a reviewed adapter and plan a download without reading card license metadata.

**Alternatives considered**: Recording a license identifier, displaying Hub card terms, blocking an
"invalid" license, and requiring an acknowledgement were rejected because each would make the
application inspect, display, record, acknowledge, or enforce license information.

## Device resolution and Windows CUDA baseline

**Decision**: Resolve CUDA, then MPS, then CPU, intersected with selected adapter capabilities. Use
PyTorch 2.13.0's official CUDA 13.0 Windows wheels for RTX 5080 production with NVIDIA driver 580.88+
and verify `torch.cuda.is_available()`, RTX 5080 identity, CUDA 13.0 build, compute capability 12.0,
compiled `sm_120` support when exposed, and BF16/FP16 allocations before model-content download.

**Rationale**: PyTorch 2.13 keeps CUDA 13.0 as the default build and removes standard CUDA 12.8/12.9
builds; PyTorch's 2.12 guidance directs Blackwell users to CUDA 13.0+ and a 580.88+ Windows driver.
Sources: [PyTorch 2.13 release](https://pytorch.org/blog/pytorch-2-13-release-blog/) and
[PyTorch 2.12 release](https://pytorch.org/blog/pytorch-2-12-release-blog/).

**Alternatives considered**: CUDA 12.8 is no longer the current standard production wheel and CUDA
13.2 remains experimental. A single requirements file cannot choose both macOS and Windows CUDA wheels
safely. Nightlies are not a production baseline. Automatic MPS-to-CPU fallback after allocation was
rejected because it can multiply latency.

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

## Fixed successful-bundle retention and publication

**Decision**: Stage a request under fixed project `outputs/.work/<request-id>/`, copy normalized
original inputs there before inference, and atomically rename the verified directory to
`outputs/<request-id>/`. On success retain the original image, reference audio, derived voice data,
synthesized speech, pre-lip video, post-lip video, final MP4, manifest, and sanitized metadata. Clean
the unpublished staging directory only on failure or cancellation. The bundle root has no UI,
environment, or request override.

**Rationale**: Directory publication makes history discovery atomic and implements the explicit
full-retention requirement. Keeping every successful intermediate makes later diagnosis and
filesystem-based voice reuse possible, while isolating `.work` prevents incomplete results from
appearing in history.

**Alternatives considered**: Deleting intermediates after success, retaining only the MP4, configurable
output roots, in-app bundle deletion, and automatic expiry were rejected by the clarified scope.

## Read-only history and advisory voice dependencies

**Decision**: Discover history only from validated `outputs/<request-id>/manifest.json` files. Because
browser upload components may copy a selected local file and hide its original absolute path, classify
retained reuse by content: index available retained reference-audio SHA-256 values, prefer a validated
request-ID-bearing retained filename when present, otherwise accept only a unique digest match. Multiple
matches without a verified ID are reported as ambiguous and create no origin edge. Persist an optional
voice-origin record containing the earlier bundle ID, relative artifact path, and digest. On refresh,
compute dependents in memory and report missing/corrupt origins; never repair, delete, or rewrite bundles.
Reuse remains the ordinary filesystem picker plus a fresh consent attestation bound to the new request
ID and uploaded reference digest.

**Rationale**: The filesystem is the operator's mutation interface, so dependency integrity cannot be
enforced transactionally. A read-only scanner with advisory edges reflects external deletion without
pretending to own it. Path containment, no-follow symlink checks, relative manifest paths, digests, and
schema validation prevent a crafted bundle from escaping `outputs/` or becoming a trusted reuse source.

**Alternatives considered**: A database, in-app voice library, cascading deletes, automatic repair,
and mutable dependency indexes were rejected as unnecessary or contrary to the v1 interaction model.

## Cross-platform path and staging safety

**Decision**: Derive `project_root/outputs` in code with no override; reject a root, `.work`, bundle, or
artifact component that is a symlink or Windows reparse point/junction. Server-generate canonical UUID
bundle names. Securely copy picker uploads into staging before inference. Persist only normalized
forward-slash relative artifact paths and reject absolute, drive-qualified, UNC, alternate-separator,
`..`, non-regular, or escaping targets. Cleanup accepts an internal verified staging handle—not a user
path—and removes only the exact inactive `outputs/.work/<uuid>` without traversing links. On startup,
reconcile orphan staging directories only after owner/lock checks; published directories are immutable.

**Rationale**: Lexical prefix checks do not prevent Windows junction or symlink escapes, and filesystem
picker paths are not stable request storage. Copying inputs and combining `lstat`/reparse checks with
strict resolution/containment keeps both publication and failure cleanup bounded.

**Alternatives considered**: String-prefix containment, following symlinks, accepting client-selected
bundle IDs, and a general request cleanup API were rejected as unsafe or incompatible with retention.

## Disk-reserve preflight

**Decision**: Permit only bounded Hub metadata/dry-run inspection before preflight; before any model-file
content transfer, require missing bytes for the full dependency closure plus staging overhead and a
configurable reserve (10 GiB = `10 * 1024^3` bytes by default). Check the cache and fixed `outputs/`
destination filesystems separately, aggregating once only when they share a volume. Before inference,
require the conservative complete-bundle estimate plus reserve; after speech synthesis, refine it from
exact duration before video inference. Report required, available, reserve, logical/physical cache,
incomplete bytes, and manual cleanup candidates without deleting automatically.

**Rationale**: Complete successful bundles and immutable model revisions intentionally accumulate.
Two generation checks prevent starting expensive work that cannot be published while preserving the
operator-selected safety margin.

**Alternatives considered**: Automatic eviction, deleting old bundles, checking only at final export,
and counting sparse/shared-cache logical size as guaranteed reclaimable space were rejected.

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

**Decision**: Keep ordinary tests fully offline with fake Hub/model/worker adapters. Mark real model
download, MPS, and CUDA tests separately. Install the PyTorch 2.13/CUDA 13.0 wheel before bounded main
requirements, and create a separately locked LatentSync Windows worker environment using the same
Blackwell-safe Torch baseline. Refuse production readiness unless both clean environments and their
versioned worker handshake pass. Keep optional quantization acceleration out of the mandatory path.

**Rationale**: This meets the constitution's test-first and cross-platform requirements while avoiding
multi-gigabyte downloads in CI and on macOS. The Qwen package currently pins Transformers and
Accelerate, so the final requirements set must be resolved and tested as one compatible environment.
Source: [Qwen-TTS package metadata](https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml).

**Alternatives considered**: Exact transitive hashes are deferred until both OS lock files can be
generated from tested environments. Conda and container-only deployment were rejected because the
requested path is standard virtual environments and pip.
