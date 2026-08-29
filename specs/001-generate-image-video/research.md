# Research: Generate Image-Conditioned Lip-Synced Video

All technical unknowns from the clarified specification are resolved below. Links point to primary
project or vendor documentation reviewed on 2026-08-27.

## Capability registry for user-provided model links

**Decision**: Accept canonical Hugging Face repository links, resolve them to immutable commit SHAs,
and mark a model ready only when repository metadata matches a reviewed adapter fingerprint. The adapter
declares its roles and whether voice and lip synchronization are native, plus inputs/outputs, accepted
reference types and limits, supported duration range, frame rate, resolution, audio sample rate, dialogue
languages, prompt token capacity, devices, precision, offload and quantization policy, accelerator and
host memory profile, and allowed weight formats. Every one of these is a measured profile field.

**Rationale**: User-selected repositories cannot safely or reliably map to one generic call signature.
Diffusers auto-pipelines select classes from repository configuration, but image-to-video architectures
still expose model-specific parameters. The registry satisfies model choice without executing arbitrary
Hub code. Source: [Diffusers pipeline overview](https://huggingface.co/docs/diffusers/en/api/pipelines/overview).

**Alternatives considered**: `trust_remote_code=True` was rejected as incompatible with the security
requirement. Accepting any model tagged image-to-video was rejected because tags do not prove interface,
memory, voice, or lip-sync compatibility. Baking any one model's duration, frame rate, or token limit
into shared code was rejected outright: the previous stack encoded 49 frames, 8 FPS, and a 226-token
prompt capacity as global truths, and every one of those had to be unpicked when the model changed.

## Reference joint audio/video adapter

**Decision**: Use `MiniMaxAI/MiniMax-H3` in its **Ref2VA** (omni-reference) mode as the default reviewed
video profile, generating video and stereo audio jointly in one invocation. Load it through the official
Diffusers/Transformers classes with `trust_remote_code=False`. Retire the separate text-to-speech and
lip-synchronization providers entirely.

**Rationale**: H3 is an omni-modal system that produces synchronized video and native stereo audio from a
multimodal reference context. Because the mouth movement and the voice are predicted together by one
model, the previous three-provider pipeline collapses to a single stage. That removes the cross-provider
timebase bridge, the post-generation face preflight, the speech-first ordering constraint, the model
lease choreography between heavyweight providers, and the isolated worker process that existed only
because two providers pinned conflicting dependency stacks. Source:
[MiniMax-H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3).

**Measured profile fields** (all recorded per adapter, never as architecture constants):

| Field | Value from the model card |
|-------|---------------------------|
| Output duration | 4-15 s (**target 6-10 s; the ceiling is measured on the RTX 5080, not assumed**) |
| Output frame rate | 24 FPS |
| Output resolution | Short side 768 by default; 2K only via H3-Regenerate-2K |
| Output audio | 32 kHz stereo |
| Dialogue languages | 11 with stable support: Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish |
| Ref2VA references | Images `<= 9`; audio `<= 3` clips of 2-15 s each; `<= 12` files total |
| Precision | BF16; released weights are CFG-distilled |

**Alternatives considered**: CogVideoX-5B-I2V plus Qwen3-TTS plus LatentSync 1.5 was the previous stack.
It fitted the card comfortably but produced only 6.125 s per generation, which forced either a
ping-pong loop with visibly repeating motion or chained generation with identity drift, and it required a
separately locked worker process to reconcile conflicting dependency pins. LTX-2.5 and Wan 2.2 were
considered as longer-duration single-video alternatives but do not generate voice-cloned speech jointly,
so they would have retained the multi-provider architecture.

## No remote code required — via the root modular layout, not `Ref2VA/`

**Decision**: Load H3 through the repository root's `modular_model_index.json` as a
`MiniMaxH3ModularPipeline`, selecting the `ref2va` workflow, with `trust_remote_code=False`. Do **not**
load the `Ref2VA/` subfolder.

**Rationale**: The repository ships two loading paths for the same weights, and only one of them is
usable under the constitutional prohibition on remote code.

`Ref2VA/model_index.json` names `MiniMaxH3Pipeline`, `MiniMaxH3DiTModel`, `MiniMaxH3VideoVAE`,
`MiniMaxH3AudioVAE`, and `MiniMaxH3Qwen3VLHFEncoder`. **No upstream Diffusers or Transformers release
exports any of those names**; they belong to a MiniMax fork, which its `"_diffusers_version": "0.32.2"`
stamp reflects. Worse, `Ref2VA/video_vae/config.json` and `Ref2VA/audio_vae/config.json` each carry an
`auto_map` pointing at bundled `.py` modules shipped beside the weights, so that path requires
`trust_remote_code=True` by construction.

The repository root describes the same checkpoint in upstream Diffusers' modular format, naming only
classes the released wheels genuinely export, and its component directories contain **no `.py` files and
no `auto_map`**:

| Component | Library | Class |
|---|---|---|
| pipeline | diffusers | `MiniMaxH3ModularPipeline` (blocks `MiniMaxH3Blocks`) |
| transformer_ref | diffusers | `MiniMaxH3Transformer3DModel` |
| vae | diffusers | `AutoencoderKLMiniMaxH3` |
| audio_vae | diffusers | `AutoencoderKLMiniMaxH3Audio` |
| scheduler, audio_scheduler | diffusers | `MiniMaxH3Scheduler` |
| text_encoder | transformers | `Qwen3VLForConditionalGeneration` |
| tokenizer | transformers | `Qwen2TokenizerFast` |
| processor | transformers | `Qwen3VLProcessor` |

Verified against `diffusers==0.40.0` and `transformers==5.16.1`: all nine resolve. The prohibition on
remote code therefore holds with no exception — but only because the loading path changed.

This entry supersedes an earlier one that read the absence of a `custom_pipeline` key in
`Ref2VA/model_index.json` as proof that no remote code was involved. That inference was wrong:
`auto_map` inside a *component* config is the other, more common way a repository ships executable code,
and it was not checked.

**Alternatives considered**: Granting a scoped `trust_remote_code` exception for the `Ref2VA/` VAEs was
rejected — the root path obtains the same weights with no exception at all. Vendoring the two VAE modules
into the repository was rejected as a maintenance burden that a supported upstream path makes pointless.

## Fitting a 33B omni-model into 16 GB VRAM and 64 GB RAM

**Decision**: Run the production profile with a reviewed quantized checkpoint plus layer-wise/sequential
CPU offload, gate on **both** a 13.5 GiB peak allocator-reserved accelerator ceiling and a configured host
system-memory ceiling, and treat resident footprint per precision as a measured release value.

**Rationale**: The transformer is 33B dense, but roughly 13B of that sits in AdaLN branches whose
modulation outputs can be precomputed and cached, so those parameters need not be loaded for
inference-only deployment -- approximately 20B effective. The text encoder additionally carries
Qwen3-VL-32B weights and consumes only its 50th-layer hidden states, so a truncated load is possible.
Neither saving is safe to assume; both are load-time engineering decisions with large memory consequences
and belong in measured gates. Host RAM becomes a first-class budget because layer-wise offload keeps the
resident model in system memory and streams it to the card: at BF16 the model does not fit 64 GB, while at
INT8 it does comfortably, which is why a quantized checkpoint is an expected part of the production
profile rather than a fallback. Sparse attention is **not** in the initial open-source release, so
inference is full-attention only and memory grows with the packed multimodal sequence -- a direct argument
for the low end of the duration range.

**Alternatives considered**: BF16 without offload is impossible on 16 GB. The model card's SGLang example
uses `--num-gpus 4 --ulysses-degree 4`, but that is a speed-mode deployment recipe, not a stated minimum,
and single-GPU offloaded inference is the supported trade of latency for capacity. Multi-GPU was rejected
because the target machine has one card.

**Measured checkpoint sizes** (from the Hub blob listing, root layout, BF16):

| Component | Size | Needed by `ref2va` |
|---|---:|---|
| `text_encoder` (Qwen3-VL) | 62.13 GiB | yes, truncated at `text_encoder_layer` |
| `transformer_ref` | 61.73 GiB | yes |
| `transformer` | 61.73 GiB | **no** — `fl2va`/`t2va` only |
| `vae` | 9.70 GiB | yes |
| `audio_vae` | 0.56 GiB | yes |
| **`ref2va` working set** | **134.12 GiB** | |

Two consequences. First, `MiniMaxH3ModularPipeline.load_components(workflow="ref2va")` loads only the
components that workflow's blocks use, which excludes the 61.73 GiB `transformer` entirely; loading the
whole repository instead would waste that on every run. Second, 134 GiB of BF16 weights against a 64 GB
host ceiling makes quantization **mandatory rather than expected** — roughly INT4 to be resident, and even
INT8 does not fit. The earlier claim in this section that "at INT8 it does comfortably" was written before
these sizes were measured and is wrong; INT8 lands near 67 GiB, above the ceiling. The precision that
actually fits is a spike measurement, not a planning assumption.

The checkpoint declares its own limits as pipeline properties — `fps`, `min_duration`, `max_duration`,
`audio_sampling_rate`, `audio_channels`, `canvas_multiple`, `text_encoder_layer` — so `ModelProfile` reads
them from the loaded pipeline rather than restating them. That is the mechanism which keeps measured
values out of shared code.

## Unbounded inference time

**Decision**: Impose no latency target, SLA, maximum runtime, runtime estimate, or cost-confirmation gate.
Record measured wall-clock time as a baseline only.

**Rationale**: Layer-wise offload of a 20B-effective model on a 16 GB card trades time for feasibility by
design; a run taking hours is the expected operating point, not a fault. The constitution requires bounded
*allocation* and reproducible parameters, which the dual ceilings and recorded effective profile preserve.
Bounding wall-clock time would either fail correct runs or force a smaller model.

**Alternatives considered**: A configurable timeout was rejected because any threshold would be arbitrary
and would abort valid work. Runtime estimation was rejected because offloaded throughput varies too widely
to estimate honestly.

## Reference semantics: image plus voice timbre anchor

**Decision**: Accept one or more still images anchoring subject identity and appearance, plus exactly one
audio recording used solely as a **voice timbre anchor**. Impose no application maximum on image count --
the profile's measured reference limit is the only bound. Reject video references.
Carry spoken content in the prompt as `<d>[language]...</d>` dialogue tags built from the speech script.

**Rationale**: `<d>` is a real special token added to the H3 tokenizer configuration, so dialogue content
belongs in the prompt rather than in an audio reference. The reference recording conditions timbre only:
it is never played back, never mixed into the output, and never treated as spoken content, which is why
it **must say different words from the script** -- a rule surfaced in the UI at the point of upload.
Video references are excluded on token cost: a 15 s, 1280x768, 24 FPS clip costs roughly 86,000 tokens on
its own under the f16t4d24 latent design with 1x2x2 patchify, which breaches the memory ceiling before
generation starts.

**Alternatives considered**: Supplying the script as reference audio was rejected because it inverts the
model's design and would make the reference both timbre and content. Allowing video references was
rejected on the token arithmetic above, despite the profile permitting up to three clips. Capping images
at one was rejected as a carryover from the previous single-conditioning-frame image-to-video stage: it is
not a property of the omni-reference mode, and additional views of the same subject are the cheapest
available lever on identity stability.

## Locally built prompt structuring

**Decision**: Build prompt assembly inside the application from the published Prompting Guidance, and
retain the assembled prompt actually submitted in every successful bundle.

**Rationale**: H3-Context-IR -- the module the model card credits with much of the output quality -- is
**not** part of the open-source release and is offered only as a hosted API. Local-only operation forbids
calling it, so the application must do its own instruction parsing and context serialization. Retaining
the exact assembled prompt keeps results explainable and lets the structuring improve without changing any
contract. This is a recorded quality risk, not a solved problem.

**Alternatives considered**: Calling the hosted H3-Context-IR API was rejected as incompatible with local
operation and with the no-network-during-inference rule. Passing raw user text straight through was
rejected because the model card explicitly attributes quality loss to unstructured context.

## Local output ceiling of 768p

**Decision**: Treat 768p short side as the maximum local output resolution and place 2K out of scope.

**Rationale**: H3-Regenerate-2K is not open-sourced and is reachable only through the hosted API, which
local operation forbids. Claiming 2K support would be false for this deployment.

**Alternatives considered**: A conventional external super-resolution pass was rejected for v1; it is a
different model with its own review, memory profile, and quality characteristics.

## Duration selection and script fit

**Decision**: Produce every request from exactly one generation. Because audio and video are generated
jointly, duration is an **input** to that generation, not a measurement taken from synthesized speech.
Derive a suggested duration from the trimmed script and a per-language speaking-rate field in the adapter
profile, clamp it to the profile's supported range, and let the operator override it anywhere in that
range. Perform no pre-generation script-fit check and never reject a request for script length. Never
trim, time-stretch, truncate, or partially omit speech.

**Rationale**: The previous stack could measure speech because a dedicated text-to-speech stage produced
it before video planning. That stage no longer exists, so "does this script fit?" is not a question the
system can answer before generating. A soft, overridable suggestion is honest about that; a hard reject
gate would assert a measurement the architecture cannot take. Per-character speaking rate varies several
fold across the profile's languages, which is survivable in a default and was fatal in the old
110-character cap precisely because that cap was a hard gate.

**Alternatives considered**: Rejecting over-long scripts was rejected as unverifiable before generation.
Verifying delivery afterwards with speech recognition was rejected for v1: it needs its own model, memory
budget, and language coverage, and only reports failure after a multi-hour run. Always requesting the
profile maximum was rejected as the most expensive option on every request. Ping-pong looping of a base
clip, chained continuation, and concatenated
independent clips were all viable when audio was produced separately and muxed. None survive joint
generation: a clip whose speech is baked in cannot be repeated, reversed, or spliced without corrupting
the audio. Splitting a long script across several generations and concatenating them was rejected for v1
because it reintroduces multi-generation cost, cross-clip identity drift, and audio seams.

## Motion-prompt capacity and truncation reporting

**Decision**: Impose no application maximum on the motion prompt. Record each video adapter's text-encoder
capacity in its profile as a measured value. When a prompt exceeds it, truncate to capacity and record the original,
retained, and discarded lengths as an explicit override surfaced in the UI and in request metadata.

**Rationale**: The pipeline truncates internally regardless, so the only real choice is whether the user
is told. Reporting it as an override matches how duration and parameter overrides are already handled and
preserves the rule that no effective parameter changes silently. Truncating a conditioning prompt costs
quality only, unlike truncating speech, which would violate a functional requirement.

**Alternatives considered**: Rejecting overlong prompts reintroduces a limit the clarification removed.
Passing them through unchecked leaves metadata recording a prompt that was not the one actually used.

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

A reviewed non-safetensors weight is an explicit exception to the general safetensors policy only
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
compiled `sm_120` support when exposed, and BF16/FP16 allocations before model-content download. Also
measure installed and available **host system memory** at startup and record it alongside accelerator
capacity: the target machine has 64 GB, and layer-wise offload makes host RAM a gating resource rather
than an incidental one.

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
one face/mouth target before video generation. After video inference, run the effective lip provider's
lightweight detector/tracker across generated frames and require one consistent usable target before
starting generation.

**Rationale**: The user chose provider-specific audio limits and explicit language. Performing consent,
face, and audio validation before loading models prevents expensive invalid work and records the safety
decision at the request boundary. No post-generation face pass is needed, because mouth movement is
produced jointly with the audio by the same model rather than applied to an already-generated clip.

**Alternatives considered**: One universal audio limit was rejected by clarification. Automatic
language detection and multi-face selection are explicitly out of scope.

## Lip-sync publication policy

**Decision**: Treat adapter exceptions, invalid media, silent generated speech, or failed container
export as technical failures. Do not compute or enforce a lip-sync quality score in v1; every
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

**Alternatives considered**: Returning the adapter's raw file without stream validation can
publish silent or mismatched media. Shell-form FFmpeg commands were rejected for path and injection
safety.

## Fixed successful-bundle retention and publication

**Decision**: Stage a request under fixed project `outputs/.work/<request-id>/`, copy normalized
original inputs there before inference, and atomically rename the verified directory to
`outputs/<request-id>/`. On success retain the original images, reference audio, derived voice data, the
assembled prompt, the decoded video and audio, the final MP4, manifest, and sanitized metadata. Clean
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
require the conservative complete-bundle estimate plus reserve; refine it from the effective duration
once the duration decision is fixed, and monitor free space during every write stage. Report required, available, reserve, logical/physical cache,
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
database, and worker queue are unnecessary for a local single-user v1. Because a run may last hours, the
UI additionally reports a monotonic completion fraction during inference and decoding, and the model
library is read-only for the duration of an active generation.

## Test and packaging strategy

**Decision**: Keep ordinary tests fully offline with fake Hub and model adapters. Mark real model
download, MPS, and CUDA tests separately. Install the PyTorch 2.13/CUDA 13.0 wheel before the bounded
application requirements. Run the blocking stack-compatibility spike -- proving the H3 classes load with
`trust_remote_code=False` -- before any architectural foundation work. Refuse production readiness unless
the clean environment, the 13.5 GiB peak-reserved accelerator gate, the host system-memory ceiling, and
the measured duration ceiling on the target card all pass. Make the offline suite profile-agnostic by
running it a second time against a fixture profile whose duration, frame rate, resolution, language set,
reference limits, and token capacity all differ from H3's.

**Rationale**: This meets the constitution's test-first and cross-platform requirements while avoiding
multi-gigabyte downloads in CI and on macOS. Collapsing to a single provider removes the previous need for
two separately locked environments and a versioned worker handshake, so there is now one dependency set to
resolve and test. The profile-agnostic second run is the concrete regression guard against re-baking one
model's constants into the architecture, which is exactly what happened with the previous stack.

**Alternatives considered**: Exact transitive hashes are deferred until the lock file can be generated
from a tested environment on each OS. Conda and container-only deployment were rejected because the
requested path is standard virtual environments and pip. Asserting H3's specific numbers directly in
shared tests was rejected for the same reason those numbers are kept out of shared code.
