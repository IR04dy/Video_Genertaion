# Research: Generate Image-Conditioned Lip-Synced Video

All technical unknowns from the clarified specification are resolved below. Links point to primary
project or vendor documentation reviewed on 2026-08-27, except where a later measurement supersedes
them — see **Stack decision superseded** immediately below.

## Standing project constraint (stated 2026-09-01)

**Open-weight models and open-source tools only. Nothing is bought. Not for commercial use.**

This is the constraint several decisions in this document already rest on, and it is recorded here
because the architecture cannot express it: `ModelProfile` carries no licence field by design, and a
test enforces that absence. Licence obligations therefore live as prose here and in
`requirements.txt`, never in code.

What it settles:

* **CPML is acceptable.** `coqui/XTTS-v2`'s weights forbid commercial use and nothing else, so they
  are usable here. Chatterbox was rejected on **packaging** grounds, not licence grounds — the two
  reasons are independent and should not be conflated if either is revisited.
* **No paid or hosted inference.** No commercial APIs, no rented accelerators. Every model runs
  locally on the one RTX 5080, which is also why the 13.5 GiB reserved ceiling is a hard constraint
  rather than a tuning preference.
* **If this ever goes commercial**, the XTTS-v2 weights must be replaced before release. That
  tripwire is written into `requirements.txt` beside the pin itself.


## Stack decision superseded (2026-09-01)

**Decision**: Abandon `MiniMaxAI/MiniMax-H3`. Cover the three roles with **two models behind one
adapter**: `Wan-AI/Wan2.2-S2V-14B` for VIDEO + LIP_SYNC, and a still-unresolved TTS for VOICE.

**Rationale**: H3 does not fit this hardware, and the reason is arithmetic rather than engineering.
`text_encoder` is 62.13 GiB and `transformer_ref` 61.73 GiB at BF16. At int4 the transformer is still
**15.5 GiB, over the 13.5 GiB accelerator ceiling** — so even a successful load would stream 15.5 GiB
across PCIe on every denoising step. Every observed failure (a 47 GiB quantization peak, Windows page-file
error 1455, 2082 s/shard thrashing) was a symptom of that. The lesson worth carrying forward: check
*largest indivisible component* against the card ceiling before checking total working set, because
offload fixes the total and cannot fix the component.

Wan2.2-S2V's largest component is 30.35 GiB BF16, which is **7.6 GiB at int4** — under the ceiling with
room. Nothing co-resides: each stage loads, runs, and frees.

**Load path**: DiffSynth, not Diffusers. Neither `WanSpeechToVideoPipeline` nor
`WanS2VTransformer3DModel` exists in any Diffusers release — [huggingface/diffusers#12257] is still open —
so that route would mean running a fork of the core library. `diffsynth` is on PyPI, is additive, leaves
Diffusers stock, and ships `examples/wanvideo/model_inference_low_vram/Wan2.2-S2V-14B.py`: an official
low-VRAM path using disk offload under an explicit `vram_limit`. That makes the host-RAM exhaustion that
killed H3 architecturally impossible rather than merely avoided.

**Security posture is unchanged but the vectors differ.** Neither repository declares `auto_map`, so
`trust_remote_code` stays false. Wan additionally ships its T5 encoder and VAE as `.pth` **pickles**,
which `auto_map` assertions say nothing about; `torch.load`'s `weights_only=True` default from torch 2.6
closes that, and the stack gate asserts the default has not regressed.

**Alternatives considered**: LTX-2 is the only joint audio+video pipeline in released Diffusers, and was
rejected because its `__call__` accepts no speaker reference — it scores a scene but cannot clone a named
voice. A three-stage split (Wan I2V + TTS + LatentSync) was rejected because lip sync composited after
generation is weaker than lip sync conditioned into it, and LatentSync is equally a pinned package, so the
stricter reading buys no purity while costing a model.

[huggingface/diffusers#12257]: https://github.com/huggingface/diffusers/pull/12257

## Voice packaging conflict (resolved 2026-09-01)

**Status**: resolved. `coqui-tts==0.27.5` (XTTS-v2) is pinned; the VOICE half of the stack gate is live.

**Requirement that drove it**: Arabic. This is a hard product requirement, not a preference, which is
what rules out CosyVoice2-0.5B (9 languages, no Arabic) despite its Apache-2.0 licence and clean
packaging.

**Rejected: `ResembleAI/chatterbox`.** The capability fits — 23 languages including Arabic, MIT, 2.0 GiB
— but `chatterbox-tts` 0.1.7 hard-pins `torch==2.6.0`, `transformers==5.2.0`, `diffusers==0.29.0` and
`gradio==6.8.0`, and sources `resemble-perth` from a git URL. Confirmed in the upstream `pyproject.toml`,
so not a wheel-metadata artifact. Installing it on the production host downgrades torch below the cu130
build and `sm_120` support, silently disabling the RTX 5080 — the failure `requirements.txt` warns
about, arriving through a dependency rather than a missing wheel.

**Chosen: XTTS-v2 through `coqui-tts`.** Verified against the sdist's own `pyproject.toml`, not against
published summaries:

| Property | Finding | Consequence |
|---|---|---|
| torch | **absent from core deps**; `torch>=2.2` appears only in the `cpu`/`cuda` extras | a bare install touches torch zero times |
| torchaudio | same, `>=2.2.0`, extras only | `2.11.0+cu130` already installed, untouched |
| transformers | `>=4.57`, a floor | satisfied by the `==5.16.1` Wan pins; the two halves do not contend |
| `[tool.uv.sources]` cu128 index | uv-only | pip ignores it entirely |
| librosa | `>=0.11.0` | raised this project's floor from `>=0.10` |
| python | `>=3.10,<3.15` | 3.11 sits inside |
| Arabic | dedicated abbreviation, symbol, ordinal (`([0-9]+)(ون|ين|ث|ر|ى)`) and punctuation rules, plus an Arabic normalization test case | first-class, not incidental to a multilingual checkpoint |

The extras structure is the whole reason this works, and it is fragile against a well-meaning edit:
`coqui-tts[cuda]` would resolve torch from Coqui's own cu128 index and strip `sm_120`. `requirements.txt`
records the omission as deliberate for that reason.

**Two licences, not one.** Earlier notes conflated them. The `coqui-tts` **package** is MPL-2.0 and
imposes nothing here. The `coqui/XTTS-v2` **weights** are CPML, which forbids **commercial** use. This
project is non-commercial, so CPML is satisfied today; `requirements.txt` carries it as a tripwire, and
the recorded exit if that changes is Chatterbox in a separate virtualenv invoked as a subprocess.

**Unpickling, voice half.** `coqui/XTTS-v2` ships no Python, but four `.pth` pickles (`model.pth`,
`dvae.pth`, `mel_stats.pth`, `speakers_xtts.pth`). The video gate's `torch.load` default assertion does
not cover them, because Coqui loads them with its own code. Coqui passes `weights_only=True` explicitly
on the XTTS inference path (guarded by `is_pytorch_at_least_2_4()`), and `test_xtts_inference_path_
unpickles_safely` asserts it still does. Scope is narrow on purpose: nine `torch.load` calls elsewhere in
the package — Bark, Tortoise, neuralhmm, overflow, and the XTTS *trainer* — pass no `weights_only` at
all. None is on this application's loading path, and none may be brought onto it without re-running this
review.

## The transformers ceiling (measured on the RTX 5080 host, 2026-09-01)

**Status**: resolved, and it changes a pin T008 had already recorded as settled.

Installing the resolved stack on the production host falsified one of its assumptions. `coqui-tts`
declares `transformers>=4.57` with **no upper bound**, and that floor was read as compatibility with the
`==5.16.1` the video half had pinned. It is not:

```
TTS/tts/models/xtts.py -> TTS/tts/layers/xtts/gpt.py -> TTS/tts/layers/tortoise/autoregressive.py
  from transformers.pytorch_utils import isin_mps_friendly   # removed in transformers 5.0
```

The import is on the XTTS **inference** path, reached through the GPT layer, so it cannot be dodged by
not using Tortoise. The unbounded upstream floor is a packaging bug: the resolver installs a 5.x that
cannot import.

**Resolution**: `transformers==4.57.1`, a ceiling as much as a pin. Verified on the host that the video
half is unaffected — `WanVideoPipeline`, `ModelConfig`, `wan_video_dit_s2v` and `Wav2Vec2Model` all
import on 4.57.1. `diffsynth` declares transformers with no bound; its 5.x-only imports (Qwen3_5,
Qwen3VL, Ministral3, DINOv3, Siglip2) are lazy and belong to pipelines this application never loads.

**Second-order consequence, and the non-obvious one.** transformers 4.57.1 requires
`huggingface-hub<1.0`; gradio 6.x requires `huggingface-hub>=1.16`. They cannot coexist, so gradio is
pinned `>=5.49,<6` — a **ceiling**, where the file previously carried a floor of `>=4.44`. The trap is
the direction of the failure: raising gradio to 6.x breaks **voice**, not the UI, which is not where
anyone would look. Both `requirements.txt` and this section say so for that reason.

`torchcodec==0.16.0` is additionally required, because coqui-tts enforces it at import time from
torch>=2.9. Installed as the plain wheel rather than through `coqui-tts[codec]`, on the same reasoning
that keeps the voice requirement extras-free: the extra resolves through Coqui's cu128 index.

Two H3-era packages were removed as unused and unreferenced by any application module: `diffusers`
(replaced by diffsynth) and `hf-gradio` (a gradio 6 companion). `pip check` is clean.

**Verified on the host after the change**: torch `2.13.0+cu130`, CUDA available, capability `(12, 0)` —
`sm_120` intact, which was the property every packaging decision here was protecting.

**Constraint inherited by T040**: Arabic has a **166-character** per-generation limit
(`TTS/tts/layers/xtts/tokenizer.py`, `char_limits["ar"]`). Longer scripts must be sentence-split —
`pysbd` is already a coqui-tts core dependency — and the segments concatenated before the waveform is
measured and handed to Wan. This is a measured model constraint and belongs in the adapter's profile,
never as a literal in application code.

## Wan2.2-S2V on the RTX 5080: measured, and blocked (2026-09-01)

**Status**: the VIDEO role is **unresolved**. Every other part of the stack is proven on the host.
Six configurations were run; none produced a single denoising step. The obstacle is measured and
attributed, and it is **not** the model.

### What works

| Stage | Result |
|---|---|
| FFmpeg 9.0.1 shared libraries | installed, torchcodec loads |
| XTTS-v2 Arabic | **PASS** -- 47.5 s from a 329-char script, 24 kHz, 0.25x realtime, 2.69 GiB VRAM, judged acceptable by the operator |
| Arabic splitting | 329 chars -> 2 segments (162 + 166), no text lost |
| Download | **45.77 GiB / 49 files in 13.7 min** |
| Wan pipeline load | **PASS** -- 44 s, 11.45 GiB host RSS |
| Audio conditioning, VAE encode | reached and working (VAE encode 4 chunks in 11 s) |

### What does not

Six runs, one signature, no denoising step completed in any of them:

| # | Quantize | Offload | Resolution | Frames | Steps | Peak VRAM | Power | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | none (bf16) | cpu | 448x832 | 81 | 40 | 15.9 GiB | 60 W | no step in 26 min |
| 2 | int8 w8a16 | cpu | 448x832 | 81 | 4 | 15.86 GiB | 61 W | no step in ~12 min |
| 3 | none (bf16) | cpu | 320x576 | 33 | 8 | 15.8 GiB | 61 W | no step |
| 4 | int4 w4a16 | cuda | -- | -- | -- | -- | -- | load error (sharded) |
| 5 | int4 w4a16 | cuda | -- | -- | -- | -- | -- | load error (merged), then `mslk` missing |
| 6 | int8 w8a16 | cuda | 320x576 | 33 | 8 | 15.85 GiB | 61-64 W | no step |

**int8 changed peak VRAM by 0.04 GiB. A 5x cut in activation budget changed it by 0.1 GiB.** VRAM
pins at ~15.8 GiB regardless, because DiffSynth's `vram_limit` manager simply fills the card with
cached layers. Neither weights nor activations are the constraint.

### The actual bottleneck, and why it is not the hardware

Profiling during run 3 -- 20 seconds of measurement that was worth more than two of the runs above:

* disk read **0.2 MB/s** -- not paging; host RAM pressure was a red herring
* process CPU **99-100% of one core**
* GPU **100% "utilisation" at 61 W** of a ~360 W card, 2910 MHz, 37 C

That is a CPU-bound, Python-orchestrated per-layer transfer loop with the GPU starved between
copies. Host-to-device bandwidth was then measured directly on this host:

| Path | Measured |
|---|---|
| pageable | **8.1 GB/s** |
| pinned | **36.8 GB/s** (PCIe 5.0 x16) |

Streaming the full bf16 DiT (~28 GiB) once per step should therefore cost **0.76 s** pinned or
**3.5 s** pageable. Observed: **>1200 s**. The implementation is roughly **350x slower than the
worst-case bandwidth allows**, so the 64 GiB of host RAM and the PCIe link are both adequate and
neither is implicated. `diffsynth==2.1.5`'s offload orchestration is.

Run 6 also shows `offload_device="cuda"` does not make the model resident: process RSS stayed at
19.8 GiB and the signature was unchanged. `enable_vram_management` streams regardless of the
device asked for.

### Two upstream defects found

**1. `DiskMap` corrupts CUDA tensors.** `diffsynth/core/vram/disk_map.py` re-opens the safetensors
handle every ~1 GiB (`flush_files`, `buffer_size=10**9`). Tensors already returned are protected
only on the CPU path:

```python
if isinstance(param, torch.Tensor) and param.device.type == "cpu":
    param = param.clone()
```

CUDA tensors are not cloned, so re-opening frees their storage underneath them and the next access
raises `RuntimeError: Attempted to access the data pointer on an invalid python storage`. Sharding
is irrelevant -- a single merged 30.35 GiB file fails identically, which is how the real cause was
found. Workaround, verified to get past it: `DIFFSYNTH_DISK_MAP_BUFFER_SIZE=100000000000000`.
`safetensors` 0.8.0 itself loads to CUDA correctly; the bug is DiffSynth's.

**2. `torchao_int4_w4a16` needs an absent kernel package**, `ImportError: Requires mslk >= 1.0.0`.
Not pursued, because run 6 showed int8-resident makes no difference anyway.

### Windows-specific hazard worth remembering

The first bf16 attempt, with `offload_device` left at DiffSynth's `None` default, put every
parameter on CUDA and peaked at **41.89 GiB on a 16 GiB card**. It did not fail: WDDM silently
spills to shared system memory, so the load "succeeds" and then runs over PCIe. **On Windows the
OOM you want does not happen** -- it becomes a silent performance collapse instead. Explicit
placement is mandatory, not a tuning preference.

### A hidden network download at load time

Found by noticing an untracked `models/` directory after the runs, not by any check:

```
models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/{tokenizer.json,spiece.model,...}
```

`WanVideoPipeline.from_pretrained` takes a `tokenizer_config` whose **default is not None**:
`ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")`. Leaving it
unset makes DiffSynth resolve and **download the umt5 tokenizer from the network at load time**,
into `./models` (its `local_model_path` default), even though every other component was pinned to a
local path and the full repository was already on disk.

This directly violates T093's "zero hidden downloads during inference" and the closed-model policy
in `closed-models.md`. It is also invisible: 21 MiB, no progress output, no error.

Whatever runtime is chosen, the adapter must pass an explicit local `tokenizer_config` and the
acceptance test must assert that no network call occurs during a generation -- an assertion the
current gate does not make, because the gate only inspects package contents and repository
metadata, never a live load.

### Consequences for the profile and the architecture

* `frame_rate`, resolutions, `duration_range_seconds` and step count remain **unmeasured**. Nothing
  from the vendor table below has been confirmed by a generation, so none of it may enter a profile.
* The **13.5 GiB reserved ceiling is breached by every configuration tried** (15.8-41.9 GiB), so
  even a working run would currently fail the resource contract.
* The measured working set is **45.77 GiB / 49 files**, not the 42.60 GiB / 15 files recorded from
  the Hugging Face API below. The audio encoder ships its weights three times -- `model.safetensors`,
  `pytorch_model.bin` and `flax_model.msgpack`, ~1.26 GiB each -- and only the safetensors copy is
  loaded. Published metadata was wrong; the measurement stands.
* Arabic carries a **166-character** per-generation limit into T040 regardless of which video
  runtime is chosen.

### Open decision

The weights are not implicated and the hardware is not implicated, so **replacing the model is not
the indicated fix**. The candidate is replacing the *runtime*: ComfyUI's native Wan S2V nodes, with
GGUF Q4/Q5 quantization, are the established path for 16 GiB cards. That is an architectural change
rather than a pin change -- DiffSynth is consumed as a library, while ComfyUI would be driven as a
subprocess, which touches the adapter design and the closed-set review in `closed-models.md`. The
alternative is a smaller audio-driven model that runs in-process. Not yet decided.

## Measured constants for the Wan2.2-S2V profile (2026-09-01)

Harvested from the vendor's own configs and low-VRAM example; each still requires confirmation by a real
generation on the RTX 5080 before T040 writes it into a profile.

| Field | Value | Source |
|---|---|---|
| `frame_rate` | 16 fps | low-VRAM example; `save_video_with_audio(fps=16)` |
| frame-count grid | **4n+1** (81 frames = 5.0 s) | `num_latent_frames = (num_frames-1)//4 + 1`; VAE `temperal_downsample` gives 4x |
| audio conditioning rate | 16000 Hz | `wav2vec2.../preprocessor_config.json`; `librosa.load(sr=16000)` |
| low-VRAM resolution | 448x832 | low-VRAM example |
| denoising steps | 40 | low-VRAM example |
| audio injection | 12 of 40 layers | `audio_inject_layers` in `config.json` |
| long-form mechanism | FramePack + multi-clip | `enable_framepack: true`; `Wan2.2-S2V-14B_multi_clips.py` |
| working set | 42.60 GiB / 15 files | Hugging Face API |

The frame grid is the whole of the residual timebase seam. Because speech is synthesized before video,
its duration is **measured rather than estimated** — strictly better than both the previous three-model
design (which estimated from speaking rates) and H3 (which fitted speech into a preset). What remains is
rounding the measured seconds up to the next legal frame count and padding the audio tail, which is a
function, not a stage. `duration_range_seconds` is still unmeasured: FramePack removes the hard wall, so
the ceiling is whatever stays practical on this card.

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

> **Superseded 2026-09-01.** Retained as the record of what was measured on MiniMax-H3; see
> **Stack decision superseded** above. Not the current stack.

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

> **Superseded 2026-09-01.** Retained as the record of what was measured on MiniMax-H3; see
> **Stack decision superseded** above. Not the current stack.

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

> **Superseded 2026-09-01.** Retained as the record of what was measured on MiniMax-H3; see
> **Stack decision superseded** above. Not the current stack.

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

## Quantization backend: measured, not assumed

> **Superseded 2026-09-01.** Retained as the record of what was measured on MiniMax-H3; see
> **Stack decision superseded** above. Not the current stack.

**Decision**: Quantize the two large transformers with **optimum-quanto** at int4, resident in host
RAM, and stream them to the accelerator with sequential CPU offload. Not bitsandbytes.

**Source**: measured on the Windows RTX 5080 host, 2026-08-31, torch 2.13.0+cu130.

Only one configuration fits, and the arithmetic decides it before any backend does:

| Placement | Resident | Verdict |
|---|---|---|
| int4 in host RAM | 16.5 + 16.4 + 10.3 GiB VAE = **~43 GiB** | fits 64 GB |
| int8 in host RAM | 31.1 + 30.9 + 10.3 = **72.2 GiB** | over 64 GB |
| int4 on the accelerator | 15.5 GiB for *one* component | over the 13.5 GiB ceiling |

`text_encoder` is 62.13 GiB and `transformer_ref` 61.73 GiB at BF16. Neither fits on a 16 GB card at
any precision, so a GPU-resident load was never possible and `device_map="auto"` fails by design
rather than by misconfiguration. int8 is not a fallback: it exceeds the host ceiling even when the
VAEs are quantized too. **The requirement is therefore int4 quantization performed on the CPU**, and
each backend was tested against exactly that.

| Backend | Result |
|---|---|
| bitsandbytes, CUDA | Quantizes correctly, but cannot hold a component within the VRAM ceiling. |
| bitsandbytes, CPU | Native access violation mid-load, twice, at different weights (543/1058, then 0/1058), at 1.5 GiB RSS. Not memory, and `quantize_4bit` and `Params4bit(...).to("cpu")` both pass in isolation on every realistic shape — the corruption appears only through the loader's own conversion path. Unusable. |
| torchao 0.18, CPU | No int4 CPU path exists. `PLAIN`/`PRESHUFFLED` require `mslk >= 1.0.0`, which is not published (PyPI `mslk` is an unrelated 0.0.0 stub — do not install it); `TILE_PACKED_TO_4D` is CUDA tinygemm; `PLAIN_INT32` raises `NotImplementedError` for CPU. |
| optimum-quanto, CPU | **Works.** 3.76x compression on a 4096x4096 BF16 linear (33,556,137 to 8,915,481 bytes serialized) with a working CPU forward pass. The shortfall from 4x is scale and zero-point overhead. |

**Trap worth recording**: both libraries export a `QuantoConfig`, and the keyword differs —
`transformers.QuantoConfig(weights="int4")` versus `diffusers.QuantoConfig(weights_dtype="int4")`.
The `BitsAndBytesConfig` pair has the same name collision but at least shares a signature; passing
one library's config to the other raises "`quantization_config` is not a `BitsAndBytesConfig`", a
message that never mentions that two identically named classes exist.

**Alternatives considered**: GGUF was rejected because no pre-quantized MiniMax-H3 checkpoint exists.
HQQ is exported by Transformers but not by Diffusers, so it cannot cover `transformer_ref`. Disk
offload of BF16 weights was rejected as it re-reads 134 GiB per denoising step.

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

> **Superseded 2026-09-01.** Retained as the record of what was measured on MiniMax-H3; see
> **Stack decision superseded** above. Not the current stack.

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

## Declared checkpoint metadata (measured, not assumed)

> **Superseded 2026-09-01.** Retained as the record of what was measured on MiniMax-H3; see
> **Stack decision superseded** above. Not the current stack.

**Source**: `spikes/h3_feasibility.py --stage metadata`, run 2026-08-29 on the Windows RTX 5080 host
(Python 3.11.9, torch 2.13.0+cu130, `diffusers==0.40.0`, `transformers==5.16.1`). Read from
`MiniMaxH3ModularPipeline` loaded with `trust_remote_code=False`. Config only — the stage downloaded
`modular_model_index.json` (2.94 kB) and no weights, confirming that `from_pretrained` reads
configuration while `load_components` fetches tensors.

| Field | Declared |
|---|---|
| `fps` | 24 |
| `min_duration` / `max_duration` | 5.0 s / 15.0 s |
| `audio_sampling_rate` | 32000 |
| `audio_channels` | 2 |
| `canvas_multiple` | 32 |
| `vae_spatial_compression_ratio` | 16 |
| `vae_frames_per_chunk` / `vae_latents_per_chunk` | 17 / 5 |
| `vae_latent_channels` / `audio_latent_channels` | 24 / 32 |
| `patch_size` | [1, 2, 2] |
| `text_encoder_layer` | 50 |

**The minimum duration was previously unknown.** The supported range is `[5.0, 15.0]`, not `[0, 15]`:
this checkpoint cannot generate a clip shorter than five seconds. `DurationRange` already carries
`min_seconds` and clamps to it, and no shared code or spec text names a duration number, so this
required no change — which is the profile discipline working as designed rather than a coincidence.

`max_duration` here is the checkpoint's *declaration*. T091 is not closed by it: the ceiling still has
to survive an actual generation, because a declared bound that OOMs at 15 s is not a supported bound.

**Still unmeasured**: peak allocator-reserved bytes, resident host footprint per precision, and
wall-clock per generation. Those need `--stage load` and `--stage generate`, and T040 cannot honestly
declare a profile until they exist.

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
