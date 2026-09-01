# Image-text-to-video

Local, single-user web app. One or more still images + a motion prompt + a speech script + a
reference voice →
a downloadable MP4 with cloned-voice speech and synchronized lip movement.

Runs entirely on your machine. No hosted API calls during inference.

## Status

**The full P1 workflow runs offline on the stub profile.** No weights, no
accelerator, no network. The real model is not yet wired in.

Design lives in [`specs/001-generate-image-video/`](specs/001-generate-image-video/):
[spec](specs/001-generate-image-video/spec.md) ·
[plan](specs/001-generate-image-video/plan.md) ·
[research](specs/001-generate-image-video/research.md) ·
[data model](specs/001-generate-image-video/data-model.md) ·
[contracts](specs/001-generate-image-video/contracts/) ·
[quickstart](specs/001-generate-image-video/quickstart.md)

[tasks](specs/001-generate-image-video/tasks.md) — 94 tasks, dependency-ordered. **47 done**:
Setup, Foundational, and all of User Story 1 except the real adapter (T040).
Images + voice + prompt → a verified MP4 with a non-silent speech track, a
published bundle, and a manifest that validates against the contract schema.

The dependency gate was retargeted on 2026-09-01 when MiniMax-H3 was abandoned: one indivisible
component is 15.5 GiB at int4, over the 13.5 GiB accelerator ceiling, so no offload strategy could
close the gap. The stack is now **Wan2.2-S2V-14B** for video and lip sync, loaded through DiffSynth
because neither `WanSpeechToVideoPipeline` nor `WanS2VTransformer3DModel` exists in any Diffusers
release. The gate's network half passes against the live repositories; its package half awaits an
install on the production host.

```bash
pip install -r requirements-dev.txt && python -m pytest
```

315 offline tests, no network, no accelerator, no weights, no model stack.
`requirements.txt` adds the model stack — install the platform torch wheel first. Accelerator and stack tests are opt-in:
`-m stack_compatibility`, `-m cuda`, `-m mps`, `-m model_download`.

**Next: prove the Wan2.2-S2V load on the RTX 5080 host.** The working set is 42.6 GiB at BF16, and
DiffSynth's low-VRAM path offloads to disk under an explicit `vram_limit`, so host RAM is no longer the
binding constraint — but nothing is measured until it runs. T040's profile cannot be written honestly
until it does. See [spikes/](spikes/README.md).

## Model

Three roles — video, voice cloning, lip sync — covered by **two models behind one adapter**.
`JointAdapter` describes the interface, not the model, so composing two models inside a single
`generate()` needs no protocol change.

| Role | Model | Notes |
|---|---|---|
| VIDEO + LIP_SYNC | [`Wan-AI/Wan2.2-S2V-14B`](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) | audio conditions the denoiser at 12 of 40 layers, so lip sync is native rather than a post-pass |
| VOICE | **unresolved** — see below | Chatterbox is chosen but cannot yet be installed |

The voice slot is blocked on packaging, not on capability. `chatterbox-tts` hard-pins `torch==2.6.0`,
which on the production host removes the cu130 build and `sm_120` support. The options are a separate
virtualenv invoked as a subprocess, or XTTS-v2 via `coqui-tts` under a non-commercial licence. See
`requirements.txt` and research.md.

| | |
|---|---|
| Frame rate | 16 fps |
| Frame count | must be 4n+1 — 81 frames is 5.0 s |
| Duration | multi-clip; ceiling measured, not assumed |
| Audio | conditioning resampled to 16 kHz; delivery keeps the TTS rate |
| References | 1 image + 1 audio timbre anchor. Video references rejected. |

Neither repository declares `auto_map`, so `trust_remote_code` stays false. Wan ships its T5 encoder
and VAE as `.pth` pickles, which is a separate execution vector — closed by `torch.load`'s
`weights_only=True` default and asserted by the stack gate.

Other models can be added later as reviewed profiles.

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX 5080, 16 GB |
| RAM | 64 GB |
| CPU | Intel Core Ultra 9 285K |
| OS | Windows 11 (production), macOS 13+ (control-path dev) |
| Stack | Python 3.11, PyTorch 2.13, CUDA 13.0, driver 580.88+ |

Intended to run under DiffSynth's disk-offload path with an explicit `vram_limit` — **not yet
demonstrated on this hardware**; that is what the feasibility spike measures. **Inference time is
unbounded — a run may take hours.** No SLA, no timeout.

## Key rules

- Reference audio is a **timbre anchor only**. Never played back. Must say *different words* than the script.
- Per-request voice-cloning consent, bound to request ID + audio hash, reset on every audio change.
- Speech is never trimmed or truncated, and **no request is rejected for script length**.
  Duration is an input, not a measurement: you get a suggested duration from your script, and you
  can override it anywhere in the model's range. Speech is synthesized before video, so the effective
  duration is rounded up to the next legal frame count (4n+1) and the audio tail padded to match.
- Successful outputs are retained in `outputs/<request-id>/` as **unencrypted** files until you delete them.
- No remote code. No license handling — model license compliance is entirely yours.

## License

Not yet selected. Model licenses are the operator's responsibility.
