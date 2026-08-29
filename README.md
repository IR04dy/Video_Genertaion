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

The dependency gate passed on `diffusers==0.40.0` / `transformers==5.16.1`, and
found that the `Ref2VA/` subfolder **requires remote code** — its VAE configs
carry `auto_map` and it names classes no Diffusers release exports. The adapter
will load the repository-root modular pipeline instead.

```bash
pip install -r requirements-dev.txt && python -m pytest
```

315 offline tests, no network, no accelerator, no weights, no model stack.
`requirements.txt` adds the model stack — install the platform torch wheel first. Accelerator and stack tests are opt-in:
`-m stack_compatibility`, `-m cuda`, `-m mps`, `-m model_download`.

**Next: run `spikes/h3_feasibility.py` on the RTX 5080 host.** The `ref2va` working set is 134 GiB at
BF16 against a 64 GB host ceiling, so which quantization is actually resident is unmeasured — and T040's
profile cannot be written honestly until it is. See [spikes/](spikes/README.md).

## Model

Default profile: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3), `ref2va`
workflow. One model generates video, cloned voice, and lip sync jointly — no separate TTS or lip-sync
stage.

Loaded from the repository **root** (`modular_model_index.json`) as a `MiniMaxH3ModularPipeline`, not
from the `Ref2VA/` subfolder — the subfolder's VAE configs declare `auto_map` remote code, which this
project prohibits.

| | |
|---|---|
| Duration | read from `pipeline.min_duration` / `max_duration`; ceiling measured, not assumed |
| Output | 768p short side, 24 FPS, 32 kHz stereo |
| Languages | 11 |
| References | 1 image + 1 audio timbre anchor. Video references rejected. |

Some H3 components are closed-source — see [closed-models.md](specs/001-generate-image-video/closed-models.md).

Other models can be added later as reviewed profiles via Hugging Face URL.

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX 5080, 16 GB |
| RAM | 64 GB |
| CPU | Intel Core Ultra 9 285K |
| OS | Windows 11 (production), macOS 13+ (control-path dev) |
| Stack | Python 3.11, PyTorch 2.13, CUDA 13.0, driver 580.88+ |

Intended to run under quantized layer-wise CPU offload — **not yet demonstrated on this hardware**;
that is what the feasibility spike measures. **Inference time is unbounded — a run may take hours.**
No SLA, no timeout.

## Key rules

- Reference audio is a **timbre anchor only**. Never played back. Must say *different words* than the script.
- Per-request voice-cloning consent, bound to request ID + audio hash, reset on every audio change.
- Speech is never trimmed or truncated, and **no request is rejected for script length**.
  Duration is an input to joint generation, not a measurement of it: you get a suggested
  duration from your script, and you can override it anywhere in the model's range.
- Successful outputs are retained in `outputs/<request-id>/` as **unencrypted** files until you delete them.
- No remote code. No license handling — model license compliance is entirely yours.

## License

Not yet selected. Model licenses are the operator's responsibility.
