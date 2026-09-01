"""Feasibility probe for the Wan2.2-S2V + XTTS-v2 stack (T091, T092).

**Not product code.** Nothing here may be imported by the application. This
script exists to answer questions that cannot be answered by reading, and is
deleted once its answers are recorded in `adapters/wan_s2v.py`'s profile and in
`research.md`.

It answers four, in the order that fails cheapest first:

1. ``voice``    -- does XTTS-v2 speak intelligible Arabic, and at what cost?
2. ``download`` -- can the 42.6 GiB working set be fetched and verified?
3. ``load``     -- does the 14B fit on one 16 GB RTX 5080 under DiffSynth's
                   ``vram_limit`` offload, below the 13.5 GiB reserved ceiling?
4. ``generate`` -- does an end-to-end speech-driven clip come out, and how long
                   does it take?

Every stage writes a verdict to the JSON report. A stage that cannot run records
``"skipped"`` with a reason; only a stage that ran and failed its criterion
records ``"fail"``. The distinction matters: a missing model is not a negative
result about the model.

Run on the Windows RTX 5080 host. It cannot run on macOS -- PyTorch dropped
x86_64 macOS wheels after 2.2.2, and the pinned stack needs torch >= 2.6.

    python spikes/wan_s2v_feasibility.py --stage voice
    python spikes/wan_s2v_feasibility.py --stage all --model-path D:/Yousef/Wan2.2-S2V-14B
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Pass/fail criteria. These are the point of the spike.
# --------------------------------------------------------------------------

GIB = 1024**3

# Mirrors config.DEFAULT_MAX_RESERVED_BYTES. Duplicated rather than imported:
# a spike that imports application code cannot be deleted without touching it.
VRAM_CEILING_BYTES = int(13.5 * GIB)

# Vendor-published values, each of which this spike exists to confirm or refute.
# They are written into a profile only after a real generation agrees with them.
CLAIMED_FRAME_RATE = 16.0
CLAIMED_WIDTH, CLAIMED_HEIGHT = 448, 832
CLAIMED_STEPS = 40
AUDIO_CONDITIONING_RATE = 16000

# XTTS's Arabic character limit, asserted by the stack gate. Longer scripts are
# split here the same way the adapter will have to split them.
ARABIC_CHAR_LIMIT = 166

# Deliberately over the 166-character limit, and verified to be: a script that
# happens to fit exercises none of the splitting this stage exists to test.
# Arabic is compact, so an obvious-looking paragraph can still land under it --
# `_self_check()` below is what keeps that from going unnoticed.
ARABIC_SCRIPT = (
    "مرحبا بكم في هذا الاختبار الصوتي. "
    "نحن نتحقق من جودة النطق العربي ومن سرعة التوليد على هذا الجهاز. "
    "إذا كان الصوت واضحا ومفهوما، فإن هذا النموذج مناسب لهذا المشروع. "
    "نقيس أيضا زمن التوليد لكل مقطع، ونتأكد من أن النص الطويل يقسم إلى مقاطع "
    "قصيرة دون أن يفقد أي جزء منه. "
    "هذه الجملة الأخيرة موجودة لكي يتجاوز النص الحد المسموح به فعليا."
)

MOTION_PROMPT = "a person speaking to the camera, natural head movement, soft indoor lighting"


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class Stage:
    name: str
    verdict: str = "not_run"  # pass | fail | skipped | error
    reason: str = ""
    measurements: dict[str, Any] = field(default_factory=dict)
    seconds: float | None = None


class Report:
    """Accumulates verdicts and writes them out even when a stage explodes.

    Written after every stage rather than once at the end: an OOM during
    `generate` is exactly the result worth keeping, and a report that only
    survives a clean run would lose it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stages: dict[str, Stage] = {}
        self.environment = _environment()

        # Carry forward stages this run does not re-run. Without this, a later
        # `--stage generate` silently erases the `voice` and `download` verdicts
        # that cost real time to produce -- which is exactly what happened before
        # this was added, losing the only record of the Arabic measurement.
        if path.exists():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            for name, recorded in (previous.get("stages") or {}).items():
                self.stages[name] = Stage(
                    name=name,
                    verdict=recorded.get("verdict", "not_run"),
                    reason=recorded.get("reason", ""),
                    measurements=recorded.get("measurements") or {},
                    seconds=recorded.get("seconds"),
                )

    def stage(self, name: str) -> Stage:
        return self.stages.setdefault(name, Stage(name=name))

    def flush(self) -> None:
        payload = {
            "environment": self.environment,
            "criteria": {
                "vram_ceiling_bytes": VRAM_CEILING_BYTES,
                "vram_ceiling_gib": round(VRAM_CEILING_BYTES / GIB, 2),
                "arabic_char_limit": ARABIC_CHAR_LIMIT,
                "claimed_frame_rate": CLAIMED_FRAME_RATE,
                "claimed_resolution": [CLAIMED_WIDTH, CLAIMED_HEIGHT],
                "claimed_steps": CLAIMED_STEPS,
                "audio_conditioning_rate": AUDIO_CONDITIONING_RATE,
            },
            "stages": {
                name: {
                    "verdict": s.verdict,
                    "reason": s.reason,
                    "seconds": s.seconds,
                    "measurements": s.measurements,
                }
                for name, s in self.stages.items()
            },
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = props.name
            env["capability"] = list(torch.cuda.get_device_capability(0))
            env["vram_total_bytes"] = props.total_memory
            env["vram_total_gib"] = round(props.total_memory / GIB, 2)
    except Exception as exc:  # noqa: BLE001 - environment reporting is best effort
        env["torch_error"] = repr(exc)
    return env


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _require_ffmpeg_libraries() -> None:
    """Fail early and legibly if FFmpeg's shared libraries are not loadable.

    coqui-tts refuses to import on torch >= 2.9 unless ``torchcodec`` loads, and
    ``torchcodec`` needs FFmpeg **shared libraries** -- the ``.dll``/``.so`` set,
    not the executable that imageio-ffmpeg ships. Without them the failure is a
    wall of nested ``ctypes`` tracebacks about ``libtorchcodec_core4.dll`` that
    names neither FFmpeg nor the fix.

    PyAV bundles the libraries but is NOT a substitute: delvewheel renames them
    (``avutil-60-cc1777....dll``), so torchcodec cannot resolve them by the names
    it links against. A real FFmpeg install is required.
    """
    try:
        import torchaudio  # noqa: F401
        import torchcodec

        torchcodec._core.get_ffmpeg_library_versions()
    except Exception as exc:  # noqa: BLE001 - the point is a better message
        raise SystemExit(
            "FFmpeg shared libraries are not loadable, so coqui-tts cannot import.\n"
            f"  underlying error: {type(exc).__name__}: {exc}\n\n"
            "  Install them and open a NEW shell so the PATH change applies:\n"
            "      winget install --id Gyan.FFmpeg.Shared\n\n"
            "  imageio-ffmpeg does not satisfy this: it ships only an executable.\n"
            "  PyAV does not either: its bundled libraries are renamed by delvewheel."
        ) from exc


def _log(message: str) -> None:
    _safe_print(f"[spike] {message}")


def _safe_print(text: str) -> None:
    """Print without letting an encoding error destroy a run.

    The Windows console is cp1252 and cannot encode Arabic. A spike that dies
    on its own progress message -- after a multi-hour generation -- would be a
    worse failure than anything it is here to measure.
    """
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _reset_vram() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _host_memory() -> dict[str, Any]:
    """Resident host memory, which is where the weights actually live.

    With CPU offload the GPU peak at load is 0.0 GiB, so the load stage's VRAM
    figure proves nothing on its own. Host RAM is the constraint that replaced
    it: 14B in bfloat16 plus the T5 encoder has to fit beside everything else,
    and `config.DEFAULT_MAX_HOST_RESIDENT_BYTES` is 64 GiB.
    """
    try:
        import psutil
    except ImportError:
        return {}
    process = psutil.Process()
    virtual = psutil.virtual_memory()
    return {
        "host_rss_bytes": process.memory_info().rss,
        "host_rss_gib": round(process.memory_info().rss / GIB, 2),
        "host_available_gib": round(virtual.available / GIB, 2),
        "host_total_gib": round(virtual.total / GIB, 2),
    }


def _vram_peak() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    return {
        "peak_allocated_bytes": allocated,
        "peak_allocated_gib": round(allocated / GIB, 2),
        "peak_reserved_bytes": reserved,
        "peak_reserved_gib": round(reserved / GIB, 2),
        "ceiling_gib": round(VRAM_CEILING_BYTES / GIB, 2),
        "within_ceiling": reserved <= VRAM_CEILING_BYTES,
    }


def split_for_arabic(script: str, limit: int = ARABIC_CHAR_LIMIT) -> list[str]:
    """Split a script into segments under the per-language character limit.

    This is a rehearsal of what T040 must do, not a utility for it. XTTS refuses
    nothing over the limit -- it truncates -- so a script that is merely handed
    over whole loses its tail silently. That silence is the failure this exists
    to make impossible.

    Sentence-boundary first, then a hard character split for any single sentence
    that is itself over the limit, because a sentence longer than the limit still
    has to go somewhere.
    """
    import pysbd

    segmenter = pysbd.Segmenter(language="ar", clean=False)
    segments: list[str] = []
    buffer = ""

    for sentence in (s.strip() for s in segmenter.segment(script) if s.strip()):
        if len(sentence) > limit:
            if buffer:
                segments.append(buffer)
                buffer = ""
            for start in range(0, len(sentence), limit):
                segments.append(sentence[start : start + limit])
            continue
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= limit:
            buffer = candidate
        else:
            segments.append(buffer)
            buffer = sentence

    if buffer:
        segments.append(buffer)
    return segments


def _self_check() -> None:
    """Verify the probe can actually probe, before anything expensive runs.

    Two ways this script could pass while testing nothing: a sample script that
    happens to fit under the character limit, so no splitting occurs; and a
    splitter that stays under the limit by dropping text. Both are silent, and
    both would be discovered only by a human noticing missing speech in the
    output -- so they are checked here instead.
    """
    import re

    assert len(ARABIC_SCRIPT) > ARABIC_CHAR_LIMIT, (
        f"ARABIC_SCRIPT is {len(ARABIC_SCRIPT)} chars, under the {ARABIC_CHAR_LIMIT} "
        "limit, so the splitting path this stage exists to exercise never runs."
    )

    segments = split_for_arabic(ARABIC_SCRIPT)
    assert len(segments) > 1, "splitter returned one segment for an over-limit script"
    assert all(len(s) <= ARABIC_CHAR_LIMIT for s in segments), (
        f"splitter emitted an over-limit segment: {[len(s) for s in segments]}"
    )

    squeeze = lambda text: re.sub(r"\s+", "", text)  # noqa: E731 - local, single use
    assert squeeze("".join(segments)) == squeeze(ARABIC_SCRIPT), (
        "splitter lost or reordered text. XTTS truncates silently, so this would "
        "surface as speech that simply stops -- the exact failure being guarded."
    )


def _trim_wav(path: Path, seconds: float, out_dir: Path) -> Path:
    """Write a shortened copy of the speech. The original is never modified."""
    import soundfile as sf

    data, rate = sf.read(path)
    trimmed = out_dir / f"{path.stem}-{seconds:g}s.wav"
    sf.write(trimmed, data[: int(rate * seconds)], rate)
    return trimmed


def _fit_portrait(image, width: int = CLAIMED_WIDTH, height: int = CLAIMED_HEIGHT):
    """Crop to the target aspect, then scale. Never stretch.

    The target is 448x832 -- a tall 0.538 portrait. A bare `resize` onto that
    from a near-square photo squashes the face horizontally, and a distorted face
    is exactly the wrong input for a model whose whole job is facial motion. It
    would also make the lip-sync verdict meaningless: any badness could be blamed
    on the stretch rather than on the model.

    Centre-crop is a "cover" fit -- the largest box of the right aspect that
    fits. Pass an already-tightly-cropped portrait via `--image` when the subject
    does not sit near the centre.
    """
    from PIL import Image

    target = width / height
    source_width, source_height = image.size
    if source_width / source_height > target:
        crop_width = int(round(source_height * target))
        left = (source_width - crop_width) // 2
        image = image.crop((left, 0, left + crop_width, source_height))
    else:
        crop_height = int(round(source_width / target))
        top = (source_height - crop_height) // 2
        image = image.crop((0, top, source_width, top + crop_height))
    return image.resize((width, height), Image.LANCZOS)


def _wav_facts(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return {
            "sample_rate": rate,
            "channels": handle.getnchannels(),
            "frames": frames,
            "seconds": round(frames / rate, 3) if rate else 0.0,
        }


def frames_for(seconds: float, fps: float = CLAIMED_FRAME_RATE) -> int:
    """Round measured speech up to the next legal Wan frame count.

    Wan's VAE downsamples time by 4, so only ``4n+1`` frame counts are legal.
    Rounding **up** and padding the audio tail is deliberate: rounding down would
    truncate speech, which is the one failure mode that must not be possible.
    """
    raw = max(1, int(seconds * fps + 0.999))
    return ((raw - 1 + 3) // 4) * 4 + 1


# --------------------------------------------------------------------------
# Stage: voice
# --------------------------------------------------------------------------


def stage_voice(report: Report, out_dir: Path, reference_wav: Path | None) -> Path | None:
    """Synthesize Arabic and measure it. Returns the speech path, or None.

    Pass criterion is deliberately weak on quality and strict on mechanics: this
    can prove the audio exists, is the expected length, and is not silence. Only
    a human can judge intelligibility, so the script prints the path and says so
    rather than pretending a metric settles it.
    """
    stage = report.stage("voice")
    started = time.monotonic()
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from TTS.api import TTS

        segments = split_for_arabic(ARABIC_SCRIPT)
        stage.measurements["script_chars"] = len(ARABIC_SCRIPT)
        stage.measurements["segments"] = len(segments)
        stage.measurements["segment_lengths"] = [len(s) for s in segments]
        _log(f"Arabic script split into {len(segments)} segment(s) under {ARABIC_CHAR_LIMIT} chars")

        if any(len(s) > ARABIC_CHAR_LIMIT for s in segments):
            stage.verdict = "fail"
            stage.reason = "splitter produced a segment over the Arabic character limit"
            return None

        if reference_wav is None or not reference_wav.exists():
            stage.verdict = "skipped"
            stage.reason = (
                "no --reference-audio given. XTTS voice cloning needs a speaker "
                "reference; supply a short clean WAV of the target voice."
            )
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reset_vram()
        load_started = time.monotonic()
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        stage.measurements["load_seconds"] = round(time.monotonic() - load_started, 1)

        pieces: list[Any] = []
        rate = 24000
        synth_started = time.monotonic()
        for index, segment in enumerate(segments, start=1):
            _log(f"synthesizing segment {index}/{len(segments)}")
            wav = tts.tts(text=segment, speaker_wav=str(reference_wav), language="ar")
            pieces.append(np.asarray(wav, dtype="float32"))
            rate = tts.synthesizer.output_sample_rate
        stage.measurements["synthesis_seconds"] = round(time.monotonic() - synth_started, 1)

        speech = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        speech_path = out_dir / "speech-ar.wav"
        sf.write(speech_path, speech, rate)

        facts = _wav_facts(speech_path)
        stage.measurements.update(facts)
        stage.measurements.update(_vram_peak())
        peak = float(abs(speech).max()) if speech.size else 0.0
        stage.measurements["peak_amplitude"] = round(peak, 4)
        stage.measurements["realtime_factor"] = (
            round(stage.measurements["synthesis_seconds"] / facts["seconds"], 2)
            if facts["seconds"]
            else None
        )
        stage.measurements["implied_frame_count"] = frames_for(facts["seconds"])

        # Non-silence is asserted because a cloned voice that fails silently
        # produces a valid WAV of nothing, which every other check would pass.
        if peak < 0.01:
            stage.verdict = "fail"
            stage.reason = f"synthesized audio is effectively silent (peak {peak:.4f})"
            return None
        if facts["seconds"] < 1.0:
            stage.verdict = "fail"
            stage.reason = (
                f"synthesized only {facts['seconds']}s "
                f"for a {len(ARABIC_SCRIPT)}-char script"
            )
            return None

        stage.verdict = "pass"
        stage.reason = (
            f"{facts['seconds']}s of Arabic at {facts['sample_rate']} Hz. "
            f"LISTEN TO {speech_path.name} -- intelligibility is a human judgement, "
            "and this stage does not claim to have made it."
        )
        return speech_path

    except Exception as exc:  # noqa: BLE001 - a spike records failures, it does not raise them
        stage.verdict = "error"
        stage.reason = f"{type(exc).__name__}: {exc}"
        stage.measurements["traceback"] = traceback.format_exc(limit=6)
        return None
    finally:
        stage.seconds = round(time.monotonic() - started, 1)
        report.flush()


# --------------------------------------------------------------------------
# Stage: download
# --------------------------------------------------------------------------


def stage_download(report: Report, model_path: Path) -> bool:
    stage = report.stage("download")
    started = time.monotonic()
    try:
        if model_path.exists() and any(model_path.iterdir()):
            total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
            stage.measurements["local_path"] = str(model_path)
            stage.measurements["bytes"] = total
            stage.measurements["gib"] = round(total / GIB, 2)
            stage.measurements["files"] = sum(1 for f in model_path.rglob("*") if f.is_file())
            stage.verdict = "pass"
            stage.reason = "weights already present locally; nothing downloaded"
            return True

        from huggingface_hub import snapshot_download

        _log(f"downloading Wan-AI/Wan2.2-S2V-14B to {model_path} (~42.6 GiB)")
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-S2V-14B",
            local_dir=str(model_path),
            max_workers=4,
        )
        total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        stage.measurements["bytes"] = total
        stage.measurements["gib"] = round(total / GIB, 2)
        stage.verdict = "pass"
        stage.reason = f"downloaded {round(total / GIB, 2)} GiB"
        return True

    except Exception as exc:  # noqa: BLE001
        stage.verdict = "error"
        stage.reason = f"{type(exc).__name__}: {exc}"
        stage.measurements["traceback"] = traceback.format_exc(limit=6)
        return False
    finally:
        stage.seconds = round(time.monotonic() - started, 1)
        report.flush()


# --------------------------------------------------------------------------
# Stage: load
# --------------------------------------------------------------------------


def _quantize_config(method: str | None):
    """Build a DiffSynth quantization config, or None for bfloat16.

    The arithmetic that matters on a 16 GB card, for the 14B DiT alone:

        bfloat16  ~28 GiB  -- cannot be resident; streams every layer, every step
        int8      ~14 GiB  -- still cannot be resident once activations are added
        int4       ~7 GiB  -- resident, leaving room for activations

    Residency is the whole game. The bf16 run measured 100% GPU utilisation at
    60 W: the card was waiting on PCIe, not computing. Halving the bytes halves
    the waiting; it does not stop it. Only a model that fits stops it.
    """
    if method in (None, "none"):
        return None
    from diffsynth.core.quant.config import QuantizeConfig

    return QuantizeConfig(method=method)


def _model_configs(
    model_path: Path,
    quantize_method: str | None = None,
    offload: str = "cpu",
    dit_path: Path | None = None,
):
    """Component list for the S2V pipeline, resolved to explicit local files.

    ``ModelConfig.path`` must name the actual weight files, not the directory
    holding them: DiffSynth only expands ``origin_file_pattern`` on the download
    path, and setting ``path`` at all is what suppresses downloading. Passing a
    directory therefore skips the download *and* hands the loader something it
    cannot read -- so the paths are globbed here instead.
    """
    import torch
    from diffsynth.pipelines.wan_video import ModelConfig

    # A single consolidated file when one is supplied, otherwise the four shards.
    # DiffSynth's DiskMap cannot read a multi-shard checkpoint when a quantize
    # config is set with `offload_device="cuda"` -- it raises "Attempted to
    # access the data pointer on an invalid python storage" -- so a GPU-resident
    # quantized load needs the merged file. Measured, not assumed.
    if dit_path is not None:
        dit_shards = [str(dit_path)]
    else:
        dit_shards = sorted(
            str(p) for p in model_path.glob("diffusion_pytorch_model-*.safetensors")
        )
    if not dit_shards:
        raise SystemExit(f"no DiT weights under {model_path}; is --model-path correct?")

    # Offload placement is the whole experiment, and it must be stated: DiffSynth
    # defaults `offload_device` to None, which puts every parameter on CUDA at
    # load. Measured on this host, that peaks at 41.89 GiB -- which does not fail
    # on Windows, because WDDM silently spills to shared system memory. The load
    # then "succeeds" while running from host RAM over PCIe, which is far worse
    # than an honest OOM. Keeping the weights on the CPU and streaming them per
    # layer is the only configuration that respects the card.
    placement = {
        "offload_device": offload,
        "offload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "onload_dtype": torch.bfloat16,
        "computation_device": "cuda",
        "computation_dtype": torch.bfloat16,
    }
    quantize = _quantize_config(quantize_method)

    # Only the DiT is quantized. It is ~28 GiB of the working set and the thing
    # streamed on every step; the VAE and audio encoder are small enough that
    # quantizing them trades accuracy for nothing. The T5 encoder runs once per
    # generation, so its transfer cost is amortised to nothing as well.
    return [
        ModelConfig(path=dit_shards, quantize=quantize, **placement),
        ModelConfig(path=str(model_path / "models_t5_umt5-xxl-enc-bf16.pth"), **placement),
        ModelConfig(path=str(model_path / "Wan2.1_VAE.pth"), **placement),
        # The weight FILE, not the directory: DiffSynth torch.loads this path
        # directly. The directory form is only correct for the processor below.
        # The repository ships these weights three times (safetensors, .bin and
        # flax msgpack, ~1.26 GiB each); only the safetensors copy is loaded.
        ModelConfig(
            path=str(model_path / "wav2vec2-large-xlsr-53-english" / "model.safetensors"),
            **placement,
        ),
    ]


def _audio_processor_config(model_path: Path):
    """The Wav2Vec2 processor, which the pipeline takes separately from the models."""
    from diffsynth.pipelines.wan_video import ModelConfig

    return ModelConfig(path=str(model_path / "wav2vec2-large-xlsr-53-english"))


def stage_load(
    report: Report,
    model_path: Path,
    vram_limit: float,
    quantize_method: str | None = None,
    offload: str = "cpu",
    dit_path: Path | None = None,
):
    """Load the 14B under DiffSynth offload and record what it costs.

    ``vram_limit`` is the whole experiment. DiffSynth keeps parameters in host
    RAM and streams them per layer to stay under it, which is the only reason a
    14B is plausible on a 16 GB card at all.
    """
    stage = report.stage("load")
    started = time.monotonic()
    try:
        import torch
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        _reset_vram()
        _log(
            f"loading Wan2.2-S2V-14B with vram_limit={vram_limit} GB, "
            f"quantize={quantize_method or 'none (bfloat16)'}, offload={offload}"
        )
        stage.measurements["quantize_method"] = quantize_method or "none"
        stage.measurements["offload_device"] = offload
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=_model_configs(model_path, quantize_method, offload, dit_path),
            audio_processor_config=_audio_processor_config(model_path),
            vram_limit=vram_limit,
        )
        stage.measurements["vram_limit_gb"] = vram_limit
        stage.measurements.update(_vram_peak())
        stage.measurements.update(_host_memory())

        within = stage.measurements.get("within_ceiling")
        if within is False:
            stage.verdict = "fail"
            stage.reason = (
                f"peak reserved {stage.measurements['peak_reserved_gib']} GiB exceeds the "
                f"{round(VRAM_CEILING_BYTES / GIB, 2)} GiB ceiling at load, before generating"
            )
        else:
            stage.verdict = "pass"
            stage.reason = (
                f"loaded at {stage.measurements.get('peak_reserved_gib')} GiB reserved, "
                f"{stage.measurements.get('host_rss_gib')} GiB host RSS. "
                "A near-zero VRAM figure here is expected under CPU offload and "
                "proves little -- the ceiling is decided in `generate`."
            )
        return pipe

    except Exception as exc:  # noqa: BLE001
        stage.verdict = "error"
        stage.reason = f"{type(exc).__name__}: {exc}"
        stage.measurements["traceback"] = traceback.format_exc(limit=8)
        return None
    finally:
        stage.seconds = round(time.monotonic() - started, 1)
        report.flush()


# --------------------------------------------------------------------------
# Stage: generate
# --------------------------------------------------------------------------


def stage_generate(
    report: Report,
    pipe,
    speech_path: Path,
    image_path: Path,
    out_dir: Path,
    steps: int,
    max_speech_seconds: float | None = None,
    width: int = CLAIMED_WIDTH,
    height: int = CLAIMED_HEIGHT,
) -> None:
    """One speech-driven generation, timed, with peak VRAM recorded.

    The frame count comes from the measured speech duration rather than from a
    preset. That is the whole architectural claim of the voice-first design, so
    the spike exercises it rather than passing a round number.
    """
    stage = report.stage("generate")
    started = time.monotonic()
    try:
        import librosa
        import torch
        # `save_video_with_audio`, not `save_video`: lip sync cannot be judged
        # from a silent clip, and muxing here avoids a separate export step whose
        # own timebase could be blamed for a mismatch.
        from diffsynth.utils.data import save_video_with_audio
        from PIL import Image

        facts = _wav_facts(speech_path)
        if max_speech_seconds is not None and facts["seconds"] > max_speech_seconds:
            speech_path = _trim_wav(speech_path, max_speech_seconds, out_dir)
            facts = _wav_facts(speech_path)
            stage.measurements["trimmed_to_seconds"] = max_speech_seconds
        num_frames = frames_for(facts["seconds"])
        stage.measurements["speech_seconds"] = facts["seconds"]
        stage.measurements["num_frames"] = num_frames
        stage.measurements["frame_grid_ok"] = (num_frames - 1) % 4 == 0
        stage.measurements["video_seconds"] = round(num_frames / CLAIMED_FRAME_RATE, 3)
        stage.measurements["steps"] = steps

        audio, _ = librosa.load(str(speech_path), sr=AUDIO_CONDITIONING_RATE, mono=True)
        image = _fit_portrait(Image.open(image_path).convert("RGB"), width, height)
        stage.measurements["resolution"] = [width, height]
        # Activation memory, not weight memory, is what fills this card: int8
        # weights changed peak VRAM by 0.04 GiB. Recording the pixel budget makes
        # the two runs comparable on the term that actually moves.
        stage.measurements["activation_budget_px_frames"] = width * height * num_frames

        _reset_vram()
        # The actual width/height, not the claimed constants: a progress line that
        # reports a resolution the run is not using makes every measurement below
        # it unreadable.
        _log(f"generating {num_frames} frames at {width}x{height}, {steps} steps")
        video = pipe(
            prompt=MOTION_PROMPT,
            input_image=image,
            input_audio=audio,
            audio_sample_rate=AUDIO_CONDITIONING_RATE,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            seed=0,
        )
        stage.measurements.update(_vram_peak())
        stage.measurements.update(_host_memory())

        out_path = out_dir / "wan-s2v-probe.mp4"
        save_video_with_audio(
            video,
            str(out_path),
            str(speech_path),
            fps=int(CLAIMED_FRAME_RATE),
            quality=5,
        )
        stage.measurements["output"] = str(out_path)
        stage.measurements["output_bytes"] = out_path.stat().st_size
        stage.measurements["frames_returned"] = len(video)

        elapsed = time.monotonic() - started
        stage.measurements["seconds_per_frame"] = round(elapsed / num_frames, 2)
        stage.measurements["realtime_factor"] = round(
            elapsed / (num_frames / CLAIMED_FRAME_RATE), 1
        )

        if stage.measurements.get("within_ceiling") is False:
            stage.verdict = "fail"
            stage.reason = (
                f"peak reserved {stage.measurements['peak_reserved_gib']} GiB exceeds the "
                f"{round(VRAM_CEILING_BYTES / GIB, 2)} GiB ceiling during generation"
            )
        elif len(video) != num_frames:
            stage.verdict = "fail"
            stage.reason = f"asked for {num_frames} frames, got {len(video)}"
        else:
            stage.verdict = "pass"
            stage.reason = (
                f"{num_frames} frames in {round(elapsed / 60, 1)} min at "
                f"{stage.measurements.get('peak_reserved_gib')} GiB reserved. "
                f"WATCH {out_path.name} -- lip sync is a human judgement."
            )

    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[name-defined]
        stage.verdict = "fail"
        stage.reason = f"CUDA OOM during generation: {exc}"
        stage.measurements.update(_vram_peak())
    except Exception as exc:  # noqa: BLE001
        stage.verdict = "error"
        stage.reason = f"{type(exc).__name__}: {exc}"
        stage.measurements["traceback"] = traceback.format_exc(limit=8)
    finally:
        stage.seconds = round(time.monotonic() - started, 1)
        report.flush()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=("voice", "download", "load", "generate", "all"),
        default="voice",
        help="which probe to run. 'voice' needs no weights and fails cheapest.",
    )
    parser.add_argument("--model-path", type=Path, default=Path("D:/Yousef/Wan2.2-S2V-14B"))
    parser.add_argument("--reference-audio", type=Path, default=None, help="speaker WAV to clone")
    parser.add_argument("--image", type=Path, default=None, help="conditioning portrait")
    parser.add_argument(
        "--speech", type=Path, default=None, help="reuse a speech WAV from a prior run"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("spike-out"))
    parser.add_argument("--report", type=Path, default=Path("spike-report.json"))
    parser.add_argument(
        "--vram-limit",
        type=float,
        default=12.0,
        help="GB handed to DiffSynth's offload manager; below the 13.5 GiB ceiling by design",
    )
    parser.add_argument("--steps", type=int, default=CLAIMED_STEPS)
    parser.add_argument(
        "--quantize",
        default="none",
        help=(
            "DiffSynth quantization method for the DiT, e.g. torchao_int8_w8a16, "
            "torchao_int8_w8a8, torchao_int4_w4a16. 'none' keeps bfloat16."
        ),
    )
    parser.add_argument(
        "--dit-path",
        type=Path,
        default=None,
        help=(
            "single consolidated DiT safetensors file. Required for a quantized "
            "GPU-resident load: DiffSynth cannot read sharded weights on that path."
        ),
    )
    parser.add_argument("--width", type=int, default=CLAIMED_WIDTH)
    parser.add_argument(
        "--height",
        type=int,
        default=CLAIMED_HEIGHT,
        help=(
            "activation memory scales with width x height x frames, which is "
            "what actually fills a 16 GB card here -- int8 weights moved peak "
            "VRAM by 0.04 GiB, resolution is the term that moves it."
        ),
    )
    parser.add_argument(
        "--offload",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "where idle weights live. 'cpu' streams them per layer over PCIe; "
            "'cuda' keeps them resident, which only fits once quantized enough."
        ),
    )
    parser.add_argument(
        "--max-speech-seconds",
        type=float,
        default=None,
        help=(
            "trim the driving speech before generating. Cost scales with frame "
            "count, so the first run should prove the path at ~5s (81 frames) "
            "rather than commit hours to full length."
        ),
    )
    args = parser.parse_args(argv)

    _self_check()
    _require_ffmpeg_libraries()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = Report(args.report)
    report.flush()

    wants = ("voice", "download", "load", "generate") if args.stage == "all" else (args.stage,)

    # `generate` cannot run alone: the pipeline object lives in this process and
    # does not survive between invocations, so asking for generate always means
    # load-then-generate. Implying it here rather than erroring keeps the obvious
    # command working instead of teaching a workaround.
    if "generate" in wants and "load" not in wants:
        wants = ("load", *wants)
    speech_path = args.speech

    if "voice" in wants:
        produced = stage_voice(report, args.out_dir, args.reference_audio)
        speech_path = speech_path or produced

    if "download" in wants and not stage_download(report, args.model_path):
        _log("download stage did not succeed; skipping load and generate")
        wants = tuple(w for w in wants if w not in ("load", "generate"))

    pipe = None
    if "load" in wants:
        pipe = stage_load(
            report,
            args.model_path,
            args.vram_limit,
            None if args.quantize == "none" else args.quantize,
            args.offload,
            args.dit_path,
        )

    if "generate" in wants:
        stage = report.stage("generate")
        if pipe is None:
            stage.verdict = "skipped"
            stage.reason = "pipeline did not load"
        elif speech_path is None or not Path(speech_path).exists():
            stage.verdict = "skipped"
            stage.reason = (
                "no speech WAV; run --stage voice with --reference-audio, or pass --speech"
            )
        elif args.image is None or not args.image.exists():
            stage.verdict = "skipped"
            stage.reason = "no --image given; S2V needs a conditioning portrait"
        else:
            stage_generate(
                report,
                pipe,
                Path(speech_path),
                args.image,
                args.out_dir,
                args.steps,
                args.max_speech_seconds,
                args.width,
                args.height,
            )
        report.flush()

    print("\n" + "=" * 68)
    for name, stage in report.stages.items():
        mark = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP", "error": "ERR "}.get(
            stage.verdict, "??? "
        )
        _safe_print(f"  [{mark}] {name:<9} {stage.reason}")
    _safe_print("=" * 68)
    _safe_print(f"  report: {args.report}")

    failed = any(s.verdict in ("fail", "error") for s in report.stages.values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
