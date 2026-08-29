"""MiniMax-H3 Ref2VA feasibility spike. Disposable; not product code.

Answers three questions that reading cannot answer, in increasing order of cost,
recording what it learned even when a later stage fails:

  metadata  Do the pinned wheels load the checkpoint's config with
            trust_remote_code=False, and what does the checkpoint declare its
            duration range, frame rate and sample rate to be? No weights.
  load      Do the ref2va components fit in 16 GB VRAM + 64 GB RAM under
            quantization and offload? Reports peak reserved bytes and host RSS.
  generate  Does one generation at the checkpoint's own minimum duration
            complete, and how long does it take?

Every stage writes its measurements into the report before the next begins, so a
crash in `generate` still leaves the `load` numbers on disk. That is the entire
point: a failed run must still be an informative run.

Usage:
    python spikes/h3_feasibility.py --stage metadata
    python spikes/h3_feasibility.py --stage all --quant int4 --report spike.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ID = "MiniMaxAI/MiniMax-H3"
WORKFLOW = "ref2va"

# Measured against these, per plan.md. Both are ceilings, not targets.
VRAM_CEILING_BYTES = int(13.5 * 2**30)
HOST_CEILING_BYTES = 64 * 2**30


@dataclass
class Report:
    """Accumulates findings. Flushed to disk after every stage."""

    path: Path | None
    host: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)

    def record(self, stage: str, **payload: Any) -> None:
        self.stages.setdefault(stage, {}).update(payload)
        self.flush()

    def fail(self, stage: str, exc: BaseException) -> None:
        self.record(
            stage,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.write_text(
            json.dumps({"host": self.host, "stages": self.stages}, indent=2)
        )


def gib(value: float | None) -> float | None:
    return None if value is None else round(value / 2**30, 3)


def sample_memory() -> dict[str, Any]:
    """Both ceilings in one reading: accelerator reserved, and host resident."""
    import torch

    snapshot: dict[str, Any] = {}

    if torch.cuda.is_available():
        snapshot["device_name"] = torch.cuda.get_device_name(0)
        snapshot["allocated_gib"] = gib(torch.cuda.memory_allocated())
        snapshot["reserved_gib"] = gib(torch.cuda.memory_reserved())
        snapshot["peak_reserved_gib"] = gib(torch.cuda.max_memory_reserved())
        snapshot["peak_allocated_gib"] = gib(torch.cuda.max_memory_allocated())
        free, total = torch.cuda.mem_get_info()
        snapshot["free_gib"] = gib(free)
        snapshot["total_gib"] = gib(total)
        peak = torch.cuda.max_memory_reserved()
        snapshot["within_vram_ceiling"] = peak <= VRAM_CEILING_BYTES
    else:
        snapshot["device_name"] = None
        snapshot["unavailable_reason"] = "torch.cuda.is_available() is False"

    try:
        import psutil

        rss = psutil.Process().memory_info().rss
        snapshot["host_rss_gib"] = gib(rss)
        snapshot["host_available_gib"] = gib(psutil.virtual_memory().available)
        snapshot["within_host_ceiling"] = rss <= HOST_CEILING_BYTES
    except Exception as exc:  # psutil is optional for the spike
        snapshot["host_rss_gib"] = None
        snapshot["host_error"] = str(exc)

    return snapshot


def stage_metadata(report: Report, model_path: str) -> None:
    """Config-only. Proves the stack and harvests the checkpoint's declared limits."""
    import diffusers
    import torch  # noqa: F401  (import order matters for diffusers)
    import transformers
    from diffusers import MiniMaxH3ModularPipeline

    report.record(
        "metadata",
        diffusers_version=diffusers.__version__,
        transformers_version=transformers.__version__,
    )

    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        model_path, trust_remote_code=False
    )

    # These are the fields ModelProfile must carry. Read them; never hardcode them.
    # Anything absent is reported as None rather than guessed.
    declared = {}
    for name in (
        "fps",
        "min_duration",
        "max_duration",
        "audio_sampling_rate",
        "audio_channels",
        "canvas_multiple",
        "vae_spatial_compression_ratio",
        "vae_frames_per_chunk",
        "vae_latents_per_chunk",
        "vae_latent_channels",
        "audio_latent_channels",
        "patch_size",
        "text_encoder_layer",
    ):
        try:
            declared[name] = getattr(pipe, name)
        except Exception as exc:
            declared[name] = f"<unreadable: {type(exc).__name__}>"

    report.record("metadata", ok=True, declared_profile=declared)
    print(json.dumps(declared, indent=2, default=str))
    return pipe


def stage_load(report: Report, model_path: str, quant: str) -> Any:
    """Load only the ref2va workflow's components, under quantization + offload."""
    import torch
    from diffusers import MiniMaxH3ModularPipeline

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
    if quant != "none":
        from diffusers import BitsAndBytesConfig

        if quant == "int4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif quant == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise SystemExit(f"unknown --quant {quant!r}")

    started = time.monotonic()
    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        model_path, trust_remote_code=False
    )
    # workflow= loads only what ref2va uses, which excludes the 61.7 GiB
    # `transformer` that only the fl2va/t2va paths need.
    pipe.load_components(workflow=WORKFLOW, **kwargs)

    report.record(
        "load",
        ok=True,
        quant=quant,
        workflow=WORKFLOW,
        seconds=round(time.monotonic() - started, 1),
        memory=sample_memory(),
    )

    # Offload is applied after loading so the load-time peak is measured honestly.
    for method in ("enable_sequential_cpu_offload", "enable_model_cpu_offload"):
        if hasattr(pipe, method):
            try:
                getattr(pipe, method)()
                report.record("load", offload_applied=method)
                break
            except Exception as exc:
                report.record("load", offload_error=f"{method}: {exc}")

    return pipe


def stage_generate(
    report: Report, pipe: Any, image: Path | None, audio: Path | None
) -> None:
    """One generation at the checkpoint's own minimum duration, smallest canvas."""
    import torch
    from diffusers.modular_pipelines import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
    )

    if image is None or audio is None:
        report.record(
            "generate",
            ok=False,
            error="ref2va needs --image and --audio; skipped",
        )
        return

    duration = float(getattr(pipe, "min_duration", 4.0))
    references = [
        MiniMaxH3ImageReference(str(image)),
        MiniMaxH3AudioReference(str(audio)),
    ]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.monotonic()
    result = pipe(
        prompt="A person speaking to camera. <d>[English]This is a feasibility test.</d>",
        references=references,
        duration=duration,
        num_inference_steps=8,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    elapsed = time.monotonic() - started

    report.record(
        "generate",
        ok=True,
        duration_seconds=duration,
        # Recorded as an observation, never as a budget. Nothing asserts on it.
        wall_clock_seconds=round(elapsed, 1),
        memory=sample_memory(),
        output_type=type(result).__name__,
    )
    print(f"generated {duration}s in {elapsed / 60:.1f} min")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("metadata", "load", "generate", "all"),
        default="metadata",
    )
    parser.add_argument("--model-path", default=REPO_ID)
    parser.add_argument("--quant", choices=("none", "int8", "int4"), default="int4")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = Report(path=args.report)
    report.host = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    try:
        import torch

        report.host["torch"] = torch.__version__
        report.host["cuda_available"] = torch.cuda.is_available()
    except Exception as exc:
        report.host["torch_import_error"] = str(exc)
        report.flush()
        print(f"cannot import torch: {exc}", file=sys.stderr)
        return 1
    report.flush()

    wanted = ("metadata", "load", "generate") if args.stage == "all" else (args.stage,)
    pipe = None

    for stage in wanted:
        print(f"--- {stage} ---", flush=True)
        try:
            if stage == "metadata":
                pipe = stage_metadata(report, args.model_path)
            elif stage == "load":
                pipe = stage_load(report, args.model_path, args.quant)
            elif stage == "generate":
                if pipe is None:
                    pipe = stage_load(report, args.model_path, args.quant)
                stage_generate(report, pipe, args.image, args.audio)
        except BaseException as exc:  # including OOM, which is what we are testing
            report.fail(stage, exc)
            print(f"{stage} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            if isinstance(exc, ImportError):
                print(
                    "the model stack is not installed. Install the platform torch "
                    "wheel first, then: pip install -r requirements.txt",
                    file=sys.stderr,
                )
            if args.report:
                print(f"partial findings written to {args.report}", file=sys.stderr)
            return 1

    if args.report:
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
