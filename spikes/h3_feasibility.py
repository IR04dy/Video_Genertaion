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
    python spikes/h3_feasibility.py --stage all --quant quanto-int4 --report spike.json
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

# The weight-bearing components of the ref2va workflow. Every one of these must be
# resident for a memory reading to mean anything; see the completeness check in
# stage_load. `transformer` is absent on purpose — only t2va/fl2va denoise through it.
REF2VA_WEIGHT_COMPONENTS = ("text_encoder", "transformer_ref", "vae", "audio_vae")

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

    if quant.startswith("quanto-"):
        # quanto is the only backend measured to do int4 on the CPU here, and int4 on the
        # CPU is the only configuration that fits: 62.13 GiB and 61.73 GiB of transformers
        # cannot sit on a 16 GB card at any precision, so they must be quantized in host
        # RAM. bitsandbytes crashes natively doing this on CPU (twice, at different
        # weights, at 1.5 GiB RSS) and torchao raises NotImplementedError for every int4
        # CPU packing format. Measured 3.76x compression with a working CPU forward pass.
        #
        # Same class name in both libraries, different KEYWORD: transformers takes
        # `weights`, diffusers takes `weights_dtype`. The BitsAndBytesConfig pair below at
        # least shared a signature; these do not.
        from diffusers import QuantoConfig as DiffusersQuantoConfig
        from transformers import QuantoConfig as TransformersQuantoConfig

        bits = quant.split("-", 1)[1]
        kwargs["quantization_config"] = {
            "text_encoder": TransformersQuantoConfig(weights=bits),
            "transformer_ref": DiffusersQuantoConfig(weights_dtype=bits),
        }
    elif quant != "none":
        # Two different BitsAndBytesConfig classes, deliberately. `text_encoder` is a
        # Transformers model (Qwen3VLForConditionalGeneration) whose from_pretrained
        # type-checks the config against transformers' own class; `transformer_ref` is a
        # Diffusers model checked against Diffusers'. One class for both fails with
        # "Found `quant_method=bitsandbytes` but `quantization_config` is not a
        # `BitsAndBytesConfig`" — same name, different class, and the message never says so.
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
        from transformers import BitsAndBytesConfig as TransformersBnbConfig

        def _config(cls: Any) -> Any:
            if quant == "int4":
                return cls(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            if quant == "int8":
                return cls(load_in_8bit=True)
            raise SystemExit(f"unknown --quant {quant!r}")

        # Per component, not global. `load_components` reads a dict value as
        # {component_name: value} with an optional "default"; a component named in
        # neither is passed no quantization_config at all. The VAEs are absent on
        # purpose: they are convolutional, bitsandbytes quantizes Linear layers only, and
        # quantizing them earns "no linear modules were found in your model" and zero
        # saving. Only the two ~62 GiB transformers are worth quantizing. Names are
        # ref2va's: this workflow denoises through `transformer_ref`, not `transformer`.
        kwargs["quantization_config"] = {
            "text_encoder": _config(TransformersBnbConfig),
            "transformer_ref": _config(DiffusersBnbConfig),
        }

    # Nothing is pinned to the GPU, on purpose. At int4 the text encoder is ~15.5 GiB and
    # transformer_ref ~15.4 GiB, against a 13.5 GiB ceiling on a 16 GB card: neither
    # component fits on the accelerator even quantized, so a GPU-resident load was never
    # possible here. The weights are quantized and held in host RAM (~41 GiB with the
    # unquantized VAEs) and streamed to the card layer by layer by the offload applied
    # after loading. `device_map="auto"` failed precisely because accelerate refused to
    # pretend otherwise.
    #
    # The map is per component because the two libraries disagree about what is legal:
    #
    #   text_encoder     transformers allows an ALL-cpu map — its guard is
    #                    `values != {"cpu"} and "cpu" in values` — but rejects a mixed
    #                    GPU/CPU one under bnb 4-bit.
    #   transformer_ref  diffusers' bnb quantizer rejects `cpu` anywhere in a device_map
    #                    with no all-cpu exception, so it is passed none and loads on the
    #                    CPU by default.
    #   audio_vae        AutoencoderKLMiniMaxH3Audio does not implement
    #                    `_no_split_modules`, so any device_map raises outright.
    #   vae              loads either way; left alone.
    #
    # max_memory is gone with the auto map: it budgeted a GPU/CPU split that no longer
    # happens at load time.
    # bitsandbytes only. Its 4-bit quantizer refuses any device_map that dispatches to the
    # CPU unless the map is exclusively CPU, so text_encoder needs the explicit all-cpu
    # form. quanto has no such validation, and transformers already loads shard by shard
    # onto the CPU when no device_map is given.
    if torch.cuda.is_available() and not quant.startswith("quanto-"):
        kwargs["device_map"] = {"text_encoder": {"": "cpu"}}

    # `modular_model_index.json` hardcodes pretrained_model_name_or_path
    # "MiniMaxAI/MiniMax-H3" for every component. from_pretrained(<local dir>) therefore
    # reads the index from disk and then fetches each component from the HUB anyway: a
    # local --model-path is silently ignored and every byte is downloaded a second time.
    # Overriding the field on the load_components() call redirects all components at the
    # local tree; each spec keeps its own `subfolder`, so they resolve to
    # <root>/text_encoder, <root>/transformer_ref, <root>/vae, and so on.
    local_root = Path(model_path)
    if local_root.is_dir():
        kwargs["pretrained_model_name_or_path"] = str(local_root)

    started = time.monotonic()
    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        model_path, trust_remote_code=False
    )
    report.record(
        "load",
        source="local" if local_root.is_dir() else "hub",
        model_path=str(model_path),
        device_map=kwargs.get("device_map"),
    )
    # workflow= loads only what ref2va uses, which excludes the 61.7 GiB
    # `transformer` that only the fl2va/t2va paths need.
    pipe.load_components(workflow=WORKFLOW, **kwargs)

    # load_components() logs a warning and CARRIES ON when a component fails to load;
    # it does not raise. An incomplete pipeline is therefore indistinguishable from a
    # successful one, and the memory numbers below would describe weights that were
    # never loaded — a 62 GiB text_encoder that silently failed reads as a triumph.
    # Check before recording anything, so a partial load fails loudly instead.
    present = {
        name: getattr(pipe, name, None) is not None
        for name in REF2VA_WEIGHT_COMPONENTS
    }
    report.record("load", components_present=present)
    absent = sorted(name for name, ok in present.items() if not ok)
    if absent:
        raise RuntimeError(
            f"{WORKFLOW} is missing after load_components: {', '.join(absent)}. "
            "Diffusers logged each failure as a warning and continued, so the run "
            "looked healthy; check the log for the per-component traceback. No memory "
            "reading is taken, because it would describe a pipeline that never loaded."
        )

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
    parser.add_argument(
        "--quant",
        choices=("none", "int8", "int4", "quanto-int8", "quanto-int4"),
        default="quanto-int4",
        help="int8/int4 are bitsandbytes; quanto-* is optimum-quanto, the only backend "
        "measured to quantize on the CPU here.",
    )
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
        # Written BEFORE the work starts. Every stage records its findings on
        # completion, which leaves a multi-hour download entirely unrecorded: a
        # process killed below the interpreter (native crash, closed console)
        # writes nothing at all, and an empty report cannot be told apart from a
        # run that never began. This marker makes that distinction visible.
        report.record(stage, started_at=time.strftime("%Y-%m-%dT%H:%M:%S"), ok=None)
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
