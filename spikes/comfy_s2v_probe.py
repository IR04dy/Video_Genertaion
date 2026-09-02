"""Drive ComfyUI's native Wan S2V nodes over the HTTP API (T091 continuation).

**Not product code.** A disposable probe, like `wan_s2v_feasibility.py`, and
deleted once its answers reach `research.md`.

Why this exists: `diffsynth==2.1.5` reached generation but never completed a
denoising step on this host, in six configurations. Profiling attributed that to
its offload loop -- one CPU core saturated, GPU starved at 61 W -- and not to the
weights or the hardware, which stream at a measured 36.8 GB/s pinned. ComfyUI is
the alternative runtime for the same model, and it announces
"async weight offloading with 2 streams" and "Enabled pinned memory" at startup,
which is precisely the mechanism DiffSynth was missing.

The API, not the GUI, deliberately: it is reproducible, it is scriptable, and it
is how an adapter would drive ComfyUI as a subprocess.

    python spikes/comfy_s2v_probe.py --frames 33 --width 320 --height 576 --steps 8
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SERVER = "http://127.0.0.1:8188"

# Filenames as ComfyUI sees them, relative to its own models/ subdirectories.
DIT = "wan2.2_s2v_14B_fp8_scaled.safetensors"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"
AUDIO_ENCODER = "wav2vec2_large_english_fp16.safetensors"

POSITIVE = "a person speaking to the camera, natural head movement, soft indoor lighting"
NEGATIVE = "blurry, distorted face, static, watermark, text"


def build_workflow(
    *, audio: str, image: str, width: int, height: int, frames: int, steps: int, seed: int
) -> dict:
    """The API-format graph. Node ids are strings; links are [node_id, slot]."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": DIT, "weight_dtype": "default"}},
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "wan", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {
            "class_type": "AudioEncoderLoader",
            "inputs": {"audio_encoder_name": AUDIO_ENCODER},
        },
        "5": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "6": {"class_type": "LoadImage", "inputs": {"image": image}},
        "7": {
            "class_type": "AudioEncoderEncode",
            "inputs": {"audio_encoder": ["4", 0], "audio": ["5", 0]},
        },
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": POSITIVE}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": NEGATIVE}},
        "10": {
            "class_type": "WanSoundImageToVideo",
            "inputs": {
                "positive": ["8", 0],
                "negative": ["9", 0],
                "vae": ["3", 0],
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
                "audio_encoder_output": ["7", 0],
                "ref_image": ["6", 0],
            },
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["10", 2],
                "seed": seed,
                "steps": steps,
                "cfg": 6.0,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["12", 0], "fps": 16.0, "audio": ["5", 0]},
        },
        "14": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["13", 0], "filename_prefix": "wan-s2v-probe", "format": "auto",
                       "codec": "auto"},
        },
    }


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{SERVER}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{SERVER}{path}") as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default="speech-ar-2s.wav")
    parser.add_argument("--image", default="portrait.jpg")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--frames", type=int, default=33, help="must be 4n+1")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("comfy-report.json"))
    args = parser.parse_args()

    if (args.frames - 1) % 4 != 0:
        raise SystemExit(f"--frames must be 4n+1; {args.frames} is not")

    client_id = str(uuid.uuid4())
    workflow = build_workflow(
        audio=args.audio,
        image=args.image,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        seed=args.seed,
    )

    print(f"[probe] {args.frames} frames at {args.width}x{args.height}, {args.steps} steps",
          flush=True)
    started = time.time()
    try:
        queued = post("/prompt", {"prompt": workflow, "client_id": client_id})
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        raise SystemExit(f"ComfyUI rejected the workflow:\n{body}") from error

    prompt_id = queued["prompt_id"]
    print(f"[probe] queued {prompt_id}", flush=True)

    # Poll rather than open a websocket: fewer moving parts, and the only thing
    # needed is the terminal state plus wall-clock.
    while True:
        history = get(f"/history/{prompt_id}")
        if prompt_id in history:
            break
        time.sleep(5)

    elapsed = time.time() - started
    entry = history[prompt_id]
    status = entry.get("status", {})
    outputs = entry.get("outputs", {})

    files = [
        f"{item.get('subfolder','')}/{item.get('filename','')}".lstrip("/")
        for node in outputs.values()
        for key in ("images", "videos", "gifs")
        for item in node.get(key, [])
    ]

    report = {
        "prompt_id": prompt_id,
        "status": status.get("status_str"),
        "completed": status.get("completed"),
        "seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / args.steps, 1),
        "config": {
            "frames": args.frames,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "dit": DIT,
        },
        "outputs": files,
        "messages": status.get("messages", [])[-6:],
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[probe] {status.get('status_str')} in {elapsed / 60:.1f} min "
          f"({elapsed / args.steps:.1f}s/step)", flush=True)
    for name in files:
        print(f"[probe] output: {name}", flush=True)
    return 0 if status.get("status_str") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
