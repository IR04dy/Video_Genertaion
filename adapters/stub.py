"""Deterministic offline stub adapter (T039).

The reference implementation of `JointAdapter`, and the thing every offline test
runs against. It writes real PNG frames and a real WAV using only the standard
library, so the offline suite needs no Pillow, no NumPy, no torch, and no weights.

Its profile values are **arbitrary and deliberately unlike the production
profile's** — 12 fps, 96x64, 8 kHz mono, a made-up language. If a test passes
against the real model but fails here, something measured has leaked out of a
profile into shared code, which is exactly what
`tests/unit/test_profile_agnostic.py` exists to catch.
"""

from __future__ import annotations

import hashlib
import math
import struct
import wave
import zlib
from pathlib import Path

from adapters.base import (
    CancellationToken,
    GenerationArtifacts,
    GenerationInputs,
    ProgressCallback,
    emit,
)
from domain import (
    AudioOutput,
    DeviceKind,
    DurationRange,
    MemoryProfile,
    ModelProfile,
    ModelRole,
    OffloadMode,
    ReferenceLimits,
    Resolution,
)

STUB_PROFILE = ModelProfile(
    adapter_key="stub",
    profile_id="stub@1",
    roles={ModelRole.VIDEO, ModelRole.VOICE, ModelRole.LIP_SYNC},
    native_capabilities={ModelRole.VOICE, ModelRole.LIP_SYNC},
    pipeline_class="StubJointPipeline",
    supported_devices={DeviceKind.CPU},
    dtype_policy={DeviceKind.CPU: {"preferred": "float32", "allowed": ["float32"]}},
    memory_profiles=[
        MemoryProfile(
            offload_mode=OffloadMode.NONE,
            quantization=None,
            expected_peak_reserved_bytes=0,
            expected_host_resident_bytes=64 * 1024 * 1024,
        )
    ],
    duration_range_seconds=DurationRange(min_seconds=1.0, max_seconds=6.0, default_seconds=2.0),
    frame_rate=12.0,
    resolutions=[Resolution(width=96, height=64)],
    audio_output=AudioOutput(sample_rate=8000, channels=1),
    dialogue_languages=["Testish", "Fixtureish"],
    speaking_rates={"Testish": 10.0},
    reference_limits=ReferenceLimits(
        accepted={"image": 3, "audio": 1},
        rejected={"video": "video references are refused on token cost"},
        audio_clip_seconds=DurationRange(min_seconds=1.0, max_seconds=5.0, default_seconds=2.0),
    ),
    prompt_capacity_tokens=128,
    dialogue_tag_form="<d>[{language}]{text}</d>",
    input_contract={"images": "one or more", "audio": "exactly one timbre anchor"},
    output_contract={"video": True, "audio": True, "joint": True},
    weight_policy={"extensions": [".safetensors"], "required": []},
    resource_profile=MemoryProfile(
        offload_mode=OffloadMode.NONE,
        quantization=None,
        expected_peak_reserved_bytes=0,
        expected_host_resident_bytes=64 * 1024 * 1024,
    ),
)


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Minimal solid-colour PNG. Real format, no imaging dependency."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


class StubAdapter:
    """Deterministic joint adapter. Same inputs, byte-identical outputs."""

    def __init__(self, profile: ModelProfile | None = None) -> None:
        self._profile = profile or STUB_PROFILE
        self._loaded = False

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    def load(
        self,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        progress: ProgressCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> None:
        if cancel:
            cancel.raise_if_cancelled(stage="load_model")
        emit(progress, "load_model", 1.0, "stub weights ready")
        self._loaded = True

    def generate(
        self,
        inputs: GenerationInputs,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> GenerationArtifacts:
        if not self._loaded:
            self.load(progress=progress, cancel=cancel)

        profile = self._profile
        resolution = profile.resolutions[0]
        frame_rate = profile.frame_rate
        sample_rate = profile.audio_output.sample_rate
        channels = profile.audio_output.channels

        frame_count = inputs.duration.effective_num_frames
        seconds = inputs.duration.effective_duration_seconds

        # Scoped by request id, not just by the staging directory: two requests
        # given the same working directory must not overwrite each other's frames.
        workdir = Path(inputs.audio_path).parent / f"stub-{inputs.request_id}"
        frames_dir = workdir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Seed-derived colour ramp: deterministic, and visibly different per seed.
        seed_bytes = hashlib.sha256(str(inputs.seed).encode()).digest()

        for index in range(frame_count):
            if cancel and index % 8 == 0:
                cancel.raise_if_cancelled(stage="generate")
            shade = (index * 7 + seed_bytes[0]) % 256
            (frames_dir / f"{index:06d}.png").write_bytes(
                _png(resolution.width, resolution.height, (shade, seed_bytes[1], seed_bytes[2]))
            )
            emit(progress, "generate", (index + 1) / frame_count, f"frame {index + 1}")

        # A non-silent tone, so the export stage's non-silence check is meaningful.
        audio_out = workdir / "audio.wav"
        total_samples = int(seconds * sample_rate)
        with wave.open(str(audio_out), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            frames = bytearray()
            for n in range(total_samples):
                value = int(12000 * math.sin(2 * math.pi * 220 * n / sample_rate))
                frames += struct.pack("<h", value) * channels
            handle.writeframes(bytes(frames))

        emit(progress, "generate", 1.0, "stub generation complete")

        return GenerationArtifacts(
            frames_path=frames_dir,
            audio_path=audio_out,
            frame_rate=frame_rate,
            audio_sample_rate=sample_rate,
            audio_channels=channels,
            width=resolution.width,
            height=resolution.height,
            frame_count=frame_count,
        )

    def unload(self) -> None:
        self._loaded = False
