"""Shared fixtures (T022).

The important one is `fixture_profile_kwargs`: a profile whose every measured
value differs from both MiniMax-H3's and the stub's. `test_profile_agnostic.py`
re-runs the suite against it, so any duration, frame rate, resolution, sample
rate, language, reference limit, or token capacity that has leaked out of a
profile into shared code shows up as a failure there rather than in production.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.stub import StubAdapter, _png
from domain import (
    AudioOutput,
    ConsentAttestation,
    DeviceKind,
    DurationRange,
    MemoryProfile,
    ModelProfile,
    ModelRole,
    OffloadMode,
    ReferenceLimits,
    Resolution,
)


@pytest.fixture
def sample_profile_kwargs() -> dict:
    """A minimal valid profile. Values are arbitrary; assert on none of them."""
    return {
        "adapter_key": "sample",
        "profile_id": "sample@1",
        "roles": {ModelRole.VIDEO, ModelRole.VOICE, ModelRole.LIP_SYNC},
        "native_capabilities": {ModelRole.VOICE, ModelRole.LIP_SYNC},
        "pipeline_class": "SamplePipeline",
        "supported_devices": {DeviceKind.CPU},
        "duration_range_seconds": DurationRange(
            min_seconds=2.0, max_seconds=8.0, default_seconds=4.0
        ),
        "frame_rate": 10.0,
        "resolutions": [Resolution(width=64, height=64)],
        "audio_output": AudioOutput(sample_rate=16000, channels=1),
        "dialogue_languages": ["Sampleish"],
        "speaking_rates": {"Sampleish": 12.0},
        "reference_limits": ReferenceLimits(
            accepted={"image": 2, "audio": 1}, rejected={"video": "token cost"}
        ),
        "prompt_capacity_tokens": 100,
        "dialogue_tag_form": "<d>[{language}]{text}</d>",
    }


@pytest.fixture
def fixture_profile_kwargs() -> dict:
    """A profile sharing NO measured value with H3 or the stub.

    H3: 24 fps, 768p short side, 32 kHz stereo, 11 languages, 4-15 s.
    Stub: 12 fps, 96x64, 8 kHz mono, 2 languages, 1-6 s.
    Here: 7 fps, 48x32, 11025 Hz stereo, 1 language, 3-13 s.
    """
    return {
        "adapter_key": "profile-agnostic-fixture",
        "profile_id": "fixture@1",
        "roles": {ModelRole.VIDEO, ModelRole.VOICE, ModelRole.LIP_SYNC},
        "native_capabilities": {ModelRole.VOICE, ModelRole.LIP_SYNC},
        "pipeline_class": "FixturePipeline",
        "supported_devices": {DeviceKind.CPU},
        "dtype_policy": {DeviceKind.CPU: {"preferred": "float32", "allowed": ["float32"]}},
        "memory_profiles": [
            MemoryProfile(
                offload_mode=OffloadMode.MODEL_CPU,
                quantization="fixture-int5",
                expected_peak_reserved_bytes=1234,
                expected_host_resident_bytes=5678,
            )
        ],
        "duration_range_seconds": DurationRange(
            min_seconds=3.0, max_seconds=13.0, default_seconds=7.0
        ),
        "frame_rate": 7.0,
        "resolutions": [Resolution(width=48, height=32)],
        "audio_output": AudioOutput(sample_rate=11025, channels=2),
        "dialogue_languages": ["Otherish"],
        "speaking_rates": {"Otherish": 3.5},
        "reference_limits": ReferenceLimits(
            accepted={"image": 9, "audio": 1},
            rejected={"video": "video references are refused on token cost"},
            audio_clip_seconds=DurationRange(
                min_seconds=2.0, max_seconds=11.0, default_seconds=4.0
            ),
        ),
        "prompt_capacity_tokens": 37,
        "dialogue_tag_form": "[[{language}]] {text}",
    }


@pytest.fixture
def fixture_profile(fixture_profile_kwargs: dict) -> ModelProfile:
    return ModelProfile(**fixture_profile_kwargs)


@pytest.fixture
def stub_adapter() -> StubAdapter:
    return StubAdapter()


@pytest.fixture
def fixture_adapter(fixture_profile: ModelProfile) -> StubAdapter:
    """The stub driven by the deliberately-different fixture profile."""
    return StubAdapter(profile=fixture_profile)


@pytest.fixture
def project_roots(tmp_path: Path) -> dict[str, Path]:
    """Isolated `outputs/`, `outputs/.work/`, and `.model-cache/` roots."""
    outputs = tmp_path / "outputs"
    staging = outputs / ".work"
    cache = tmp_path / ".model-cache"
    for path in (outputs, staging, cache):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "project": tmp_path,
        "outputs": outputs,
        "staging": staging,
        "model_cache": cache,
    }


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "subject.png"
    path.write_bytes(_png(32, 32, (180, 140, 120)))
    return path


@pytest.fixture
def sample_images(tmp_path: Path) -> list[Path]:
    paths = []
    for index, shade in enumerate((180, 150, 120)):
        path = tmp_path / f"subject-{index}.png"
        path.write_bytes(_png(32, 32, (shade, 140, 120)))
        paths.append(path)
    return paths


@pytest.fixture
def sample_waveform(tmp_path: Path) -> Path:
    """A short non-silent mono WAV, usable as a reference timbre anchor."""
    path = tmp_path / "reference.wav"
    sample_rate, seconds = 8000, 2.0
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(
            b"".join(
                struct.pack("<h", int(9000 * math.sin(2 * math.pi * 180 * n / sample_rate)))
                for n in range(int(sample_rate * seconds))
            )
        )
    return path


@pytest.fixture
def silent_waveform(tmp_path: Path) -> Path:
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 8000)
    return path


@pytest.fixture
def consent_for(sample_waveform: Path):
    """Build a consent attestation bound to a given audio digest."""

    def _make(
        audio_path: Path | None = None,
        *,
        request_id: uuid.UUID | None = None,
        digest: str | None = None,
    ):
        import hashlib

        source = Path(audio_path) if audio_path is not None else sample_waveform
        return ConsentAttestation(
            request_id=request_id or uuid.uuid4(),
            reference_audio_sha256=digest or hashlib.sha256(source.read_bytes()).hexdigest(),
            confirmed=True,
            confirmed_at=datetime.now(UTC),
        )

    return _make


class RecordingProgress:
    """Captures the ProgressEvent stream the engine publishes to its sink."""

    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)

    def phases(self) -> list[str]:
        """Phases in first-seen order, collapsing consecutive repeats."""
        seen: list[str] = []
        for event in self.events:
            if not seen or seen[-1] != event.phase:
                seen.append(event.phase)
        return seen

    def fractions_for(self, phase: str) -> list[float]:
        return [e.fraction for e in self.events if e.phase == phase and e.fraction is not None]


@pytest.fixture
def progress() -> RecordingProgress:
    return RecordingProgress()


# --------------------------------------------------------------------------
# Stage 2 fixtures: engines, counting adapters, published bundles
# --------------------------------------------------------------------------


class CountingAdapter:
    """Wraps the stub and counts invocations.

    Exists to prove the model is called exactly once per request. A loop or a
    silent retry would show up here and nowhere else.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.load_calls = 0
        self.generate_calls = 0

    @property
    def profile(self):
        return self._inner.profile

    def load(self, **kwargs):
        self.load_calls += 1
        return self._inner.load(**kwargs)

    def generate(self, inputs, **kwargs):
        self.generate_calls += 1
        return self._inner.generate(inputs, **kwargs)

    def unload(self):
        return self._inner.unload()


class ExplodingAdapter:
    """Fails inside generation, to exercise the failure path."""

    def __init__(self, inner, exc) -> None:
        self._inner = inner
        self._exc = exc

    @property
    def profile(self):
        return self._inner.profile

    def load(self, **kwargs):
        return self._inner.load(**kwargs)

    def generate(self, inputs, **kwargs):
        raise self._exc

    def unload(self):
        return self._inner.unload()


@pytest.fixture
def counting_adapter(stub_adapter):
    return CountingAdapter(stub_adapter)


@pytest.fixture
def engine_for():
    """Build an engine bound to a profile and a temporary outputs root."""

    def build(
        profile,
        tmp_path,
        *,
        adapter=None,
        fail_generation: bool = False,
        fail_oom: bool = False,
        fail_verification: bool = False,
    ):
        import ui_contract
        from adapters.stub import StubAdapter
        from errors import GenerationError, OomError
        from pipeline import VideoGenerationEngine

        ui_contract.register_profile(profile)

        base = adapter or StubAdapter(profile)
        if fail_generation:
            base = ExplodingAdapter(base, GenerationError("the generator failed"))
        elif fail_oom:
            base = ExplodingAdapter(base, OomError("out of accelerator memory"))

        engine = VideoGenerationEngine(
            adapter=base,
            outputs_root=tmp_path / "outputs",
            profile=profile,
        )

        if fail_verification:
            from errors import ExportError

            def refuse(*args, **kwargs):
                raise ExportError("verification refused for this test")

            import pipeline as _pipeline

            # Restored by the autouse _restore_pipeline_module fixture below.
            _pipeline.verify_output = refuse

        return engine

    return build


@pytest.fixture(autouse=True)
def _restore_pipeline_module():
    """Undo any monkeypatching an engine fixture applied to the module."""
    import pipeline as _pipeline

    original = _pipeline.verify_output
    yield
    _pipeline.verify_output = original


def _run_for_bundle(tmp_path, profile, images, audio, seed=7):
    from adapters.stub import StubAdapter
    from pipeline import VideoGenerationEngine

    engine = VideoGenerationEngine(
        adapter=StubAdapter(profile),
        outputs_root=tmp_path / "outputs",
        profile=profile,
    )
    result = engine.run(
        image_paths=images,
        audio_path=audio,
        motion_prompt="a slow zoom",
        speech_script="hello there",
        language=profile.dialogue_languages[0],
        consent_confirmed=True,
        seed=seed,
    )
    assert result.state.value == "complete", result.error
    return json.loads(Path(result.manifest_path).read_text())


@pytest.fixture
def published_bundle(tmp_path, stub_adapter, sample_image, sample_waveform):
    return _run_for_bundle(tmp_path, stub_adapter.profile, [sample_image], sample_waveform)


@pytest.fixture
def published_bundle_multi(tmp_path, stub_adapter, sample_images, sample_waveform):
    return _run_for_bundle(
        tmp_path, stub_adapter.profile, sample_images[:2], sample_waveform, seed=8
    )
