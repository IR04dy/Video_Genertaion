"""T009: typed domain records and their serialization round-trips.

Every measured value used here is deliberately arbitrary. Nothing in this file
may assert a real model's duration range, frame rate, resolution, sample rate,
language set, reference limit, or token capacity — those live in adapter
profiles, and a shared assertion about one is an invariant violation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from domain import (
    ArtifactKind,
    ArtifactRecord,
    AssembledPrompt,
    AudioOutput,
    ConsentAttestation,
    DeviceKind,
    DialogueSegment,
    DurationDecision,
    DurationRange,
    ErrorCode,
    ErrorDetail,
    GenerationResult,
    MemoryProfile,
    MemorySnapshot,
    ModelProfile,
    ModelRole,
    MotionTruncation,
    OffloadMode,
    ProgressEvent,
    ReferenceLimits,
    RequestState,
    Resolution,
    RuntimeProfile,
    StageKind,
)

SHA = "a" * 64


def test_model_profile_round_trips(sample_profile_kwargs: dict) -> None:
    profile = ModelProfile(**sample_profile_kwargs)
    restored = ModelProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile


def test_native_capabilities_must_be_subset_of_roles(sample_profile_kwargs: dict) -> None:
    kwargs = sample_profile_kwargs | {
        "roles": {ModelRole.VIDEO},
        "native_capabilities": {ModelRole.VOICE},
    }
    with pytest.raises(ValidationError, match="native_capabilities"):
        ModelProfile(**kwargs)


def test_native_voice_requires_video_role(sample_profile_kwargs: dict) -> None:
    kwargs = sample_profile_kwargs | {
        "roles": {ModelRole.VOICE},
        "native_capabilities": {ModelRole.VOICE},
    }
    with pytest.raises(ValidationError, match="video"):
        ModelProfile(**kwargs)


def test_profile_requires_at_least_one_device(sample_profile_kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ModelProfile(**(sample_profile_kwargs | {"supported_devices": set()}))


def test_profile_carries_no_license_field(sample_profile_kwargs: dict) -> None:
    """License data is intentionally absent from the model, not merely unused."""
    fields = set(ModelProfile.model_fields)
    assert not {f for f in fields if "licen" in f.lower()}


def test_duration_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError):
        DurationRange(min_seconds=9.0, max_seconds=3.0, default_seconds=5.0)


def test_duration_range_default_must_be_inside_range() -> None:
    with pytest.raises(ValidationError):
        DurationRange(min_seconds=3.0, max_seconds=9.0, default_seconds=11.0)


def test_consent_rejects_unconfirmed() -> None:
    with pytest.raises(ValidationError):
        ConsentAttestation(
            request_id=uuid.uuid4(),
            reference_audio_sha256=SHA,
            confirmed=False,
            confirmed_at=datetime.now(UTC),
        )


def test_consent_rejects_malformed_digest() -> None:
    with pytest.raises(ValidationError):
        ConsentAttestation(
            request_id=uuid.uuid4(),
            reference_audio_sha256="not-a-digest",
            confirmed=True,
            confirmed_at=datetime.now(UTC),
        )


def test_duration_decision_derives_frames_from_profile_frame_rate() -> None:
    """`effective_num_frames` follows duration x the PROFILE's frame rate."""
    decision = DurationDecision(
        suggested_duration_seconds=5.0,
        speaking_rate_used=None,
        requested_duration_seconds=None,
        operator_overrode=False,
        effective_duration_seconds=5.0,
        effective_num_frames=75,
        frame_rate=15.0,  # arbitrary; not any real model's rate
        resolution=Resolution(width=128, height=128),
        audio_sample_rate=8000,
        overrides=[],
        profile_id="fixture@1",
    )
    assert decision.effective_num_frames == 75


def test_duration_decision_rejects_frame_count_disagreeing_with_rate() -> None:
    with pytest.raises(ValidationError, match="effective_num_frames"):
        DurationDecision(
            suggested_duration_seconds=5.0,
            speaking_rate_used=None,
            requested_duration_seconds=None,
            operator_overrode=False,
            effective_duration_seconds=5.0,
            effective_num_frames=999,
            frame_rate=15.0,
            resolution=Resolution(width=128, height=128),
            audio_sample_rate=8000,
            overrides=[],
            profile_id="fixture@1",
        )


def test_assembled_prompt_round_trips() -> None:
    prompt = AssembledPrompt(
        motion_text="a slow pan",
        dialogue_segments=[DialogueSegment(language="Testish", text="hello there")],
        rendered="a slow pan <d>[Testish]hello there</d>",
        token_count=11,
        token_capacity=64,
        motion_truncation=MotionTruncation(
            original_length=40, retained_length=10, discarded_length=30
        ),
        structuring_version="1",
    )
    assert AssembledPrompt.model_validate_json(prompt.model_dump_json()) == prompt


def test_assembled_prompt_records_over_capacity_without_refusing() -> None:
    """No request terminates because of script length, so an over-capacity
    prompt is a recorded fact rather than a validation failure."""
    prompt = AssembledPrompt(
        motion_text="x",
        dialogue_segments=[DialogueSegment(language="Testish", text="hi")],
        rendered="x",
        token_count=65,
        token_capacity=64,
        motion_truncation=None,
        structuring_version="1",
    )
    assert prompt.over_capacity is True


def test_assembled_prompt_within_capacity_is_not_flagged() -> None:
    prompt = AssembledPrompt(
        motion_text="x",
        dialogue_segments=[DialogueSegment(language="Testish", text="hi")],
        rendered="x",
        token_count=10,
        token_capacity=64,
        motion_truncation=None,
        structuring_version="1",
    )
    assert prompt.over_capacity is False


def test_motion_truncation_lengths_must_be_consistent() -> None:
    with pytest.raises(ValidationError):
        MotionTruncation(original_length=40, retained_length=10, discarded_length=5)


def test_artifact_record_rejects_absolute_path() -> None:
    with pytest.raises(ValidationError):
        ArtifactRecord(
            kind=ArtifactKind.FINAL_MP4,
            relative_path="/etc/passwd",
            media_type="video/mp4",
            size_bytes=1,
            sha256=SHA,
            created_by_stage=StageKind.EXPORT,
        )


@pytest.mark.parametrize("bad", ["../escape.mp4", "a/../../b.mp4", "C:/x.mp4", "a\\b.mp4"])
def test_artifact_record_rejects_escaping_paths(bad: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRecord(
            kind=ArtifactKind.FINAL_MP4,
            relative_path=bad,
            media_type="video/mp4",
            size_bytes=1,
            sha256=SHA,
            created_by_stage=StageKind.EXPORT,
        )


def test_progress_event_fraction_bounds() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            ProgressEvent(
                request_id=uuid.uuid4(),
                phase="generate",
                fraction=bad,
                message="working",
                memory=None,
                timestamp=datetime.now(UTC),
            )


def test_progress_event_allows_null_fraction() -> None:
    event = ProgressEvent(
        request_id=uuid.uuid4(),
        phase="load_model",
        fraction=None,
        message="loading",
        memory=None,
        timestamp=datetime.now(UTC),
    )
    assert event.fraction is None


def test_memory_snapshot_round_trips() -> None:
    snapshot = MemorySnapshot(
        available=True,
        device_name="Fixture Device",
        allocated_bytes=1,
        reserved_bytes=2,
        peak_allocated_bytes=3,
        peak_reserved_bytes=4,
        free_bytes=5,
        total_bytes=6,
        host_resident_bytes=7,
        reserved_gate_passed=True,
        unavailable_reason=None,
    )
    assert MemorySnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_unavailable_snapshot_requires_a_reason() -> None:
    with pytest.raises(ValidationError, match="unavailable_reason"):
        MemorySnapshot(available=False, unavailable_reason=None)


@pytest.mark.parametrize(
    "state", [RequestState.COMPLETE, RequestState.FAILED, RequestState.CANCELLED]
)
def test_generation_result_accepts_only_terminal_states(state: RequestState) -> None:
    result = GenerationResult(
        request_id=uuid.uuid4(),
        state=state,
        video_path=None,
        bundle_path=None,
        manifest_path=None,
        artifact_inventory=[],
        retained_bytes=0,
        execution_plan={},
        duration_seconds=1.0,
        memory_by_stage={},
        error=None
        if state is RequestState.COMPLETE
        else ErrorDetail(
            code=ErrorCode.GENERATION, message="failed", retryable=True, suggestions=[]
        ),
    )
    assert result.state is state


def test_generation_result_rejects_non_terminal_state() -> None:
    with pytest.raises(ValidationError, match="terminal"):
        GenerationResult(
            request_id=uuid.uuid4(),
            state=RequestState.RUNNING,
            video_path=None,
            bundle_path=None,
            manifest_path=None,
            artifact_inventory=[],
            retained_bytes=0,
            execution_plan={},
            duration_seconds=1.0,
            memory_by_stage={},
            error=None,
        )


def test_failed_result_must_carry_an_error() -> None:
    with pytest.raises(ValidationError, match="error"):
        GenerationResult(
            request_id=uuid.uuid4(),
            state=RequestState.FAILED,
            video_path=None,
            bundle_path=None,
            manifest_path=None,
            artifact_inventory=[],
            retained_bytes=0,
            execution_plan={},
            duration_seconds=1.0,
            memory_by_stage={},
            error=None,
        )


def test_complete_result_must_not_carry_an_error() -> None:
    with pytest.raises(ValidationError, match="error"):
        GenerationResult(
            request_id=uuid.uuid4(),
            state=RequestState.COMPLETE,
            video_path=None,
            bundle_path=None,
            manifest_path=None,
            artifact_inventory=[],
            retained_bytes=0,
            execution_plan={},
            duration_seconds=1.0,
            memory_by_stage={},
            error=ErrorDetail(
                code=ErrorCode.INTERNAL, message="x", retryable=False, suggestions=[]
            ),
        )


def test_runtime_profile_offload_mode_is_declared_not_discovered() -> None:
    profile = RuntimeProfile(
        name="fixture",
        device=DeviceKind.CPU,
        dtype="float32",
        offload_mode=OffloadMode.LAYER_WISE,
        quantization="fixture-int4",
        max_reserved_bytes=1024,
        max_host_resident_bytes=4096,
        minimum_free_headroom_bytes=None,
        warnings=[],
    )
    assert profile.offload_mode is OffloadMode.LAYER_WISE
    assert profile.quantization == "fixture-int4"


def test_enum_value_sets_match_the_data_model() -> None:
    assert {k.value for k in ArtifactKind} == {
        "original_image",
        "reference_audio",
        "derived_voice",
        "assembled_prompt",
        "decoded_video",
        "decoded_audio",
        "final_mp4",
        "metadata",
    }
    assert {s.value for s in StageKind} == {
        "validate",
        "prepare_references",
        "assemble_prompt",
        "plan_duration",
        "load_model",
        "generate",
        "decode",
        "export",
        "verify",
        "metadata",
        "publish",
    }
    assert {d.value for d in DeviceKind} == {"cuda", "mps", "cpu"}
    assert {o.value for o in OffloadMode} == {
        "none",
        "model_cpu",
        "sequential_cpu",
        "layer_wise",
    }


def test_no_lip_sync_or_speech_artifact_kinds_exist() -> None:
    """Joint generation means no separate speech or lip-sync artifact."""
    values = {k.value for k in ArtifactKind}
    assert "speech_track" not in values
    assert not any("lip" in v for v in values)


def test_audio_output_and_reference_limits_round_trip() -> None:
    audio = AudioOutput(sample_rate=8000, channels=1)
    limits = ReferenceLimits(
        accepted={"image": 2, "audio": 1},
        rejected={"video": "token cost"},
        audio_clip_seconds=DurationRange(min_seconds=1.0, max_seconds=4.0, default_seconds=2.0),
    )
    assert AudioOutput.model_validate_json(audio.model_dump_json()) == audio
    assert ReferenceLimits.model_validate_json(limits.model_dump_json()) == limits


def test_memory_profile_round_trips() -> None:
    mp = MemoryProfile(
        offload_mode=OffloadMode.NONE,
        quantization=None,
        expected_peak_reserved_bytes=1024,
        expected_host_resident_bytes=2048,
    )
    assert MemoryProfile.model_validate_json(mp.model_dump_json()) == mp
