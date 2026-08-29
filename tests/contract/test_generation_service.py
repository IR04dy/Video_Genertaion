"""T031: the generation-service contract, driven with fakes only."""

from __future__ import annotations

import pytest

from domain import RequestState
from errors import ConsentError


def test_successful_run_reaches_a_complete_terminal_state(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    engine = engine_for(stub_adapter.profile, tmp_path)
    result = engine.run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="a slow zoom",
        speech_script="hello there",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    assert result.state is RequestState.COMPLETE
    assert result.error is None


def test_success_publishes_a_bundle_and_a_playable_video(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    engine = engine_for(stub_adapter.profile, tmp_path)
    result = engine.run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    from pathlib import Path

    assert Path(result.video_path).exists()
    assert Path(result.bundle_path).is_dir()
    assert Path(result.manifest_path).exists()
    assert result.retained_bytes > 0


def test_preview_and_download_are_the_same_published_path(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    """Two paths would let the user download a different file than they watched."""
    engine = engine_for(stub_adapter.profile, tmp_path)
    result = engine.run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    from pathlib import Path

    assert Path(result.video_path).parent == Path(result.bundle_path)


def test_video_path_is_absent_until_verification_passes(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    engine = engine_for(stub_adapter.profile, tmp_path, fail_verification=True)
    result = engine.run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    assert result.state is RequestState.FAILED
    assert result.video_path is None


def test_consent_is_a_precondition(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    engine = engine_for(stub_adapter.profile, tmp_path)
    with pytest.raises(ConsentError):
        engine.run(
            image_paths=[sample_image],
            audio_path=sample_waveform,
            motion_prompt="m",
            speech_script="hello",
            language="Testish",
            consent_confirmed=False,
            seed=1,
        )


def test_the_model_is_invoked_exactly_once(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for, counting_adapter
) -> None:
    """One generation per request. No loop, no retry, no second pass."""
    engine = engine_for(stub_adapter.profile, tmp_path, adapter=counting_adapter)
    engine.run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    assert counting_adapter.generate_calls == 1


def test_failure_publishes_nothing_and_leaves_no_staging(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    engine = engine_for(stub_adapter.profile, tmp_path, fail_generation=True)
    result = engine.run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    assert result.state is RequestState.FAILED
    assert result.bundle_path is None
    outputs = tmp_path / "outputs"
    leftovers = list((outputs / ".work").glob("*")) if (outputs / ".work").exists() else []
    assert leftovers == []


def test_failure_leaves_previously_published_bundles_untouched(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    good = engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    engine_for(stub_adapter.profile, tmp_path, fail_generation=True).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=2,
    )
    from pathlib import Path

    assert Path(good.bundle_path).is_dir()
    assert Path(good.video_path).exists()


def test_result_records_the_effective_execution_plan(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    result = engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=42,
    )
    plan = result.execution_plan
    assert plan["seed"] == 42
    assert plan["language"] == "Testish"
    assert plan["profile_id"] == stub_adapter.profile.profile_id


def test_seed_is_generated_and_recorded_when_absent(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    result = engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=None,
    )
    assert isinstance(result.execution_plan["seed"], int)


def test_same_seed_gives_the_same_bytes(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    from pathlib import Path

    digests = []
    for _ in range(2):
        result = engine_for(stub_adapter.profile, tmp_path).run(
            image_paths=[sample_image],
            audio_path=sample_waveform,
            motion_prompt="m",
            speech_script="hello",
            language="Testish",
            consent_confirmed=True,
            seed=99,
        )
        from media import sha256_file

        digests.append(sha256_file(Path(result.video_path)))
    assert digests[0] == digests[1]


# --- Progress contract (T065, US3) -----------------------------------------


ORDERED_PHASES = [
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
]


def test_phases_arrive_in_the_documented_order(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for, progress
) -> None:
    engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
        progress=progress,
    )
    seen = [p for p in progress.phases() if p in ORDERED_PHASES]
    positions = [ORDERED_PHASES.index(p) for p in seen]
    assert positions == sorted(positions)


def test_events_are_scoped_to_the_request(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for, progress
) -> None:
    result = engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
        progress=progress,
    )
    assert {e.request_id for e in progress.events} == {result.request_id}


def test_fractions_are_monotonic_within_a_stage(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for, progress
) -> None:
    engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
        progress=progress,
    )
    for phase in ("generate", "decode"):
        fractions = [
            e.fraction for e in progress.events if e.phase == phase and e.fraction is not None
        ]
        assert fractions == sorted(fractions)


def test_exactly_one_terminal_event(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for, progress
) -> None:
    engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
        progress=progress,
    )
    terminal = [e for e in progress.events if e.phase in {"complete", "failed", "cancelled"}]
    assert len(terminal) == 1


def test_progress_carries_both_memory_readings(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for, progress
) -> None:
    engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
        progress=progress,
    )
    with_memory = [e for e in progress.events if e.memory is not None]
    assert with_memory
    assert any(e.memory.host_resident_bytes for e in with_memory)


def test_progress_never_leaks_prompts_or_absolute_paths(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for, progress
) -> None:
    engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="SECRETMOTION",
        speech_script="SECRETSPEECH",
        language="Testish",
        consent_confirmed=True,
        seed=1,
        progress=progress,
    )
    blob = " ".join(e.message for e in progress.events)
    assert "SECRETMOTION" not in blob
    assert "SECRETSPEECH" not in blob
    assert str(tmp_path) not in blob


def test_a_broken_progress_consumer_does_not_fail_the_run(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    def exploding(event):
        raise RuntimeError("consumer is broken")

    result = engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
        progress=exploding,
    )
    assert result.state is RequestState.COMPLETE
