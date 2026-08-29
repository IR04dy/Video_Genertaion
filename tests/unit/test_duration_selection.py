"""T027: duration is an INPUT, suggested from the profile, never a rejection."""

from __future__ import annotations

import inspect as _inspect

import pytest

from domain import DurationDecision
from errors import DurationError
from execution import decide_duration, suggest_duration


def test_suggestion_uses_the_profile_speaking_rate(stub_adapter) -> None:
    profile = stub_adapter.profile
    language = "Testish"
    rate = profile.speaking_rates[language]
    script = "x" * 30
    seconds, used = suggest_duration(script=script, language=language, profile=profile)
    assert used is not None and used.rate == rate
    assert seconds == pytest.approx(profile.duration_range_seconds.clamp(30 / rate))


def test_suggestion_is_clamped_to_the_profile_range(stub_adapter) -> None:
    profile = stub_adapter.profile
    seconds, _ = suggest_duration(script="x" * 100_000, language="Testish", profile=profile)
    assert seconds == profile.duration_range_seconds.max_seconds


def test_short_script_is_clamped_up_to_the_minimum(stub_adapter) -> None:
    profile = stub_adapter.profile
    seconds, _ = suggest_duration(script="hi", language="Testish", profile=profile)
    assert seconds == profile.duration_range_seconds.min_seconds


def test_missing_language_rate_falls_back_without_failing(stub_adapter) -> None:
    """`Fixtureish` is a supported language with no measured rate."""
    profile = stub_adapter.profile
    seconds, used = suggest_duration(script="hello", language="Fixtureish", profile=profile)
    assert used is None
    assert seconds == profile.duration_range_seconds.default_seconds


def test_decision_records_the_suggestion_when_accepted(stub_adapter) -> None:
    decision = decide_duration(
        script="hello there", language="Testish", profile=stub_adapter.profile
    )
    assert isinstance(decision, DurationDecision)
    assert decision.operator_overrode is False
    assert decision.effective_duration_seconds == decision.suggested_duration_seconds


def test_operator_override_is_accepted_and_recorded(stub_adapter) -> None:
    profile = stub_adapter.profile
    target = profile.duration_range_seconds.max_seconds
    decision = decide_duration(
        script="hello", language="Testish", profile=profile, requested_seconds=target
    )
    assert decision.operator_overrode is True
    assert decision.effective_duration_seconds == target
    assert decision.requested_duration_seconds == target
    assert decision.overrides


def test_override_outside_the_profile_range_is_refused(stub_adapter) -> None:
    """The ONLY duration error: an override the profile cannot honour."""
    profile = stub_adapter.profile
    with pytest.raises(DurationError):
        decide_duration(
            script="hello",
            language="Testish",
            profile=profile,
            requested_seconds=profile.duration_range_seconds.max_seconds + 1,
        )


def test_override_below_the_minimum_is_refused(stub_adapter) -> None:
    profile = stub_adapter.profile
    with pytest.raises(DurationError):
        decide_duration(
            script="hello",
            language="Testish",
            profile=profile,
            requested_seconds=profile.duration_range_seconds.min_seconds / 2,
        )


def test_frames_follow_the_profile_frame_rate(stub_adapter) -> None:
    profile = stub_adapter.profile
    decision = decide_duration(script="hello", language="Testish", profile=profile)
    expected = round(decision.effective_duration_seconds * profile.frame_rate)
    assert decision.effective_num_frames == expected
    assert decision.frame_rate == profile.frame_rate


def test_decision_carries_profile_resolution_and_sample_rate(stub_adapter) -> None:
    profile = stub_adapter.profile
    decision = decide_duration(script="hello", language="Testish", profile=profile)
    assert decision.resolution == profile.resolutions[0]
    assert decision.audio_sample_rate == profile.audio_output.sample_rate
    assert decision.profile_id == profile.profile_id


def test_a_very_long_script_is_never_rejected(stub_adapter) -> None:
    """The invariant this whole design exists to protect.

    Joint generation cannot measure speech before producing it, so there is no
    fit check to fail. A long script yields a clamped suggestion, not an error.
    """
    decision = decide_duration(
        script="word " * 50_000, language="Testish", profile=stub_adapter.profile
    )
    assert (
        decision.effective_duration_seconds
        == stub_adapter.profile.duration_range_seconds.max_seconds
    )


def test_no_length_based_rejection_path_exists() -> None:
    """Guards against a fit check being reintroduced later."""
    source = _inspect.getsource(decide_duration) + _inspect.getsource(suggest_duration)
    lowered = source.lower()
    for banned in ("too long", "script_too_long", "does not fit", "exceeds the script"):
        assert banned not in lowered


def test_decision_is_profile_agnostic(fixture_profile) -> None:
    """Same call, entirely different measured numbers, no code change."""
    language = fixture_profile.dialogue_languages[0]
    decision = decide_duration(script="hello", language=language, profile=fixture_profile)
    assert decision.frame_rate == fixture_profile.frame_rate
    assert fixture_profile.duration_range_seconds.contains(decision.effective_duration_seconds)
