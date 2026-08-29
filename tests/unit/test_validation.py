"""T028: request preflight. Ordering matters as much as the rules."""

from __future__ import annotations

import pytest

from errors import DurationError, LanguageError, ValidationError
from validation import validate_guidance, validate_language, validate_seed


def test_language_must_be_in_the_profile_list(stub_adapter) -> None:
    validate_language("Testish", profile=stub_adapter.profile)
    with pytest.raises(LanguageError):
        validate_language("Klingon", profile=stub_adapter.profile)


def test_language_check_is_exact_not_prefix(stub_adapter) -> None:
    with pytest.raises(LanguageError):
        validate_language("Test", profile=stub_adapter.profile)


def test_language_list_comes_from_the_profile(fixture_profile) -> None:
    validate_language(fixture_profile.dialogue_languages[0], profile=fixture_profile)
    with pytest.raises(LanguageError):
        validate_language("Testish", profile=fixture_profile)


@pytest.mark.parametrize("seed", [0, 1, 2**63 - 1])
def test_valid_seeds_are_accepted(seed: int) -> None:
    assert validate_seed(seed) == seed


@pytest.mark.parametrize("seed", [-1, 2**63, 2**64])
def test_out_of_range_seeds_are_refused(seed: int) -> None:
    with pytest.raises(ValidationError):
        validate_seed(seed)


def test_null_seed_is_generated_and_in_range() -> None:
    generated = validate_seed(None)
    assert 0 <= generated < 2**63


def test_non_integer_seed_is_refused() -> None:
    with pytest.raises(ValidationError):
        validate_seed("hello")  # type: ignore[arg-type]


def test_guidance_within_profile_bounds_is_accepted(stub_adapter) -> None:
    assert validate_guidance(1.0, profile=stub_adapter.profile) == 1.0


def test_guidance_may_be_absent(stub_adapter) -> None:
    assert validate_guidance(None, profile=stub_adapter.profile) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_guidance_is_refused(bad: float, stub_adapter) -> None:
    with pytest.raises(ValidationError):
        validate_guidance(bad, profile=stub_adapter.profile)


def test_preflight_reports_the_first_failure_in_a_stable_order(
    tmp_path, stub_adapter, sample_image, sample_waveform, consent_for
) -> None:
    """Language is checked before duration, so a request wrong in both ways
    always reports the language problem — the one the operator can act on."""
    import uuid as _uuid

    from execution import preflight

    request_id = _uuid.uuid4()
    with pytest.raises(LanguageError):
        preflight(
            request_id=request_id,
            motion_prompt="m",
            speech_script="s",
            language="Klingon",
            requested_seconds=stub_adapter.profile.duration_range_seconds.max_seconds + 99,
            seed=None,
            guidance_scale=None,
            profile=stub_adapter.profile,
        )


def test_preflight_reports_duration_when_language_is_fine(stub_adapter) -> None:
    import uuid as _uuid

    from execution import preflight

    with pytest.raises(DurationError):
        preflight(
            request_id=_uuid.uuid4(),
            motion_prompt="m",
            speech_script="s",
            language="Testish",
            requested_seconds=stub_adapter.profile.duration_range_seconds.max_seconds + 99,
            seed=None,
            guidance_scale=None,
            profile=stub_adapter.profile,
        )


def test_preflight_returns_prompt_and_duration_on_success(stub_adapter) -> None:
    import uuid as _uuid

    from execution import preflight

    prompt, duration, seed = preflight(
        request_id=_uuid.uuid4(),
        motion_prompt="a zoom",
        speech_script="hello",
        language="Testish",
        requested_seconds=None,
        seed=7,
        guidance_scale=None,
        profile=stub_adapter.profile,
    )
    assert prompt.rendered
    assert duration.effective_num_frames > 0
    assert seed == 7
