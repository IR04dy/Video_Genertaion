"""T034: the leak detector.

Every assertion here runs the SAME code against a fixture profile whose measured
values are all different from the stub's and from the production profile's. A model-specific
number that has leaked into shared code fails here and nowhere else, which is the
entire reason this file exists.
"""

from __future__ import annotations

import pytest

from execution import decide_duration
from prompting import assemble_prompt


@pytest.fixture(params=["stub", "fixture"])
def either_profile(request, stub_adapter, fixture_profile):
    return stub_adapter.profile if request.param == "stub" else fixture_profile


def test_the_two_profiles_actually_differ(stub_adapter, fixture_profile) -> None:
    """If this ever passes trivially, the rest of the file proves nothing."""
    a, b = stub_adapter.profile, fixture_profile
    assert a.frame_rate != b.frame_rate
    assert a.audio_output.sample_rate != b.audio_output.sample_rate
    assert a.resolutions[0] != b.resolutions[0]
    assert a.prompt_capacity_tokens != b.prompt_capacity_tokens
    assert set(a.dialogue_languages).isdisjoint(b.dialogue_languages)
    assert a.dialogue_tag_form != b.dialogue_tag_form
    assert a.duration_range_seconds.max_seconds != b.duration_range_seconds.max_seconds
    assert a.reference_limits.accepted["image"] != b.reference_limits.accepted["image"]


def test_duration_uses_only_profile_values(either_profile) -> None:
    language = either_profile.dialogue_languages[0]
    decision = decide_duration(script="hello there", language=language, profile=either_profile)
    assert decision.frame_rate == either_profile.frame_rate
    assert decision.audio_sample_rate == either_profile.audio_output.sample_rate
    assert decision.resolution == either_profile.resolutions[0]
    assert either_profile.duration_range_seconds.contains(decision.effective_duration_seconds)
    assert decision.profile_id == either_profile.profile_id


def test_frames_track_each_profiles_own_rate(either_profile) -> None:
    language = either_profile.dialogue_languages[0]
    decision = decide_duration(script="hello", language=language, profile=either_profile)
    assert decision.effective_num_frames == round(
        decision.effective_duration_seconds * either_profile.frame_rate
    )


def test_prompt_uses_each_profiles_own_tag_form(either_profile) -> None:
    language = either_profile.dialogue_languages[0]
    prompt = assemble_prompt(
        motion_prompt="m", speech_script="hello", language=language, profile=either_profile
    )
    assert (
        either_profile.dialogue_tag_form.format(language=language, text="hello") in prompt.rendered
    )
    assert prompt.token_capacity == either_profile.prompt_capacity_tokens


def test_each_profile_rejects_the_other_profiles_language(stub_adapter, fixture_profile) -> None:
    from errors import LanguageError

    with pytest.raises(LanguageError):
        assemble_prompt(
            motion_prompt="m",
            speech_script="s",
            language=fixture_profile.dialogue_languages[0],
            profile=stub_adapter.profile,
        )
    with pytest.raises(LanguageError):
        assemble_prompt(
            motion_prompt="m",
            speech_script="s",
            language="Testish",
            profile=fixture_profile,
        )


def test_reference_limits_track_the_profile(
    tmp_path, either_profile, sample_images, sample_waveform, consent_for
) -> None:
    import uuid

    from errors import ReferenceError
    from execution import stage_references
    from media import FaceResult

    class OK:
        def detect(self, path):
            return FaceResult(face_count=1, has_mouth=True)

    limit = either_profile.reference_limits.accepted["image"]
    request_id = uuid.uuid4()
    with pytest.raises(ReferenceError):
        stage_references(
            request_id=request_id,
            image_paths=(sample_images * 9)[: limit + 1],
            audio_path=sample_waveform,
            consent=consent_for(sample_waveform, request_id=request_id),
            profile=either_profile,
            staging_root=tmp_path / f"work-{limit}",
            detector=OK(),
        )


def test_end_to_end_runs_against_both_profiles(
    tmp_path, either_profile, sample_image, sample_waveform, consent_for, engine_for
) -> None:
    """The strongest form of the check: a full generation, twice, same code."""
    engine = engine_for(either_profile, tmp_path)
    result = engine.run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="a slow zoom",
        speech_script="hello there",
        language=either_profile.dialogue_languages[0],
        consent_confirmed=True,
        seed=3,
    )
    assert result.state.value == "complete"
