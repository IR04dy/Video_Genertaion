"""T026: prompt assembly. Motion may be truncated; dialogue never is."""

from __future__ import annotations

import pytest

from domain import AssembledPrompt
from errors import LanguageError
from prompting import assemble_prompt, count_tokens


def test_dialogue_uses_the_profile_tag_form(stub_adapter) -> None:
    profile = stub_adapter.profile
    prompt = assemble_prompt(
        motion_prompt="a slow zoom",
        speech_script="hello world",
        language="Testish",
        profile=profile,
    )
    assert isinstance(prompt, AssembledPrompt)
    expected = profile.dialogue_tag_form.format(language="Testish", text="hello world")
    assert expected in prompt.rendered


def test_tag_form_comes_from_the_profile_not_a_constant(fixture_profile) -> None:
    language = fixture_profile.dialogue_languages[0]
    prompt = assemble_prompt(
        motion_prompt="m", speech_script="s", language=language, profile=fixture_profile
    )
    assert fixture_profile.dialogue_tag_form.format(language=language, text="s") in prompt.rendered


def test_motion_text_precedes_the_dialogue(stub_adapter) -> None:
    prompt = assemble_prompt(
        motion_prompt="MOTIONMARK",
        speech_script="SPEECHMARK",
        language="Testish",
        profile=stub_adapter.profile,
    )
    assert prompt.rendered.index("MOTIONMARK") < prompt.rendered.index("SPEECHMARK")


def test_language_must_be_supported_by_the_profile(stub_adapter) -> None:
    with pytest.raises(LanguageError):
        assemble_prompt(
            motion_prompt="m",
            speech_script="s",
            language="Nonexistentish",
            profile=stub_adapter.profile,
        )


def test_token_count_is_measured_against_profile_capacity(stub_adapter) -> None:
    prompt = assemble_prompt(
        motion_prompt="a b c",
        speech_script="d e f",
        language="Testish",
        profile=stub_adapter.profile,
    )
    assert prompt.token_capacity == stub_adapter.profile.prompt_capacity_tokens
    assert prompt.token_count <= prompt.token_capacity


def test_long_motion_is_truncated_and_recorded_as_an_override(stub_adapter) -> None:
    profile = stub_adapter.profile
    prompt = assemble_prompt(
        motion_prompt="motion " * 500,
        speech_script="hello",
        language="Testish",
        profile=profile,
    )
    assert prompt.motion_truncation is not None
    t = prompt.motion_truncation
    assert t.discarded_length > 0
    assert t.retained_length + t.discarded_length == t.original_length
    assert prompt.token_count <= profile.prompt_capacity_tokens


def test_truncation_is_never_silent(stub_adapter) -> None:
    prompt = assemble_prompt(
        motion_prompt="motion " * 500,
        speech_script="hello",
        language="Testish",
        profile=stub_adapter.profile,
    )
    assert prompt.motion_truncation is not None


def test_short_motion_records_no_truncation(stub_adapter) -> None:
    prompt = assemble_prompt(
        motion_prompt="a slow zoom",
        speech_script="hello",
        language="Testish",
        profile=stub_adapter.profile,
    )
    assert prompt.motion_truncation is None


def test_speech_script_is_never_truncated(stub_adapter) -> None:
    """The invariant. Speech is carried in full or the request fails; it is
    never quietly shortened to make the prompt fit."""
    script = "sentence. " * 400
    prompt = assemble_prompt(
        motion_prompt="m",
        speech_script=script,
        language="Testish",
        profile=stub_adapter.profile,
    )
    assert script.strip() in prompt.rendered
    assert prompt.dialogue_segments[0].text == script.strip()


def test_dialogue_is_not_reordered_or_dropped(stub_adapter) -> None:
    script = "first. second. third."
    prompt = assemble_prompt(
        motion_prompt="m",
        speech_script=script,
        language="Testish",
        profile=stub_adapter.profile,
    )
    joined = " ".join(seg.text for seg in prompt.dialogue_segments)
    assert joined == script.strip()


def test_a_script_too_long_for_capacity_still_carries_the_full_script(stub_adapter) -> None:
    """Capacity pressure is absorbed by motion, never by speech. When the
    script alone exceeds capacity, the recorded count exceeds it honestly
    rather than the script being cut to fit."""
    script = "word " * 5_000
    prompt = assemble_prompt(
        motion_prompt="m",
        speech_script=script,
        language="Testish",
        profile=stub_adapter.profile,
    )
    assert script.strip() in prompt.rendered
    assert prompt.motion_text == ""
    assert prompt.over_capacity is True


def test_empty_script_is_refused(stub_adapter) -> None:
    from errors import ValidationError as VErr

    with pytest.raises(VErr):
        assemble_prompt(
            motion_prompt="m",
            speech_script="   ",
            language="Testish",
            profile=stub_adapter.profile,
        )


def test_inputs_are_trimmed(stub_adapter) -> None:
    prompt = assemble_prompt(
        motion_prompt="  zoom  ",
        speech_script="  hello  ",
        language="Testish",
        profile=stub_adapter.profile,
    )
    assert prompt.motion_text == "zoom"
    assert prompt.dialogue_segments[0].text == "hello"


def test_count_tokens_is_monotonic_in_length() -> None:
    assert count_tokens("a") <= count_tokens("a b") <= count_tokens("a b c")


def test_count_tokens_uses_an_injected_tokenizer_when_given() -> None:
    prompt_tokens = count_tokens("a b c", tokenizer=lambda text: list(text))
    assert prompt_tokens == 5


def test_structuring_version_is_recorded(stub_adapter) -> None:
    prompt = assemble_prompt(
        motion_prompt="m",
        speech_script="s",
        language="Testish",
        profile=stub_adapter.profile,
    )
    assert prompt.structuring_version
