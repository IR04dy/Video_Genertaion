"""T032: the UI handler contract, tested without launching a server."""

from __future__ import annotations

import ui_contract as ui


def test_multiple_images_are_accepted(stub_adapter, sample_images) -> None:
    state = ui.initial_state(stub_adapter.profile)
    state = ui.on_images_changed(state, sample_images)
    assert len(state.image_paths) == len(sample_images)


def test_no_application_maximum_is_imposed_by_the_ui(stub_adapter, sample_images) -> None:
    """The UI shows the PROFILE's limit; it invents no limit of its own."""
    state = ui.initial_state(stub_adapter.profile)
    assert state.image_limit == stub_adapter.profile.reference_limits.accepted["image"]


def test_the_timbre_anchor_rule_is_displayed(stub_adapter) -> None:
    text = ui.reference_audio_help(stub_adapter.profile).lower()
    assert "different words" in text
    assert "not" in text and ("played" in text or "playback" in text)


def test_help_text_states_the_recording_is_never_played_back(stub_adapter) -> None:
    assert "never" in ui.reference_audio_help(stub_adapter.profile).lower()


def test_language_choices_come_from_the_profile(stub_adapter, fixture_profile) -> None:
    assert ui.language_choices(stub_adapter.profile) == list(
        stub_adapter.profile.dialogue_languages
    )
    assert ui.language_choices(fixture_profile) == list(fixture_profile.dialogue_languages)


def test_suggested_duration_is_offered_and_editable(stub_adapter) -> None:
    state = ui.initial_state(stub_adapter.profile)
    state = ui.on_script_changed(state, "hello there", language="Testish")
    assert state.suggested_duration > 0
    assert state.duration_editable is True
    assert state.duration_min == stub_adapter.profile.duration_range_seconds.min_seconds
    assert state.duration_max == stub_adapter.profile.duration_range_seconds.max_seconds


def test_suggestion_updates_as_the_script_changes(stub_adapter) -> None:
    state = ui.initial_state(stub_adapter.profile)
    short = ui.on_script_changed(state, "hi", language="Testish").suggested_duration
    long = ui.on_script_changed(state, "word " * 200, language="Testish").suggested_duration
    assert long > short


def test_consent_defaults_to_false(stub_adapter) -> None:
    assert ui.initial_state(stub_adapter.profile).consent_confirmed is False


def test_consent_resets_after_every_submit(stub_adapter, sample_image, sample_waveform) -> None:
    state = ui.initial_state(stub_adapter.profile)
    state = ui.on_consent_changed(state, True)
    assert state.consent_confirmed is True
    state = ui.after_submit(state)
    assert state.consent_confirmed is False


def test_consent_resets_when_the_reference_audio_changes(
    stub_adapter, sample_waveform, silent_waveform
) -> None:
    """The single most important UI rule: consent is per-recording."""
    state = ui.initial_state(stub_adapter.profile)
    state = ui.on_audio_changed(state, sample_waveform)
    state = ui.on_consent_changed(state, True)
    assert state.consent_confirmed is True
    state = ui.on_audio_changed(state, silent_waveform)
    assert state.consent_confirmed is False


def test_reselecting_the_same_audio_still_resets_consent(stub_adapter, sample_waveform) -> None:
    state = ui.initial_state(stub_adapter.profile)
    state = ui.on_audio_changed(state, sample_waveform)
    state = ui.on_consent_changed(state, True)
    state = ui.on_audio_changed(state, sample_waveform)
    assert state.consent_confirmed is False


def test_submit_is_blocked_without_consent(stub_adapter, sample_image, sample_waveform) -> None:
    state = ui.initial_state(stub_adapter.profile)
    state = ui.on_images_changed(state, [sample_image])
    state = ui.on_audio_changed(state, sample_waveform)
    state = ui.on_script_changed(state, "hello", language="Testish")
    ready, reason = ui.can_submit(state)
    assert ready is False
    assert "consent" in reason.lower()


def test_submit_is_ready_when_everything_is_supplied(
    stub_adapter, sample_image, sample_waveform
) -> None:
    state = ui.initial_state(stub_adapter.profile)
    state = ui.on_images_changed(state, [sample_image])
    state = ui.on_audio_changed(state, sample_waveform)
    state = ui.on_script_changed(state, "hello", language="Testish")
    state = ui.on_motion_changed(state, "a zoom")
    state = ui.on_consent_changed(state, True)
    ready, reason = ui.can_submit(state)
    assert ready is True, reason


def test_success_maps_to_preview_and_download(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    result = engine_for(stub_adapter.profile, tmp_path).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    outputs = ui.result_outputs(result)
    assert outputs.video_path == result.video_path
    assert outputs.download_path == result.video_path
    assert outputs.error_message is None


def test_failure_clears_video_and_download(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    result = engine_for(stub_adapter.profile, tmp_path, fail_generation=True).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    outputs = ui.result_outputs(result)
    assert outputs.video_path is None
    assert outputs.download_path is None
    assert outputs.error_message


def test_failure_shows_ordered_recovery_suggestions(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    result = engine_for(stub_adapter.profile, tmp_path, fail_oom=True).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    outputs = ui.result_outputs(result)
    assert outputs.suggestions
    assert outputs.suggestions == list(result.error.suggestions)


def test_error_message_is_sanitized(
    tmp_path, stub_adapter, sample_image, sample_waveform, engine_for
) -> None:
    result = engine_for(stub_adapter.profile, tmp_path, fail_generation=True).run(
        image_paths=[sample_image],
        audio_path=sample_waveform,
        motion_prompt="m",
        speech_script="hello",
        language="Testish",
        consent_confirmed=True,
        seed=1,
    )
    assert str(tmp_path) not in ui.result_outputs(result).error_message


def test_memory_summary_reports_both_ceilings(stub_adapter) -> None:
    """A card-only reading would hide the budget that layer-wise offload spends."""
    summary = ui.memory_summary()
    assert "host" in summary.lower()
    assert any(word in summary.lower() for word in ("gpu", "accelerator", "device", "cpu"))


def test_memory_summary_states_when_no_accelerator_is_present(stub_adapter) -> None:
    summary = ui.memory_summary(accelerator_available=False)
    assert "unavailable" in summary.lower() or "no accelerator" in summary.lower()


def test_progress_rendering_shows_phase_and_fraction() -> None:
    text = ui.render_progress(phase="generate", fraction=0.25, elapsed_seconds=90.0)
    assert "generate" in text.lower()
    assert "25" in text
    assert "1:30" in text or "90" in text


def test_progress_rendering_omits_any_estimate_of_time_remaining() -> None:
    """Inference is unbounded; an ETA would be a fabrication."""
    text = ui.render_progress(phase="generate", fraction=0.25, elapsed_seconds=90.0).lower()
    for banned in ("remaining", "eta", "estimated", "left"):
        assert banned not in text


def test_server_binds_to_loopback_with_sharing_disabled() -> None:
    launch = ui.launch_kwargs()
    assert launch["server_name"] == "127.0.0.1"
    assert launch["share"] is False
