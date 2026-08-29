"""T029: container export and the verification that gates `video_path`."""

from __future__ import annotations

import pytest

from errors import AppError, CodecError, ExportError
from export import (
    build_ffmpeg_args,
    export_mp4,
    probe_media,
    verify_output,
)


def _artifacts(tmp_path, stub_adapter, seconds=2.0):
    """Run the stub adapter to get real frames and a real waveform."""
    import uuid

    from adapters.base import GenerationInputs
    from execution import decide_duration
    from prompting import assemble_prompt

    profile = stub_adapter.profile
    duration = decide_duration(
        script="hello", language="Testish", profile=profile, requested_seconds=seconds
    )
    prompt = assemble_prompt(
        motion_prompt="m", speech_script="hello", language="Testish", profile=profile
    )
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "ref.wav").write_bytes(b"")
    return stub_adapter.generate(
        GenerationInputs(
            request_id=str(uuid.uuid4()),
            image_paths=(),
            audio_path=work / "ref.wav",
            prompt=prompt,
            duration=duration,
            seed=1,
        )
    )


def test_ffmpeg_args_are_a_list_never_a_shell_string(tmp_path) -> None:
    """Argument vectors only. A shell string would make any filename with a
    quote or semicolon into command injection."""
    args = build_ffmpeg_args(
        frames_dir=tmp_path / "frames",
        audio_path=tmp_path / "a.wav",
        out_path=tmp_path / "o.mp4",
        frame_rate=12.0,
    )
    assert isinstance(args, list)
    assert all(isinstance(a, str) for a in args)


def test_ffmpeg_args_do_not_interpolate_untrusted_text(tmp_path) -> None:
    weird = tmp_path / "a b; rm -rf /.wav"
    args = build_ffmpeg_args(
        frames_dir=tmp_path / "frames",
        audio_path=weird,
        out_path=tmp_path / "o.mp4",
        frame_rate=12.0,
    )
    assert str(weird) in args  # present as ONE argument, not spliced
    assert not any(";" in a and a != str(weird) for a in args)


def test_export_produces_a_real_mp4(tmp_path, stub_adapter) -> None:
    art = _artifacts(tmp_path, stub_adapter)
    out = export_mp4(
        frames_dir=art.frames_path,
        audio_path=art.audio_path,
        out_path=tmp_path / "out.mp4",
        frame_rate=art.frame_rate,
    )
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[4:8] == b"ftyp"


def test_probe_reports_both_streams(tmp_path, stub_adapter) -> None:
    art = _artifacts(tmp_path, stub_adapter)
    out = export_mp4(
        frames_dir=art.frames_path,
        audio_path=art.audio_path,
        out_path=tmp_path / "out.mp4",
        frame_rate=art.frame_rate,
    )
    probe = probe_media(out)
    assert probe.has_video is True
    assert probe.has_audio is True
    assert probe.video_seconds > 0
    assert probe.audio_seconds > 0


def test_verify_accepts_a_well_formed_output(tmp_path, stub_adapter) -> None:
    art = _artifacts(tmp_path, stub_adapter)
    out = export_mp4(
        frames_dir=art.frames_path,
        audio_path=art.audio_path,
        out_path=tmp_path / "out.mp4",
        frame_rate=art.frame_rate,
    )
    result = verify_output(out, expected_seconds=2.0, frame_rate=art.frame_rate)
    assert result.ok is True
    assert result.non_silent is True


def test_verify_rejects_a_silent_audio_track(tmp_path, stub_adapter, silent_waveform) -> None:
    """A silent speech track means the voice never spoke; that is a failure,
    not a successful render."""
    import struct as _s
    import wave as _w

    art = _artifacts(tmp_path, stub_adapter, seconds=2.0)

    # Silence of the RIGHT length, so the duration check cannot fire first and
    # hide the assertion this test is actually making.
    silent = tmp_path / "silent-2s.wav"
    with _w.open(str(silent), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(art.audio_sample_rate)
        handle.writeframes(_s.pack("<h", 0) * (art.audio_sample_rate * 2))

    out = export_mp4(
        frames_dir=art.frames_path,
        audio_path=silent,
        out_path=tmp_path / "silent.mp4",
        frame_rate=art.frame_rate,
    )
    with pytest.raises(ExportError, match="silen"):
        verify_output(out, expected_seconds=2.0, frame_rate=art.frame_rate)


def test_verify_rejects_streams_disagreeing_by_more_than_one_frame(tmp_path, stub_adapter) -> None:
    art = _artifacts(tmp_path, stub_adapter, seconds=2.0)
    out = export_mp4(
        frames_dir=art.frames_path,
        audio_path=art.audio_path,
        out_path=tmp_path / "out.mp4",
        frame_rate=art.frame_rate,
    )
    with pytest.raises(ExportError, match="duration"):
        verify_output(out, expected_seconds=5.0, frame_rate=art.frame_rate)


def test_tolerance_is_one_frame_at_the_profile_rate(tmp_path, stub_adapter) -> None:
    """The tolerance scales with the profile's frame rate; it is not a constant."""
    art = _artifacts(tmp_path, stub_adapter, seconds=2.0)
    out = export_mp4(
        frames_dir=art.frames_path,
        audio_path=art.audio_path,
        out_path=tmp_path / "out.mp4",
        frame_rate=art.frame_rate,
    )
    within = 2.0 + (1.0 / art.frame_rate) * 0.9
    assert verify_output(out, expected_seconds=within, frame_rate=art.frame_rate).ok


def test_verify_rejects_a_file_that_is_not_a_container(tmp_path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a container")
    with pytest.raises((CodecError, ExportError, AppError)):
        verify_output(bad, expected_seconds=1.0, frame_rate=12.0)


def test_export_refuses_when_there_are_no_frames(tmp_path) -> None:
    empty = tmp_path / "frames"
    empty.mkdir()
    (tmp_path / "a.wav").write_bytes(b"")
    with pytest.raises(AppError):
        export_mp4(
            frames_dir=empty,
            audio_path=tmp_path / "a.wav",
            out_path=tmp_path / "o.mp4",
            frame_rate=12.0,
        )


def test_publication_is_atomic(tmp_path, stub_adapter) -> None:
    """No partially written file is ever visible at the destination."""
    from export import publish_atomically

    art = _artifacts(tmp_path, stub_adapter)
    staged = export_mp4(
        frames_dir=art.frames_path,
        audio_path=art.audio_path,
        out_path=tmp_path / "staged.mp4",
        frame_rate=art.frame_rate,
    )
    dest = tmp_path / "published" / "final.mp4"
    dest.parent.mkdir()
    published = publish_atomically(staged, dest)
    assert published.exists()
    assert not staged.exists()


def test_temporary_files_are_cleaned_up_on_failure(tmp_path) -> None:
    empty = tmp_path / "frames"
    empty.mkdir()
    (tmp_path / "a.wav").write_bytes(b"")
    with pytest.raises(AppError):
        export_mp4(
            frames_dir=empty,
            audio_path=tmp_path / "a.wav",
            out_path=tmp_path / "o.mp4",
            frame_rate=12.0,
        )
    assert list(tmp_path.glob("*.part")) == []
    assert not (tmp_path / "o.mp4").exists()
