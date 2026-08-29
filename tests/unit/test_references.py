"""T023: reference staging. No application maximum; the profile is the bound."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from domain import ReferenceSet
from errors import AppError, ReferenceError
from execution import stage_references
from media import FaceResult


class FakeDetector:
    def __init__(self, result: FaceResult | None = None, per_path=None) -> None:
        self._result = result or FaceResult(face_count=1, has_mouth=True)
        self._per_path = per_path or {}

    def detect(self, path: Path) -> FaceResult:
        return self._per_path.get(Path(path).name, self._result)


def _stage(tmp_path, profile, images, audio, consent_for, **kw):
    request_id = kw.pop("request_id", None) or uuid.uuid4()
    return stage_references(
        request_id=request_id,
        image_paths=images,
        audio_path=audio,
        consent=kw.pop("consent", None) or consent_for(audio, request_id=request_id),
        profile=profile,
        staging_root=tmp_path / "work",
        detector=kw.pop("detector", FakeDetector()),
        **kw,
    )


def test_single_image_is_accepted(
    tmp_path, stub_adapter, sample_image, sample_waveform, consent_for
):
    refs = _stage(tmp_path, stub_adapter.profile, [sample_image], sample_waveform, consent_for)
    assert isinstance(refs, ReferenceSet)
    assert len(refs.image_paths) == 1


def test_multiple_images_are_accepted_up_to_the_profile_limit(
    tmp_path, stub_adapter, sample_images, sample_waveform, consent_for
):
    limit = stub_adapter.profile.reference_limits.accepted["image"]
    refs = _stage(
        tmp_path, stub_adapter.profile, sample_images[:limit], sample_waveform, consent_for
    )
    assert len(refs.image_paths) == limit


def test_exceeding_the_profile_image_limit_is_a_reference_error(
    tmp_path, stub_adapter, sample_images, sample_waveform, consent_for
):
    limit = stub_adapter.profile.reference_limits.accepted["image"]
    too_many = (sample_images * 5)[: limit + 1]
    with pytest.raises(ReferenceError):
        _stage(tmp_path, stub_adapter.profile, too_many, sample_waveform, consent_for)


def test_the_limit_comes_from_the_profile_not_the_application(
    tmp_path, fixture_profile, sample_images, sample_waveform, consent_for
):
    """A different profile must yield a different bound, with no code change."""
    limit = fixture_profile.reference_limits.accepted["image"]
    refs = _stage(
        tmp_path, fixture_profile, (sample_images * 9)[:limit], sample_waveform, consent_for
    )
    assert len(refs.image_paths) == limit


def test_zero_images_is_refused(tmp_path, stub_adapter, sample_waveform, consent_for):
    with pytest.raises(AppError):
        _stage(tmp_path, stub_adapter.profile, [], sample_waveform, consent_for)


def test_each_image_must_contain_exactly_one_face(
    tmp_path, stub_adapter, sample_images, sample_waveform, consent_for
):
    detector = FakeDetector(
        per_path={Path(sample_images[1]).name: FaceResult(face_count=2, has_mouth=True)}
    )
    with pytest.raises(AppError) as excinfo:
        _stage(
            tmp_path,
            stub_adapter.profile,
            sample_images[:2],
            sample_waveform,
            consent_for,
            detector=detector,
        )
    assert "face" in str(excinfo.value).lower()


def test_an_image_with_no_face_is_refused(
    tmp_path, stub_adapter, sample_image, sample_waveform, consent_for
):
    detector = FakeDetector(FaceResult(face_count=0, has_mouth=False))
    with pytest.raises(AppError):
        _stage(
            tmp_path,
            stub_adapter.profile,
            [sample_image],
            sample_waveform,
            consent_for,
            detector=detector,
        )


def test_an_image_with_no_usable_mouth_is_refused(
    tmp_path, stub_adapter, sample_image, sample_waveform, consent_for
):
    detector = FakeDetector(FaceResult(face_count=1, has_mouth=False))
    with pytest.raises(AppError):
        _stage(
            tmp_path,
            stub_adapter.profile,
            [sample_image],
            sample_waveform,
            consent_for,
            detector=detector,
        )


def test_video_references_are_always_rejected(
    tmp_path, stub_adapter, sample_image, sample_waveform, consent_for
):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    with pytest.raises(ReferenceError, match="video"):
        _stage(tmp_path, stub_adapter.profile, [sample_image, video], sample_waveform, consent_for)


def test_rejected_kinds_are_recorded_on_the_set(
    tmp_path, stub_adapter, sample_image, sample_waveform, consent_for
):
    refs = _stage(tmp_path, stub_adapter.profile, [sample_image], sample_waveform, consent_for)
    assert "video" in refs.rejected_kinds


def test_audio_role_is_always_the_timbre_anchor(
    tmp_path, stub_adapter, sample_image, sample_waveform, consent_for
):
    refs = _stage(tmp_path, stub_adapter.profile, [sample_image], sample_waveform, consent_for)
    assert refs.audio_role == "timbre_anchor"


def test_references_are_copied_into_staging_not_referenced_in_place(
    tmp_path, stub_adapter, sample_image, sample_waveform, consent_for
):
    refs = _stage(tmp_path, stub_adapter.profile, [sample_image], sample_waveform, consent_for)
    staged = Path(refs.image_paths[0])
    assert staged.exists()
    assert staged != sample_image
    assert (tmp_path / "work") in staged.parents


def test_source_paths_record_where_each_staged_copy_came_from(
    tmp_path, stub_adapter, sample_image, sample_waveform, consent_for
):
    refs = _stage(tmp_path, stub_adapter.profile, [sample_image], sample_waveform, consent_for)
    assert set(refs.source_paths) == {*refs.image_paths, refs.audio_path}


def test_measurements_are_recorded_per_reference(
    tmp_path, stub_adapter, sample_image, sample_waveform, consent_for
):
    refs = _stage(tmp_path, stub_adapter.profile, [sample_image], sample_waveform, consent_for)
    assert refs.measured
    for entry in refs.measured.values():
        assert "sha256" in entry


def test_audio_shorter_than_the_profile_minimum_is_refused(
    tmp_path, stub_adapter, sample_image, consent_for
):
    import struct as _s
    import wave as _w

    short = tmp_path / "tooshort.wav"
    with _w.open(str(short), "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(8000)
        h.writeframes(_s.pack("<h", 100) * 400)  # 0.05 s
    with pytest.raises(ReferenceError):
        _stage(tmp_path, stub_adapter.profile, [sample_image], short, consent_for)


def test_consent_must_match_the_staged_audio_digest(
    tmp_path, stub_adapter, sample_image, sample_waveform, silent_waveform, consent_for
):
    """Consent bound to a DIFFERENT recording must not authorize this one."""
    from errors import ConsentError

    request_id = uuid.uuid4()
    wrong = consent_for(silent_waveform, request_id=request_id)
    with pytest.raises(ConsentError):
        _stage(
            tmp_path,
            stub_adapter.profile,
            [sample_image],
            sample_waveform,
            consent_for,
            request_id=request_id,
            consent=wrong,
        )
