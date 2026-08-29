"""T033: the P1 independent test.

One full pass with the stub profile — no network, no accelerator, no weights —
publishing a complete bundle and probing the real MP4 that comes out. If this
passes, the P1 workflow works end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain import RequestState


@pytest.fixture
def completed(tmp_path, stub_adapter, sample_images, sample_waveform, engine_for):
    engine = engine_for(stub_adapter.profile, tmp_path)
    result = engine.run(
        image_paths=sample_images[:2],
        audio_path=sample_waveform,
        motion_prompt="a slow zoom toward the subject",
        speech_script="hello there, this is a test of the joint pipeline",
        language="Testish",
        consent_confirmed=True,
        seed=1234,
    )
    assert result.state is RequestState.COMPLETE, result.error
    return result


def test_run_completes(completed) -> None:
    assert completed.state is RequestState.COMPLETE
    assert completed.error is None


def test_final_mp4_exists_and_is_a_container(completed) -> None:
    video = Path(completed.video_path)
    assert video.exists() and video.stat().st_size > 0
    assert video.read_bytes()[4:8] == b"ftyp"


def test_output_has_one_video_and_one_non_silent_audio_stream(completed) -> None:
    from export import probe_media

    probe = probe_media(Path(completed.video_path))
    assert probe.has_video and probe.has_audio


def test_streams_agree_within_one_frame(completed, stub_adapter) -> None:
    from export import probe_media

    probe = probe_media(Path(completed.video_path))
    tolerance = 1.0 / stub_adapter.profile.frame_rate
    assert abs(probe.video_seconds - probe.audio_seconds) <= tolerance * 2


def test_bundle_contains_every_required_artifact(completed) -> None:
    manifest = json.loads(Path(completed.manifest_path).read_text())
    kinds = {a["kind"] for a in manifest["artifacts"]}
    assert kinds >= {
        "original_image",
        "reference_audio",
        "derived_voice",
        "assembled_prompt",
        "decoded_video",
        "decoded_audio",
        "final_mp4",
        "metadata",
    }


def test_both_source_images_are_retained(completed) -> None:
    manifest = json.loads(Path(completed.manifest_path).read_text())
    images = [a for a in manifest["artifacts"] if a["kind"] == "original_image"]
    assert len(images) == 2


def test_every_manifest_artifact_exists_on_disk(completed) -> None:
    bundle = Path(completed.bundle_path)
    manifest = json.loads(Path(completed.manifest_path).read_text())
    for artifact in manifest["artifacts"]:
        assert (bundle / artifact["relative_path"]).exists(), artifact["relative_path"]


def test_recorded_digests_match_the_retained_bytes(completed) -> None:
    from media import sha256_file

    bundle = Path(completed.bundle_path)
    manifest = json.loads(Path(completed.manifest_path).read_text())
    for artifact in manifest["artifacts"]:
        path = bundle / artifact["relative_path"]
        assert sha256_file(path) == artifact["sha256"], artifact["relative_path"]


def test_bundle_is_published_under_the_request_uuid(completed) -> None:
    assert Path(completed.bundle_path).name == str(completed.request_id)


def test_no_staging_directory_survives(completed, tmp_path) -> None:
    work = tmp_path / "outputs" / ".work"
    assert not work.exists() or list(work.iterdir()) == []


def test_assembled_prompt_is_retained_and_carries_the_script(completed) -> None:
    bundle = Path(completed.bundle_path)
    manifest = json.loads(Path(completed.manifest_path).read_text())
    entry = next(a for a in manifest["artifacts"] if a["kind"] == "assembled_prompt")
    text = (bundle / entry["relative_path"]).read_text()
    assert "hello there, this is a test of the joint pipeline" in text


def test_reference_audio_is_retained_but_never_became_the_output(completed) -> None:
    """The anchor is kept for provenance; it must not be the speech track."""
    from media import sha256_file

    bundle = Path(completed.bundle_path)
    manifest = json.loads(Path(completed.manifest_path).read_text())
    ref = next(a for a in manifest["artifacts"] if a["kind"] == "reference_audio")
    decoded = next(a for a in manifest["artifacts"] if a["kind"] == "decoded_audio")
    assert sha256_file(bundle / ref["relative_path"]) != sha256_file(
        bundle / decoded["relative_path"]
    )


def test_decoded_video_is_not_a_copy_of_the_final_mp4(completed) -> None:
    """Distinct artifacts must have distinct bytes.

    `decoded_video` is the picture stream before muxing; if it ever equals
    `final_mp4` the artifact records nothing and the bundle carries its largest
    file twice.
    """
    from export import probe_media
    from media import sha256_file

    bundle = Path(completed.bundle_path)
    manifest = json.loads(Path(completed.manifest_path).read_text())
    decoded = next(a for a in manifest["artifacts"] if a["kind"] == "decoded_video")
    final = next(a for a in manifest["artifacts"] if a["kind"] == "final_mp4")

    assert decoded["relative_path"] != final["relative_path"]
    assert sha256_file(bundle / decoded["relative_path"]) != sha256_file(
        bundle / final["relative_path"]
    )

    # And it is genuinely picture-only.
    probe = probe_media(bundle / decoded["relative_path"])
    assert probe.has_video is True
    assert probe.has_audio is False


def test_manifest_records_duration_provenance(completed) -> None:
    manifest = json.loads(Path(completed.manifest_path).read_text())
    params = manifest["parameters"]
    assert params["suggested_duration_seconds"] > 0
    assert params["effective_duration_seconds"] > 0
    assert params["operator_overrode_duration"] is False


def test_retained_bytes_matches_the_directory(completed) -> None:
    from storage import directory_size

    assert completed.retained_bytes == directory_size(Path(completed.bundle_path))


def test_manifest_validates_against_the_contract_schema(completed) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "specs/001-generate-image-video/contracts/request-bundle.schema.json"
    )
    manifest = json.loads(Path(completed.manifest_path).read_text())
    jsonschema.validate(manifest, json.loads(schema_path.read_text()))
