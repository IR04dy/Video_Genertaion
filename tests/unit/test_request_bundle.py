"""T030: the published manifest must satisfy the contract schema."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "specs/001-generate-image-video/contracts/request-bundle.schema.json"
)


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def test_schema_is_valid_json(schema: dict) -> None:
    assert schema["title"] == "RequestBundleManifest"


def test_required_artifact_kinds_are_all_present(published_bundle: dict) -> None:
    kinds = {a["kind"] for a in published_bundle["artifacts"]}
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


def test_manifest_validates_against_the_contract(published_bundle: dict, schema: dict) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(published_bundle, schema)


def test_multiple_images_are_all_retained(published_bundle_multi: dict) -> None:
    images = [a for a in published_bundle_multi["artifacts"] if a["kind"] == "original_image"]
    assert len(images) > 1


def test_exactly_one_of_each_singleton_kind(published_bundle: dict) -> None:
    from collections import Counter

    counts = Counter(a["kind"] for a in published_bundle["artifacts"])
    for kind in ("reference_audio", "final_mp4", "metadata", "assembled_prompt"):
        assert counts[kind] == 1


def test_every_artifact_path_is_relative_and_contained(published_bundle: dict) -> None:
    from storage import is_safe_relative

    for artifact in published_bundle["artifacts"]:
        assert is_safe_relative(artifact["relative_path"])


def test_every_artifact_carries_a_digest_and_size(published_bundle: dict) -> None:
    for artifact in published_bundle["artifacts"]:
        assert len(artifact["sha256"]) == 64
        assert artifact["size_bytes"] >= 0


def test_consent_is_recorded_and_true(published_bundle: dict) -> None:
    assert published_bundle["consent"]["confirmed"] is True
    assert len(published_bundle["consent"]["reference_audio_sha256"]) == 64


def test_consent_binds_to_this_request(published_bundle: dict) -> None:
    assert published_bundle["consent"]["request_id"] == published_bundle["request_id"]


def test_duration_provenance_is_recorded(published_bundle: dict) -> None:
    """The manifest must show WHERE the duration came from, not just its value."""
    params = published_bundle["parameters"]
    assert params["suggested_duration_seconds"] > 0
    assert "operator_overrode_duration" in params
    assert "speaking_rate_used" in params
    assert params["effective_duration_seconds"] > 0


def test_profile_id_is_recorded_so_measurements_are_attributable(published_bundle: dict) -> None:
    assert published_bundle["parameters"]["profile_id"]


def test_plaintext_disclosure_is_explicit(published_bundle: dict) -> None:
    assert published_bundle["plaintext_sensitive_artifacts"] is True


def test_state_is_complete(published_bundle: dict) -> None:
    assert published_bundle["state"] == "complete"


def test_bundle_relative_path_is_the_request_uuid(published_bundle: dict) -> None:
    expected = f"outputs/{published_bundle['request_id']}"
    assert published_bundle["bundle_relative_path"] == expected
    uuid.UUID(published_bundle["request_id"])


def test_manifest_carries_no_license_data(published_bundle: dict) -> None:
    blob = json.dumps(published_bundle).lower()
    assert "license" not in blob


def test_manifest_carries_no_absolute_paths(published_bundle: dict) -> None:
    blob = json.dumps(published_bundle)
    assert "/Users/" not in blob
    assert "C:\\" not in blob


def test_memory_by_stage_is_present(published_bundle: dict) -> None:
    assert isinstance(published_bundle["memory_by_stage"], dict)
