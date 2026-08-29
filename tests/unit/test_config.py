"""T014: settings. Roots are fixed; ceilings and reserve are configurable."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from config import Settings, load_settings

GIB = 1024**3


def test_roots_are_fixed_relative_to_the_project(tmp_path: Path) -> None:
    settings = load_settings(env={}, project_root=tmp_path)
    assert settings.outputs_root == (tmp_path / "outputs").resolve()
    assert settings.model_cache_root == (tmp_path / ".model-cache").resolve()


def test_no_environment_variable_can_move_the_bundle_root(tmp_path: Path) -> None:
    """The single most important assertion in this file."""
    # The literal paths below are the ATTACK, not a temp-file mistake: the test
    # asserts these values are ignored entirely.
    hostile = {
        "APP_OUTPUTS_ROOT": "/tmp/evil",
        "OUTPUTS_ROOT": "/tmp/evil",
        "APP_BUNDLE_ROOT": "/tmp/evil",
        "BUNDLE_ROOT": "/tmp/evil",
        "APP_MODEL_CACHE_ROOT": "/tmp/evil",
    }
    settings = load_settings(env=hostile, project_root=tmp_path)
    assert settings.outputs_root == (tmp_path / "outputs").resolve()
    assert settings.model_cache_root == (tmp_path / ".model-cache").resolve()


def test_settings_expose_no_root_override_field() -> None:
    fields = set(Settings.model_fields)
    assert "bundle_root" not in fields
    for name in fields:
        assert not name.endswith("_root_override")


def test_disk_reserve_defaults_to_ten_gib(tmp_path: Path) -> None:
    assert load_settings(env={}, project_root=tmp_path).disk_reserve_bytes == 10 * GIB


def test_disk_reserve_is_configurable(tmp_path: Path) -> None:
    settings = load_settings(env={"APP_DISK_RESERVE_BYTES": str(2 * GIB)}, project_root=tmp_path)
    assert settings.disk_reserve_bytes == 2 * GIB


def test_both_ceilings_are_present_and_configurable(tmp_path: Path) -> None:
    settings = load_settings(
        env={"APP_MAX_RESERVED_BYTES": "123", "APP_MAX_HOST_RESIDENT_BYTES": "456"},
        project_root=tmp_path,
    )
    assert settings.max_reserved_bytes == 123
    assert settings.max_host_resident_bytes == 456


def test_ceiling_defaults_are_the_documented_ones(tmp_path: Path) -> None:
    settings = load_settings(env={}, project_root=tmp_path)
    assert settings.max_reserved_bytes == int(13.5 * GIB)
    assert settings.max_host_resident_bytes == 64 * GIB


def test_bind_defaults_to_loopback(tmp_path: Path) -> None:
    assert load_settings(env={}, project_root=tmp_path).bind_host == "127.0.0.1"


def test_cancellation_grace_is_configurable(tmp_path: Path) -> None:
    settings = load_settings(env={"APP_CANCELLATION_GRACE_SECONDS": "5"}, project_root=tmp_path)
    assert settings.cancellation_grace_seconds == 5


def test_settings_carry_no_hub_token_field() -> None:
    for name in Settings.model_fields:
        assert "token" not in name.lower()


def test_paths_are_pathlib_and_absolute(tmp_path: Path) -> None:
    settings = load_settings(env={}, project_root=tmp_path)
    for value in (settings.outputs_root, settings.model_cache_root, settings.staging_root):
        assert isinstance(value, Path)
        assert value.is_absolute()


def test_staging_root_is_inside_outputs(tmp_path: Path) -> None:
    settings = load_settings(env={}, project_root=tmp_path)
    assert settings.staging_root.parent == settings.outputs_root
    assert settings.staging_root.name == ".work"


@pytest.mark.parametrize("bad", ["-1", "0", "not-a-number"])
def test_invalid_numeric_settings_are_rejected(tmp_path: Path, bad: str) -> None:
    """A malformed budget must fail loudly rather than fall back to a default."""
    with pytest.raises((ValueError, PydanticValidationError)):
        load_settings(env={"APP_DISK_RESERVE_BYTES": bad}, project_root=tmp_path)
