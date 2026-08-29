"""Settings (T019). Roots are fixed; budgets are configurable.

The bundle root is deliberately not configurable. Every published output lives in
`<project>/outputs/<request-id>` and every staging directory in
`<project>/outputs/.work/<request-id>`. A redirectable output root would turn one
environment variable into a path-containment hole, so no override exists — not a
documented-but-discouraged one, none. `test_config.py` asserts that hostile
`APP_OUTPUTS_ROOT` / `APP_BUNDLE_ROOT` values change nothing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

GIB = 1024**3

# Fixed directory names. Not settings.
OUTPUTS_DIR_NAME = "outputs"
MODEL_CACHE_DIR_NAME = ".model-cache"
STAGING_DIR_NAME = ".work"

DEFAULT_DISK_RESERVE_BYTES = 10 * GIB
DEFAULT_MAX_RESERVED_BYTES = int(13.5 * GIB)
DEFAULT_MAX_HOST_RESIDENT_BYTES = 64 * GIB
DEFAULT_CANCELLATION_GRACE_SECONDS = 30


class Settings(BaseModel):
    """Resolved runtime configuration.

    Carries no Hugging Face token field: the application never reads, stores, or
    forwards one, and a field would invite one to be set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_root: Path
    outputs_root: Path
    staging_root: Path
    model_cache_root: Path

    bind_host: str = "127.0.0.1"
    bind_port: Annotated[int, Field(gt=0, le=65535)] = 7860

    disk_reserve_bytes: Annotated[int, Field(gt=0)] = DEFAULT_DISK_RESERVE_BYTES
    max_reserved_bytes: Annotated[int, Field(gt=0)] = DEFAULT_MAX_RESERVED_BYTES
    max_host_resident_bytes: Annotated[int, Field(gt=0)] = DEFAULT_MAX_HOST_RESIDENT_BYTES

    runtime_profile: str = "default"
    cancellation_grace_seconds: Annotated[int, Field(gt=0)] = DEFAULT_CANCELLATION_GRACE_SECONDS


def _read_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return int(raw)  # ValueError propagates; a malformed budget must not be guessed


def load_settings(
    env: Mapping[str, str] | None = None, *, project_root: Path | str | None = None
) -> Settings:
    """Build settings from the environment.

    `env` is injected rather than read from `os.environ` directly so tests can
    present a hostile environment without mutating the process.
    """
    env = os.environ if env is None else env
    root = Path(project_root or Path(__file__).resolve().parent).resolve()

    outputs_root = (root / OUTPUTS_DIR_NAME).resolve()

    return Settings(
        project_root=root,
        # Not read from `env`. See the module docstring.
        outputs_root=outputs_root,
        staging_root=(outputs_root / STAGING_DIR_NAME),
        model_cache_root=(root / MODEL_CACHE_DIR_NAME).resolve(),
        bind_host=env.get("APP_BIND_HOST", "127.0.0.1"),
        bind_port=_read_int(env, "APP_BIND_PORT", 7860),
        disk_reserve_bytes=_read_int(env, "APP_DISK_RESERVE_BYTES", DEFAULT_DISK_RESERVE_BYTES),
        max_reserved_bytes=_read_int(env, "APP_MAX_RESERVED_BYTES", DEFAULT_MAX_RESERVED_BYTES),
        max_host_resident_bytes=_read_int(
            env, "APP_MAX_HOST_RESIDENT_BYTES", DEFAULT_MAX_HOST_RESIDENT_BYTES
        ),
        runtime_profile=env.get("APP_RUNTIME_PROFILE", "default"),
        cancellation_grace_seconds=_read_int(
            env, "APP_CANCELLATION_GRACE_SECONDS", DEFAULT_CANCELLATION_GRACE_SECONDS
        ),
    )
