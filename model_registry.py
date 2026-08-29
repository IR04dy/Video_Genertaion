"""Reviewed adapter registry and profile resolution (T041).

The registry maps an adapter key to a reviewed adapter class. It is deliberately
a closed set: an arbitrary repository cannot introduce executable code by being
downloaded, because nothing here loads a class named by remote metadata.

Profile resolution is where "measured, not assumed" is enforced. A profile
missing any measured capability field is marked incompatible rather than
defaulted, because a default would be this application inventing a model
capability — the exact mistake the CogVideoX constants were.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from domain import ModelProfile
from errors import ModelIncompatibleError

REQUIRED_PROFILE_FIELDS = (
    "duration_range_seconds",
    "frame_rate",
    "resolutions",
    "audio_output",
    "dialogue_languages",
    "reference_limits",
    "prompt_capacity_tokens",
    "dialogue_tag_form",
)
"""Every measured field a profile must carry to be usable. No defaults exist."""


@dataclass(frozen=True)
class AdapterRegistration:
    key: str
    factory: Callable[..., object]
    description: str


_REGISTRY: dict[str, AdapterRegistration] = {}


def register(key: str, factory: Callable[..., object], description: str = "") -> None:
    _REGISTRY[key] = AdapterRegistration(key=key, factory=factory, description=description)


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


def get_adapter(key: str, **kwargs):
    """Instantiate a reviewed adapter by key.

    Refuses unknown keys instead of importing anything named by the caller: an
    adapter arrives by being registered in this process, never by name lookup.
    """
    registration = _REGISTRY.get(key)
    if registration is None:
        raise ModelIncompatibleError(
            f"{key!r} is not a reviewed adapter. "
            f"Available: {', '.join(available_adapters()) or 'none'}"
        )
    adapter = registration.factory(**kwargs)
    # Enforced here rather than trusted: a profile missing a measured field would
    # otherwise reach the engine and be defaulted somewhere downstream.
    validate_profile(adapter.profile)
    return adapter


def validate_profile(profile: ModelProfile) -> ModelProfile:
    """Refuse a profile that has not measured everything it must declare."""
    missing = [
        field
        for field in REQUIRED_PROFILE_FIELDS
        if getattr(profile, field, None) in (None, [], {})
    ]
    if missing:
        raise ModelIncompatibleError(
            f"profile {profile.profile_id!r} is missing measured fields: {', '.join(missing)}. "
            "Measure them on this hardware rather than assuming published values."
        )
    if not profile.resolutions:
        raise ModelIncompatibleError(f"profile {profile.profile_id!r} declares no resolution")
    if not profile.dialogue_languages:
        raise ModelIncompatibleError(f"profile {profile.profile_id!r} declares no language")
    return profile


def _register_builtin() -> None:
    from adapters.stub import StubAdapter

    register("stub", StubAdapter, "Deterministic offline stub; no weights, no network.")


_register_builtin()
