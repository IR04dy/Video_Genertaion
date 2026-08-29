"""Request field validation (T028's implementation half).

Nothing here knows any model's values. Every bound is read from the profile
passed in, which is what lets the same functions gate two profiles with entirely
different capabilities.
"""

from __future__ import annotations

import math
import secrets

from domain import ModelProfile
from errors import LanguageError, ValidationError

SEED_MAX = 2**63 - 1
"""Bound on the seed field itself, not a model capability."""


def validate_language(language: str, *, profile: ModelProfile) -> str:
    if language not in profile.dialogue_languages:
        raise LanguageError(
            f"{language!r} is not a supported language for this model. "
            f"Supported: {', '.join(profile.dialogue_languages)}"
        )
    return language


def validate_seed(seed: int | None) -> int:
    """Return the effective seed, generating one when absent.

    `bool` is excluded explicitly: it is a subclass of `int`, and `True` silently
    becoming seed 1 would be a confusing way to lose reproducibility.
    """
    if seed is None:
        return secrets.randbelow(SEED_MAX)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError("seed must be an integer")
    if not 0 <= seed <= SEED_MAX:
        raise ValidationError(f"seed must be between 0 and {SEED_MAX}")
    return seed


def validate_guidance(value: float | None, *, profile: ModelProfile) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("guidance scale must be a number")
    if not math.isfinite(value):
        raise ValidationError("guidance scale must be a finite number")

    bounds = profile.input_contract.get("guidance_scale") if profile.input_contract else None
    if isinstance(bounds, dict):
        low, high = bounds.get("min"), bounds.get("max")
        if low is not None and value < low:
            raise ValidationError(f"guidance scale must be at least {low}")
        if high is not None and value > high:
            raise ValidationError(f"guidance scale must be at most {high}")
    return float(value)
