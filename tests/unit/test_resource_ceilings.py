"""T011: a profile must clear BOTH ceilings, and declare rather than discover."""

from __future__ import annotations

import pytest

from devices import CeilingVerdict, check_memory_profile, select_memory_profile
from domain import MemoryProfile, OffloadMode

CEILINGS = {"max_reserved_bytes": 1000, "max_host_resident_bytes": 5000}


def profile(peak: int, host: int, **kw) -> MemoryProfile:
    return MemoryProfile(
        offload_mode=kw.get("offload_mode", OffloadMode.NONE),
        quantization=kw.get("quantization"),
        expected_peak_reserved_bytes=peak,
        expected_host_resident_bytes=host,
    )


def test_profile_within_both_ceilings_is_accepted() -> None:
    verdict = check_memory_profile(profile(900, 4000), **CEILINGS)
    assert verdict.accepted is True
    assert verdict.breaches == []


def test_profile_breaching_the_accelerator_ceiling_is_rejected() -> None:
    verdict = check_memory_profile(profile(1001, 4000), **CEILINGS)
    assert verdict.accepted is False
    assert "accelerator" in " ".join(verdict.breaches)


def test_profile_breaching_the_host_ceiling_is_rejected() -> None:
    """Satisfying the VRAM gate is not sufficient; host RAM is co-equal."""
    verdict = check_memory_profile(profile(900, 5001), **CEILINGS)
    assert verdict.accepted is False
    assert "host" in " ".join(verdict.breaches)


def test_profile_breaching_both_reports_both() -> None:
    verdict = check_memory_profile(profile(1001, 5001), **CEILINGS)
    assert verdict.accepted is False
    assert len(verdict.breaches) == 2


def test_ceilings_at_exactly_the_limit_are_accepted() -> None:
    verdict = check_memory_profile(profile(1000, 5000), **CEILINGS)
    assert verdict.accepted is True


def test_verdict_is_a_value_not_an_exception() -> None:
    assert isinstance(check_memory_profile(profile(9999, 9999), **CEILINGS), CeilingVerdict)


def test_selection_picks_the_first_profile_clearing_both_ceilings() -> None:
    candidates = [
        profile(5000, 9000, offload_mode=OffloadMode.NONE),
        profile(900, 4000, offload_mode=OffloadMode.LAYER_WISE, quantization="fx-int4"),
    ]
    chosen = select_memory_profile(candidates, **CEILINGS)
    assert chosen is candidates[1]


def test_selection_returns_none_when_nothing_fits() -> None:
    assert select_memory_profile([profile(9999, 9999)], **CEILINGS) is None


def test_offload_and_quantization_come_from_the_profile() -> None:
    """Never probed, never inferred from the host — read off the declaration."""
    chosen = select_memory_profile(
        [profile(900, 4000, offload_mode=OffloadMode.SEQUENTIAL_CPU, quantization="fx-int8")],
        **CEILINGS,
    )
    assert chosen is not None
    assert chosen.offload_mode is OffloadMode.SEQUENTIAL_CPU
    assert chosen.quantization == "fx-int8"


def test_check_requires_both_ceilings_to_be_supplied() -> None:
    """A caller that forgets one ceiling must fail loudly, not silently half-gate."""
    with pytest.raises(TypeError):
        check_memory_profile(profile(900, 4000), max_reserved_bytes=1000)  # type: ignore[call-arg]
