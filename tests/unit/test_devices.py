"""T010: device resolution, dtype policy, and BOTH memory measurements."""

from __future__ import annotations

import pytest

from devices import (
    accelerator_snapshot,
    host_resident_bytes,
    resolve_device,
    select_dtype,
)
from domain import DeviceKind


class FakeBackends:
    """Stands in for the parts of torch that device resolution consults."""

    def __init__(self, cuda: bool = False, mps: bool = False) -> None:
        self._cuda = cuda
        self._mps = mps

    def cuda_available(self) -> bool:
        return self._cuda

    def mps_available(self) -> bool:
        return self._mps


def test_prefers_cuda_when_available() -> None:
    assert resolve_device(backends=FakeBackends(cuda=True, mps=True)) is DeviceKind.CUDA


def test_falls_back_to_mps_without_cuda() -> None:
    assert resolve_device(backends=FakeBackends(cuda=False, mps=True)) is DeviceKind.MPS


def test_falls_back_to_cpu_without_any_accelerator() -> None:
    assert resolve_device(backends=FakeBackends()) is DeviceKind.CPU


def test_requested_device_is_honoured_when_available() -> None:
    backends = FakeBackends(cuda=True, mps=True)
    assert resolve_device(DeviceKind.MPS, backends=backends) is DeviceKind.MPS


def test_requested_device_falls_back_when_unavailable() -> None:
    """An unavailable request degrades rather than raising: CPU always works."""
    assert resolve_device(DeviceKind.CUDA, backends=FakeBackends()) is DeviceKind.CPU


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        (DeviceKind.CUDA, "bfloat16"),
        (DeviceKind.MPS, "float32"),
        (DeviceKind.CPU, "float32"),
    ],
)
def test_default_dtype_policy(device: DeviceKind, expected: str) -> None:
    assert select_dtype(device) == expected


def test_profile_dtype_policy_overrides_the_default() -> None:
    policy = {DeviceKind.CPU: {"preferred": "bfloat16", "allowed": ["bfloat16"]}}
    assert select_dtype(DeviceKind.CPU, policy=policy) == "bfloat16"


def test_profile_policy_rejects_a_dtype_not_in_allowed() -> None:
    policy = {DeviceKind.CPU: {"preferred": "float16", "allowed": ["float32"]}}
    with pytest.raises(ValueError, match="allowed"):
        select_dtype(DeviceKind.CPU, policy=policy)


def test_snapshot_on_cpu_is_unavailable_with_a_reason() -> None:
    snapshot = accelerator_snapshot(DeviceKind.CPU)
    assert snapshot.available is False
    assert snapshot.unavailable_reason


def test_snapshot_always_reports_host_resident_bytes() -> None:
    """Host RAM is a first-class budget, so it is measured even with no accelerator."""
    snapshot = accelerator_snapshot(DeviceKind.CPU)
    assert snapshot.host_resident_bytes is not None
    assert snapshot.host_resident_bytes > 0


def test_host_resident_bytes_is_positive() -> None:
    assert host_resident_bytes() > 0


def test_snapshot_applies_the_reserved_gate_not_the_allocated_one() -> None:
    """Production gates peak RESERVED bytes; allocated is reported but not gated."""
    snapshot = accelerator_snapshot(
        DeviceKind.CPU,
        max_reserved_bytes=1,
        probe=lambda: {
            "device_name": "Fixture",
            "allocated_bytes": 10,
            "reserved_bytes": 20,
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 20,
            "free_bytes": 100,
            "total_bytes": 200,
        },
    )
    assert snapshot.available is True
    assert snapshot.reserved_gate_passed is False


def test_reserved_gate_passes_under_the_ceiling() -> None:
    snapshot = accelerator_snapshot(
        DeviceKind.CPU,
        max_reserved_bytes=1000,
        probe=lambda: {
            "device_name": "Fixture",
            "allocated_bytes": 10,
            "reserved_bytes": 20,
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 20,
            "free_bytes": 100,
            "total_bytes": 200,
        },
    )
    assert snapshot.reserved_gate_passed is True


def test_no_ceiling_leaves_the_gate_unjudged() -> None:
    snapshot = accelerator_snapshot(
        DeviceKind.CPU,
        max_reserved_bytes=None,
        probe=lambda: {
            "device_name": "Fixture",
            "peak_reserved_bytes": 20,
        },
    )
    assert snapshot.reserved_gate_passed is None
