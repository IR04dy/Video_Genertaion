"""Device resolution, dtype policy, and both memory measurements (T017).

Torch is imported lazily throughout. The offline test suite must run on machines
where torch is old, broken, or absent — and device resolution itself is pure
policy that only needs two booleans, so it takes them through an injectable
`backends` object rather than reaching for the library.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from domain import DeviceKind, MemoryProfile, MemorySnapshot

# Fallback precision per device when a profile declares no policy of its own.
# These are properties of the BACKENDS, not of any model: MPS has no usable
# bfloat16 path in the supported torch versions, and CPU bfloat16 is slow enough
# to be a worse default than float32. A profile may override either.
_DEFAULT_DTYPE: dict[DeviceKind, str] = {
    DeviceKind.CUDA: "bfloat16",
    DeviceKind.MPS: "float32",
    DeviceKind.CPU: "float32",
}

# Preference order when nothing is requested.
_PREFERENCE: tuple[DeviceKind, ...] = (DeviceKind.CUDA, DeviceKind.MPS, DeviceKind.CPU)


class Backends(Protocol):
    def cuda_available(self) -> bool: ...
    def mps_available(self) -> bool: ...


class _TorchBackends:
    """The real probe. Any failure means 'not available', never an exception."""

    def cuda_available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def mps_available(self) -> bool:
        try:
            import torch

            return bool(torch.backends.mps.is_available())
        except Exception:
            return False


def resolve_device(
    requested: DeviceKind | None = None, *, backends: Backends | None = None
) -> DeviceKind:
    """Resolve CUDA -> MPS -> CPU, honouring a request when it is available.

    An unavailable request degrades to the best available device rather than
    raising: CPU always works, and refusing to start on a machine without an
    accelerator would make the app untestable off the production host.
    """
    probe = backends or _TorchBackends()
    available = {
        DeviceKind.CUDA: probe.cuda_available(),
        DeviceKind.MPS: probe.mps_available(),
        DeviceKind.CPU: True,
    }
    if requested is not None and available.get(requested):
        return requested
    return next(device for device in _PREFERENCE if available[device])


def select_dtype(
    device: DeviceKind, *, policy: dict[DeviceKind, dict[str, Any]] | None = None
) -> str:
    """The effective precision for a device, from the profile's policy if given.

    A policy naming a preferred dtype outside its own allowed list is a defect in
    that profile, so it raises rather than silently picking something else.
    """
    if policy and device in policy:
        entry = policy[device]
        preferred = entry.get("preferred")
        allowed = entry.get("allowed")
        if preferred is None:
            raise ValueError(f"dtype policy for {device.value} declares no preferred dtype")
        if allowed is not None and preferred not in allowed:
            raise ValueError(
                f"dtype policy for {device.value} prefers {preferred!r}, "
                f"which is not in its allowed list {list(allowed)!r}"
            )
        return str(preferred)
    return _DEFAULT_DTYPE[device]


def host_resident_bytes() -> int:
    """Resident set size of this process.

    Host RAM is a first-class budget because layer-wise offload keeps the model
    resident in system memory and streams it to the card, so this is sampled on
    every device — including CPU-only hosts with no accelerator to report on.
    """
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        try:  # POSIX fallback, so a missing psutil does not blind the gate
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB; macOS reports bytes.
            import sys

            return int(usage if sys.platform == "darwin" else usage * 1024)
        except Exception:
            return 0


def _cuda_probe() -> dict[str, Any]:
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "device_name": torch.cuda.get_device_name(0),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def accelerator_snapshot(
    device: DeviceKind,
    *,
    max_reserved_bytes: int | None = None,
    probe: Callable[[], dict[str, Any]] | None = None,
) -> MemorySnapshot:
    """Sample accelerator memory, and always sample host resident memory.

    The gate is applied to peak **reserved** bytes, not peak allocated: the
    allocator's reservation is what actually occupies the card, and gating on
    allocated would pass profiles that in practice exhaust VRAM.
    """
    host = host_resident_bytes()

    if probe is None:
        if device is not DeviceKind.CUDA:
            return MemorySnapshot(
                available=False,
                device_name=None,
                host_resident_bytes=host,
                unavailable_reason=f"no accelerator memory reporting on {device.value}",
            )
        probe = _cuda_probe

    try:
        values = probe() if probe else {}
    except Exception as exc:
        return MemorySnapshot(
            available=False,
            host_resident_bytes=host,
            unavailable_reason=f"accelerator query failed: {type(exc).__name__}",
        )

    peak_reserved = values.get("peak_reserved_bytes")
    gate: bool | None = None
    if max_reserved_bytes is not None and peak_reserved is not None:
        gate = peak_reserved <= max_reserved_bytes

    return MemorySnapshot(
        available=True,
        device_name=values.get("device_name"),
        allocated_bytes=values.get("allocated_bytes"),
        reserved_bytes=values.get("reserved_bytes"),
        peak_allocated_bytes=values.get("peak_allocated_bytes"),
        peak_reserved_bytes=peak_reserved,
        free_bytes=values.get("free_bytes"),
        total_bytes=values.get("total_bytes"),
        host_resident_bytes=host,
        reserved_gate_passed=gate,
        unavailable_reason=None,
    )


@dataclass(frozen=True)
class CeilingVerdict:
    """Why a memory profile was accepted or refused. A value, not an exception.

    Selection tries several profiles and expects most to fail, so a refusal is
    ordinary control flow here rather than an error.
    """

    accepted: bool
    breaches: list[str] = field(default_factory=list)


def check_memory_profile(
    profile: MemoryProfile,
    *,
    max_reserved_bytes: int,
    max_host_resident_bytes: int,
) -> CeilingVerdict:
    """Gate a profile on BOTH ceilings.

    Both parameters are keyword-only and required. A caller that supplies one and
    forgets the other gets a TypeError rather than a half-applied gate, which is
    the failure mode that would let an over-budget profile through.
    """
    breaches: list[str] = []
    if profile.expected_peak_reserved_bytes > max_reserved_bytes:
        breaches.append(
            f"accelerator ceiling: expects {profile.expected_peak_reserved_bytes} bytes "
            f"reserved, ceiling is {max_reserved_bytes}"
        )
    if profile.expected_host_resident_bytes > max_host_resident_bytes:
        breaches.append(
            f"host ceiling: expects {profile.expected_host_resident_bytes} bytes "
            f"resident, ceiling is {max_host_resident_bytes}"
        )
    return CeilingVerdict(accepted=not breaches, breaches=breaches)


def select_memory_profile(
    candidates: Sequence[MemoryProfile] | Iterable[MemoryProfile],
    *,
    max_reserved_bytes: int,
    max_host_resident_bytes: int,
) -> MemoryProfile | None:
    """First candidate clearing both ceilings, or None.

    Order is the adapter's declared preference — typically cheapest offload
    first — so the first fit is the best fit rather than an arbitrary one.
    """
    for candidate in candidates:
        verdict = check_memory_profile(
            candidate,
            max_reserved_bytes=max_reserved_bytes,
            max_host_resident_bytes=max_host_resident_bytes,
        )
        if verdict.accepted:
            return candidate
    return None
