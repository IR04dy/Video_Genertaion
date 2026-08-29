"""T012: stable error codes, sanitized messages, ordered recovery suggestions."""

from __future__ import annotations

import pytest

from domain import ErrorCode
from errors import (
    AppError,
    CancelledError,
    ConsentError,
    DiskError,
    DurationError,
    HostMemoryError,
    OomError,
    ReferenceError,
    ValidationError,
    sanitize,
    to_detail,
)

REQUIRED_CODES = {
    "validation",
    "consent",
    "face",
    "reference",
    "language",
    "duration",
    "oom",
    "host_memory",
    "disk",
    "generation",
    "export",
    "codec",
    "cancelled",
    "internal",
}


def test_every_required_code_exists() -> None:
    assert {c.value for c in ErrorCode} >= REQUIRED_CODES


def test_codes_are_stable_strings() -> None:
    """Codes cross the UI boundary, so they must be values, not ordinals."""
    for code in ErrorCode:
        assert isinstance(code.value, str) and code.value == code.value.lower()


@pytest.mark.parametrize(
    ("exc_type", "code"),
    [
        (ValidationError, ErrorCode.VALIDATION),
        (ConsentError, ErrorCode.CONSENT),
        (ReferenceError, ErrorCode.REFERENCE),
        (DurationError, ErrorCode.DURATION),
        (OomError, ErrorCode.OOM),
        (HostMemoryError, ErrorCode.HOST_MEMORY),
        (DiskError, ErrorCode.DISK),
        (CancelledError, ErrorCode.CANCELLED),
    ],
)
def test_exception_carries_its_code(exc_type: type[AppError], code: ErrorCode) -> None:
    assert exc_type("boom").code is code


def test_every_app_error_subclass_declares_a_code() -> None:
    def walk(cls: type) -> list[type]:
        subs = cls.__subclasses__()
        return subs + [g for s in subs for g in walk(s)]

    for sub in walk(AppError):
        assert getattr(sub, "code", None) is not None, f"{sub.__name__} has no code"


def test_oom_is_retryable_and_host_memory_is_not() -> None:
    """Retrying an OOM after freeing VRAM can work; a host breach needs a
    different profile, so offering a bare retry would be misleading."""
    assert OomError("x").retryable is True
    assert HostMemoryError("x").retryable is False


def test_suggestions_are_ordered_and_nonempty_for_recoverable_errors() -> None:
    for exc in (OomError("x"), HostMemoryError("x"), DiskError("x")):
        detail = to_detail(exc)
        assert detail.suggestions
        assert detail.suggestions == list(detail.suggestions)


def test_cancelled_offers_no_recovery_suggestions() -> None:
    """Cancellation is the operator's own action, not a fault to recover from."""
    assert to_detail(CancelledError("cancelled")).suggestions == []


def test_to_detail_produces_the_typed_record() -> None:
    detail = to_detail(ValidationError("bad input"))
    assert detail.code is ErrorCode.VALIDATION
    assert detail.message == "bad input"


@pytest.mark.parametrize(
    "raw",
    [
        "/Users/someone/secret/photo.png failed",
        r"C:\Users\someone\voice.wav failed",
        "token hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 rejected",
        "Authorization: Bearer abcdef123456",
    ],
)
def test_sanitize_removes_paths_and_credentials(raw: str) -> None:
    cleaned = sanitize(raw)
    for leak in ("/Users/someone", "someone", "hf_ABCDEF", "abcdef123456"):
        assert leak not in cleaned


def test_sanitize_keeps_the_basename_so_messages_stay_actionable() -> None:
    assert "photo.png" in sanitize("/Users/someone/secret/photo.png failed")


def test_sanitize_is_idempotent() -> None:
    once = sanitize("/Users/someone/secret/photo.png failed")
    assert sanitize(once) == once


def test_error_detail_message_is_sanitized_on_construction() -> None:
    detail = to_detail(ValidationError("/Users/someone/x.png is invalid"))
    assert "/Users/someone" not in detail.message


def test_translate_maps_framework_oom_to_the_oom_code() -> None:
    from errors import translate

    class FakeCudaOom(RuntimeError):
        pass

    err = translate(FakeCudaOom("CUDA out of memory. Tried to allocate 2.00 GiB"), stage="generate")
    assert err.code is ErrorCode.OOM


def test_translate_maps_memory_error_to_host_memory() -> None:
    from errors import translate

    err = translate(MemoryError("cannot allocate"), stage="load_model")
    assert err.code is ErrorCode.HOST_MEMORY


def test_translate_maps_disk_full_to_disk() -> None:
    from errors import translate

    exc = OSError(28, "No space left on device")
    assert translate(exc, stage="export").code is ErrorCode.DISK


def test_translate_preserves_an_already_typed_error() -> None:
    from errors import translate

    original = ConsentError("not confirmed")
    assert translate(original, stage="validate") is original


def test_translate_falls_back_to_internal() -> None:
    from errors import translate

    assert translate(ZeroDivisionError("x"), stage="generate").code is ErrorCode.INTERNAL


def test_translate_never_leaks_the_raw_traceback_into_the_message() -> None:
    from errors import translate

    err = translate(ZeroDivisionError("/Users/someone/x.py line 3"), stage="generate")
    assert "/Users/someone" not in to_detail(err).message
