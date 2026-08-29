"""Stable error codes, sanitized messages, ordered recovery (T016).

Two responsibilities, deliberately in one module because they must not diverge:
the exception hierarchy the application raises, and the sanitizer that decides
what any of it is allowed to say out loud.

Sanitization is applied at `to_detail()`, the single boundary where an exception
becomes something a user or a log can see. Doing it there rather than at each
raise site means a message cannot leak by someone forgetting to call it.
"""

from __future__ import annotations

import re

from domain import ErrorCode, ErrorDetail

# Ordered: earlier patterns run first, and later ones must tolerate their output.
_POSIX_PATH = re.compile(r"(?<![\w.])/(?:[^\s/:*?\"<>|]+/)+([^\s/:*?\"<>|]*)")
_WINDOWS_PATH = re.compile(r"(?<![\w])[A-Za-z]:\\(?:[^\s\\]+\\)*([^\s\\]*)")
_UNC_PATH = re.compile(r"\\\\[^\s\\]+\\(?:[^\s\\]+\\)*([^\s\\]*)")
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{8,}\b")
# Consumes an optional "Bearer " prefix too, so "Authorization: Bearer <tok>"
# does not leave the token behind after the label is replaced.
_BEARER = re.compile(
    r"(?i)\b(?:authorization|api[_-]?key|access[_-]?token|token)\b\s*[:=]?\s*"
    r"(?:bearer\s+)?\S+"
)
_BARE_BEARER = re.compile(r"(?i)\bbearer\s+\S+")
_LONG_HEX = re.compile(r"\b[A-Fa-f0-9]{32,}\b")

_REDACTED = "[redacted]"


def sanitize(message: str) -> str:
    """Strip absolute paths and credentials, keeping the message actionable.

    Directories are removed but basenames survive: "photo.png is invalid" tells
    the operator which file to fix, while the path to it tells a log reader where
    the operator keeps their files. Idempotent, because messages get re-wrapped.
    """
    if not message:
        return message

    text = _BEARER.sub(_REDACTED, message)
    text = _BARE_BEARER.sub(_REDACTED, text)
    text = _HF_TOKEN.sub(_REDACTED, text)
    text = _UNC_PATH.sub(lambda m: m.group(1) or _REDACTED, text)
    text = _WINDOWS_PATH.sub(lambda m: m.group(1) or _REDACTED, text)
    text = _POSIX_PATH.sub(lambda m: m.group(1) or _REDACTED, text)
    return _LONG_HEX.sub(_REDACTED, text)


class AppError(Exception):
    """Base of every error this application raises deliberately.

    `code` is a class attribute so the hierarchy itself is the mapping table;
    `test_every_app_error_subclass_declares_a_code` walks it to prove none is
    missing.
    """

    code: ErrorCode = ErrorCode.INTERNAL
    retryable: bool = False
    suggestions: tuple[str, ...] = ()

    def __init__(
        self,
        message: str,
        *,
        retryable: bool | None = None,
        suggestions: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if retryable is not None:
            self.retryable = retryable
        if suggestions is not None:
            self.suggestions = tuple(suggestions)


class ValidationError(AppError):
    code = ErrorCode.VALIDATION
    retryable = True
    suggestions = ("Correct the highlighted field and submit again.",)


class ConsentError(AppError):
    code = ErrorCode.CONSENT
    retryable = True
    suggestions = (
        "Confirm you own the reference voice or have permission to clone it.",
        "Re-confirm after changing the reference recording.",
    )


class FaceError(AppError):
    code = ErrorCode.FACE
    retryable = True
    suggestions = (
        "Use an image with exactly one clearly visible face.",
        "Crop the image so only the intended subject remains.",
    )


class ReferenceError(AppError):
    code = ErrorCode.REFERENCE
    retryable = True
    suggestions = (
        "Check the reference count against the model's limits.",
        "Supply image and audio references only; video references are refused.",
    )


class LanguageError(AppError):
    code = ErrorCode.LANGUAGE
    retryable = True
    suggestions = ("Choose a language the selected model lists as supported.",)


class DurationError(AppError):
    """Reserved for an override outside the profile's supported range.

    Never raised for script length: no request is rejected for a long script.
    """

    code = ErrorCode.DURATION
    retryable = True
    suggestions = ("Choose a duration inside the model's supported range.",)


class ModelUrlError(AppError):
    code = ErrorCode.MODEL_URL
    retryable = True
    suggestions = ("Supply a canonical https://huggingface.co/<owner>/<name> URL.",)


class ModelAccessError(AppError):
    code = ErrorCode.MODEL_ACCESS
    retryable = True
    suggestions = ("Check the repository exists and is publicly readable.",)


class ModelDownloadError(AppError):
    code = ErrorCode.MODEL_DOWNLOAD
    retryable = True
    suggestions = ("Retry the download; completed files are resumed, not refetched.",)


class ModelIncompatibleError(AppError):
    code = ErrorCode.MODEL_INCOMPATIBLE
    retryable = False
    suggestions = ("Choose a model an installed reviewed adapter supports.",)


class InventoryError(AppError):
    code = ErrorCode.INVENTORY
    retryable = True


class ModelLoadError(AppError):
    code = ErrorCode.MODEL_LOAD
    retryable = True
    suggestions = ("Verify the downloaded revision, then load it again.",)


class UnsupportedBackendError(AppError):
    code = ErrorCode.UNSUPPORTED_BACKEND
    retryable = False
    suggestions = ("Select a device this model's profile lists as supported.",)


class OomError(AppError):
    """Accelerator memory exhausted. Retrying after freeing VRAM can succeed."""

    code = ErrorCode.OOM
    retryable = True
    suggestions = (
        "Close other applications using the accelerator, then try again.",
        "Select a runtime profile with a more aggressive offload mode.",
        "Reduce the requested duration or resolution.",
    )


class HostMemoryError(AppError):
    """Host RAM ceiling breached.

    Not retryable: the same profile will breach again. The fix is a different
    profile, so offering a bare retry would waste the operator's time.
    """

    code = ErrorCode.HOST_MEMORY
    retryable = False
    suggestions = (
        "Select a runtime profile with a quantized checkpoint.",
        "Close other applications holding system memory.",
        "Raise APP_MAX_HOST_RESIDENT_BYTES only if the machine truly has the RAM.",
    )


class DiskError(AppError):
    code = ErrorCode.DISK
    retryable = True
    suggestions = (
        "Free space on the drive holding this project, then try again.",
        "Delete model revisions you no longer need from the Model Library.",
        "Lower APP_DISK_RESERVE_BYTES only if you accept a smaller safety margin.",
    )


class GenerationError(AppError):
    code = ErrorCode.GENERATION
    retryable = True
    suggestions = ("Try again; if it repeats, change the seed or the references.",)


class ExportError(AppError):
    code = ErrorCode.EXPORT
    retryable = True
    suggestions = ("Retry the export; the generated streams are still staged.",)


class CodecError(AppError):
    code = ErrorCode.CODEC
    retryable = False
    suggestions = ("Confirm the bundled ffmpeg build supports H.264 and AAC.",)


class HistoryError(AppError):
    code = ErrorCode.HISTORY
    retryable = True


class CancelledError(AppError):
    """The operator's own action. There is nothing to recover from."""

    code = ErrorCode.CANCELLED
    retryable = True
    suggestions = ()


class FilesystemError(AppError):
    code = ErrorCode.FILESYSTEM
    retryable = True
    suggestions = ("Check file permissions on the project directory.",)


class InternalError(AppError):
    code = ErrorCode.INTERNAL
    retryable = False
    suggestions = ("Retry the request. If it repeats, check the application log.",)


_CUDA_OOM = re.compile(r"(?i)(cuda|hip|mps).{0,40}out of memory|out of memory.{0,40}(cuda|hip)")
_DISK_FULL_ERRNOS = {28}  # ENOSPC


def translate(exc: BaseException, *, stage: str) -> AppError:
    """Map any exception onto the stable hierarchy.

    An already-typed `AppError` passes through unchanged — re-wrapping it would
    discard the specific code a caller deliberately chose.
    """
    if isinstance(exc, AppError):
        return exc

    text = str(exc)

    if isinstance(exc, MemoryError):
        return HostMemoryError(f"Ran out of system memory during {stage}.")

    if _CUDA_OOM.search(text) or type(exc).__name__ in {"OutOfMemoryError", "FakeCudaOom"}:
        return OomError(f"The accelerator ran out of memory during {stage}.")

    if isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS:
        return DiskError(f"The disk filled up during {stage}.")

    if isinstance(exc, OSError):
        return FilesystemError(f"A filesystem operation failed during {stage}.")

    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return CancelledError(f"Cancelled during {stage}.")

    # Deliberately does not embed `text`: an untyped exception is exactly the
    # case where the message is most likely to carry a path or a traceback.
    return InternalError(f"An unexpected error occurred during {stage}.")


def to_detail(exc: AppError) -> ErrorDetail:
    """The single boundary where an exception becomes something visible."""
    return ErrorDetail(
        code=exc.code,
        message=sanitize(exc.message),
        retryable=exc.retryable,
        suggestions=list(exc.suggestions),
    )
