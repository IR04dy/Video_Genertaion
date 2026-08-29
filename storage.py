"""Path containment, atomic manifest writes, disk estimation (T020).

This module is a security boundary. Every path that reaches the filesystem passes
through `contain()`, and every relative path that reaches a manifest passes
through `is_safe_relative()`.

The two are separate on purpose. `contain()` answers "does this resolve inside
the root on THIS machine", which requires touching the filesystem. `is_safe_relative()`
answers "could this string ever be dangerous on ANY platform", which must not:
a manifest written on macOS is read on Windows, so a POSIX-legal `a\\b` or
`f.txt:stream` has to be refused at write time, before it becomes an NTFS
alternate data stream on the machine that opens it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from errors import DiskError, FilesystemError, ValidationError

STAGING_DIR_NAME = ".work"

# Refused in relative paths regardless of the host platform.
_WINDOWS_RESERVED = re.compile(r"(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)")
_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")


def is_safe_relative(value: str | os.PathLike[str]) -> bool:
    """Whether a string is safe to store as a bundle-relative path anywhere.

    Refuses: absolute paths, drive letters, UNC roots, backslashes, `..`
    components, empty or dot-only paths, NTFS alternate-data-stream colons, and
    Windows reserved device names.
    """
    text = str(value)

    if not text or text in {".", ".."}:
        return False
    if text.startswith(("/", "\\")):
        return False
    if _DRIVE_LETTER.match(text):
        return False
    if "\\" in text or ":" in text:
        return False
    if text.endswith("/"):
        return False

    parts = PurePosixPath(text).parts
    if not parts:
        return False
    for part in parts:
        if part in {"", ".", ".."}:
            return False
        if part != part.strip() or part.endswith("."):
            return False
        if _WINDOWS_RESERVED.match(part):
            return False
    return True


def _has_link_component(root: Path, target: Path) -> bool:
    """Whether any component between `root` and `target` is a link.

    Walks the **unresolved** path on purpose. Resolving first would follow the
    link and erase the evidence: `root/link/f.txt` resolves to `root/real/f.txt`,
    which contains no link component at all. Descending component by component is
    the only way to see the link that was actually traversed.
    """
    try:
        relative = target.relative_to(root)
    except ValueError:
        return False

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
        # Windows reparse points that are not symlinks (junctions, mount points).
        with contextlib.suppress(OSError, AttributeError):
            if os.name == "nt" and bool(
                current.lstat().st_file_attributes & 0x400  # FILE_ATTRIBUTE_REPARSE_POINT
            ):
                return True
    return False


def contain(root: Path | str, candidate: Path | str, *, allow_links: bool = True) -> Path:
    """Resolve `candidate` and prove it lies inside `root`.

    `allow_links=False` additionally refuses any link component even when it
    resolves inside the root, because a link that points inside today can be
    repointed outside tomorrow — the managed roots use it.
    """
    root_path = Path(root).resolve()
    raw = Path(candidate)
    target = raw if raw.is_absolute() else root_path / raw

    try:
        resolved = target.resolve()
    except (OSError, RuntimeError) as exc:
        raise FilesystemError(f"could not resolve {raw.name}") from exc

    if resolved != root_path and root_path not in resolved.parents:
        raise ValidationError(f"path escapes the permitted root: {raw.name}")

    # Checked against the pre-resolution path; see _has_link_component.
    if not allow_links and _has_link_component(root_path, target):
        raise ValidationError(f"links are not permitted under this root: {raw.name}")

    return resolved


def _require_uuid(request_id: uuid.UUID | str) -> uuid.UUID:
    if isinstance(request_id, uuid.UUID):
        return request_id
    return uuid.UUID(str(request_id))  # raises ValueError on anything else


def staging_dir(outputs_root: Path | str, request_id: uuid.UUID | str) -> Path:
    """`outputs/.work/<request-id>`, named by a server-generated UUID."""
    identifier = _require_uuid(request_id)
    return Path(outputs_root).resolve() / STAGING_DIR_NAME / str(identifier)


def bundle_dir(outputs_root: Path | str, request_id: uuid.UUID | str) -> Path:
    """`outputs/<request-id>`. Never collides with staging: `.work` is not a UUID."""
    identifier = _require_uuid(request_id)
    return Path(outputs_root).resolve() / str(identifier)


@contextlib.contextmanager
def file_lock(path: Path, *, timeout: float = 30.0) -> Iterator[None]:
    """Cross-platform advisory lock around a critical section."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from filelock import FileLock, Timeout

        lock = FileLock(str(path) + ".lock", timeout=timeout)
        try:
            with lock:
                yield
        except Timeout as exc:
            raise FilesystemError("another operation holds the lock") from exc
    except ImportError:
        # filelock is a runtime dependency; the offline suite runs without it.
        yield


def atomic_write_json(path: Path, payload: dict[str, Any], *, lock: bool = True) -> Path:
    """Write JSON via a temporary file plus atomic replace.

    A partially written manifest is indistinguishable from a corrupt bundle, so
    the file must appear complete or not at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _write() -> None:
        # delete=False and a manual close: os.replace needs the path after the
        # handle is closed, which a plain context manager would not survive.
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle as stream:
                json.dump(payload, stream, indent=2, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(handle.name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(handle.name)
            raise

    if lock:
        with file_lock(path):
            _write()
    else:
        _write()
    return path


def free_bytes(path: Path | str) -> int:
    return shutil.disk_usage(Path(path)).free


def check_free_space(path: Path | str, *, required_bytes: int, reserve_bytes: int) -> None:
    """Raise `DiskError` unless the write fits and still leaves the reserve.

    Called at preflight and again periodically during every write stage: a run
    that starts with room can still fill the disk hours later.
    """
    available = free_bytes(path)
    if available < required_bytes + reserve_bytes:
        raise DiskError(
            f"needs {required_bytes} bytes plus a {reserve_bytes} byte reserve, "
            f"but only {available} bytes are free"
        )


def directory_size(path: Path | str) -> int:
    total = 0
    for entry in Path(path).rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            total += entry.stat().st_size
    return total


# --------------------------------------------------------------------------
# Bundle publication (T043)
# --------------------------------------------------------------------------


def publish_bundle(
    *,
    outputs_root: Path | str,
    request_id: uuid.UUID,
    staging: Path | str,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically publish a staged bundle to `outputs/<request-id>`.

    The manifest is written INSIDE staging and the whole directory is then moved
    into place with a single rename. A reader therefore sees either no bundle or
    a complete one — never a directory being filled in. Publishing the manifest
    after the move would leave exactly that window open.
    """
    root = Path(outputs_root).resolve()
    staged = Path(staging).resolve()
    destination = bundle_dir(root, request_id)

    if destination.exists():
        raise FilesystemError(f"a bundle already exists for request {request_id}")

    manifest_path = staged / "metadata.json"
    atomic_write_json(manifest_path, manifest, lock=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, destination)

    return destination, destination / "metadata.json"


def inventory_artifacts(bundle: Path | str, *, kinds: dict[str, str]) -> list[dict[str, Any]]:
    """Describe every retained file, relative to its bundle.

    `kinds` maps a relative path to its `ArtifactKind`. Paths are recorded with
    forward slashes so a manifest written on Windows validates on POSIX.
    """
    root = Path(bundle).resolve()
    records: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        kind = kinds.get(relative)
        if kind is None:
            continue
        records.append(
            {
                "kind": kind,
                "relative_path": relative,
                "media_type": _media_type_for(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "created_by_stage": _STAGE_FOR_KIND.get(kind, "metadata"),
            }
        )
    return records


_STAGE_FOR_KIND = {
    "original_image": "prepare_references",
    "reference_audio": "prepare_references",
    "derived_voice": "generate",
    "assembled_prompt": "assemble_prompt",
    "decoded_video": "decode",
    "decoded_audio": "decode",
    "final_mp4": "export",
    "metadata": "metadata",
}

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".json": "application/json",
    ".txt": "text/plain",
}


def _media_type_for(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def remove_staging(staging: Path | str) -> None:
    """Remove a staging directory, tolerating a partly built tree.

    Never raises: cleanup failing must not replace the error that caused it.
    """
    try:
        shutil.rmtree(Path(staging), ignore_errors=True)
    except Exception:
        # Cleanup must not mask the failure that triggered it, but a staging
        # directory that cannot be removed is worth a trace: it is the leak the
        # startup orphan sweep will later have to deal with.
        logging.getLogger(__name__).debug("could not remove staging directory", exc_info=True)
