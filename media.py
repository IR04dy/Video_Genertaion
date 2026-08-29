"""Image and audio inspection, normalization, and face detection (T035).

Two design choices worth stating.

**Bounds are enforced from headers, before decoding.** A decompression bomb is a
tiny file that expands to gigapixels; checking its size after decoding is checking
after the damage. `inspect_image` reads dimensions from the header and refuses
before any pixel work happens.

**Face detection is injected.** The offline suite must run with no vision model
and no network, so `FaceDetector` is a protocol and every test passes a fake. The
real detector is chosen at wiring time, not imported here.
"""

from __future__ import annotations

import hashlib
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from errors import AppError, ReferenceError, ValidationError

DEFAULT_MAX_PIXELS = 64_000_000
"""Bound on decoded image size. An application safety limit, not a model value:
no adapter profile declares it and no model capability depends on it."""

_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int
    height: int
    format: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class FaceResult:
    face_count: int
    has_mouth: bool


class FaceDetector(Protocol):
    """Injected so the offline suite needs no vision model."""

    def detect(self, path: Path) -> FaceResult: ...


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise ValidationError(f"file not found: {resolved.name}")
    return resolved


def inspect_image(path: Path | str, *, max_pixels: int = DEFAULT_MAX_PIXELS) -> ImageInfo:
    """Read format and dimensions, refusing oversized images before decoding."""
    resolved = _require_file(path)

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a core dependency
        raise AppError(f"image support unavailable: {exc}") from exc

    # Disable Pillow's own bomb guard so OUR bound is the one that fires. Left
    # enabled, it raises during open() and the operator gets "unreadable image"
    # instead of a message naming the actual limit.
    Image.MAX_IMAGE_PIXELS = None

    try:
        with Image.open(resolved) as handle:
            # .open() parses only the header, so width/height are known here
            # without the pixel data ever being allocated.
            width, height = handle.size
            image_format = handle.format or "UNKNOWN"
    except Exception as exc:
        raise ValidationError(f"not a readable image: {resolved.name}") from exc

    if width * height > max_pixels:
        raise ValidationError(
            f"image is too large: {resolved.name} is {width}x{height}, "
            f"above the {max_pixels} pixel bound"
        )

    return ImageInfo(
        path=resolved,
        width=width,
        height=height,
        format=image_format,
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def normalize_image(source: Path | str, destination: Path | str) -> Path:
    """Apply EXIF orientation, convert to RGB, and write a deterministic PNG.

    Deterministic because the bundle records a digest: the same input must
    produce the same bytes, or provenance checks become noise.
    """
    resolved = _require_file(source)
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageOps

    with Image.open(resolved) as handle:
        oriented = ImageOps.exif_transpose(handle)
        rgb = oriented.convert("RGB")
        rgb.save(out, format="PNG", optimize=False, compress_level=6)
    return out


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - core dependency
        raise AppError(f"ffmpeg support unavailable: {exc}") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffmpeg_audio_info(path: Path) -> tuple[float, int, int] | None:
    """Duration, sample rate, channels via ffmpeg, or None if unreadable."""
    result = subprocess.run(  # noqa: S603 - argument vector, never a shell string
        [_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    text = result.stderr
    if "Audio:" not in text:
        return None

    seconds = 0.0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Duration:"):
            stamp = stripped.split("Duration:", 1)[1].split(",", 1)[0].strip()
            if stamp and stamp != "N/A":
                hours, minutes, secs = stamp.split(":")
                seconds = int(hours) * 3600 + int(minutes) * 60 + float(secs)

    audio_line = next(line for line in text.splitlines() if "Audio:" in line)
    sample_rate = 0
    channels = 1
    for part in audio_line.split(","):
        part = part.strip()
        if part.endswith("Hz"):
            sample_rate = int(part.split()[0])
        elif part == "mono":
            channels = 1
        elif part == "stereo":
            channels = 2
        elif part.endswith("channels"):
            channels = int(part.split()[0])
    return seconds, sample_rate, channels


def inspect_audio(path: Path | str) -> AudioInfo:
    """Measure a recording. WAV is read with the stdlib; anything else via ffmpeg."""
    resolved = _require_file(path)

    duration = sample_rate = channels = None
    audio_format = resolved.suffix.lstrip(".").upper() or "UNKNOWN"

    try:
        with wave.open(str(resolved), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            duration = handle.getnframes() / sample_rate if sample_rate else 0.0
            audio_format = "WAV"
    except (wave.Error, EOFError):
        probed = _ffmpeg_audio_info(resolved)
        if probed is None:
            raise ValidationError(f"not a readable audio file: {resolved.name}") from None
        duration, sample_rate, channels = probed

    if not duration or not sample_rate:
        raise ValidationError(f"audio has no measurable duration: {resolved.name}")

    return AudioInfo(
        path=resolved,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        format=audio_format,
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def looks_like_video(path: Path | str) -> bool:
    """Cheap kind check used to refuse video references.

    Extension first, then an ffmpeg probe, because the refusal must not depend on
    the operator naming the file honestly.
    """
    resolved = Path(path)
    if resolved.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        return True
    if not resolved.is_file():
        return False
    result = subprocess.run(  # noqa: S603 - argument vector
        [_ffmpeg_exe(), "-hide_banner", "-i", str(resolved)],
        capture_output=True,
        text=True,
        check=False,
    )
    return "Video:" in result.stderr and "Audio:" in result.stderr


def check_faces(path: Path | str, detector: FaceDetector) -> FaceResult:
    """Every reference image must independently show one usable face and mouth."""
    resolved = Path(path)
    result = detector.detect(resolved)

    if result.face_count == 0:
        raise ReferenceError(f"no face detected in {resolved.name}")
    if result.face_count > 1:
        raise ReferenceError(
            f"{result.face_count} faces detected in {resolved.name}; exactly one is required"
        )
    if not result.has_mouth:
        raise ReferenceError(f"no usable mouth region in {resolved.name}")
    return result


def detect_retained_origin(path: Path | str, outputs_root: Path | str):
    """Advisory `VoiceOrigin` when an upload resolves inside a published bundle.

    Advisory only: reuse never inherits the earlier consent, which must be
    reconfirmed for the new request.
    """
    import uuid as _uuid

    from domain import VoiceOrigin

    resolved = Path(path).resolve()
    root = Path(outputs_root).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return None

    parts = relative.parts
    if len(parts) < 2:
        return None
    try:
        bundle_id = _uuid.UUID(parts[0])
    except ValueError:
        return None

    return VoiceOrigin(
        bundle_id=bundle_id,
        artifact_relative_path="/".join(parts[1:]),
        artifact_sha256=sha256_file(resolved),
    )
