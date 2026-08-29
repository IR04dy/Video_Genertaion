"""T025: image and audio inspection, normalization, and bounded decode."""

from __future__ import annotations

import struct
import wave
import zlib
from pathlib import Path

import pytest

from errors import AppError
from media import (
    AudioInfo,
    FaceResult,
    ImageInfo,
    inspect_audio,
    inspect_image,
    normalize_image,
    sha256_file,
)


def test_inspect_image_reports_dimensions_and_digest(sample_image: Path) -> None:
    info = inspect_image(sample_image)
    assert isinstance(info, ImageInfo)
    assert info.width > 0 and info.height > 0
    assert len(info.sha256) == 64
    assert info.size_bytes == sample_image.stat().st_size


def test_inspect_image_rejects_a_non_image(tmp_path: Path) -> None:
    bad = tmp_path / "notreally.png"
    bad.write_bytes(b"this is not a png")
    with pytest.raises(AppError):
        inspect_image(bad)


def test_inspect_image_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AppError):
        inspect_image(tmp_path / "absent.png")


def test_decompression_bomb_is_refused_before_decoding(tmp_path: Path) -> None:
    """A 40000x40000 PNG header is ~30 bytes but 1.6 GPixels decoded.

    The bound must be enforced from the HEADER, before any pixel work, or the
    check is useless — the allocation it protects against has already happened.
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    bomb = tmp_path / "bomb.png"
    bomb.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 40000, 40000, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00" * 64, 9))
        + chunk(b"IEND", b"")
    )
    with pytest.raises(AppError, match="large"):
        inspect_image(bomb, max_pixels=8_000_000)


def test_max_pixels_boundary_is_inclusive(sample_image: Path) -> None:
    info = inspect_image(sample_image)
    exact = info.width * info.height
    assert inspect_image(sample_image, max_pixels=exact).width == info.width
    with pytest.raises(AppError):
        inspect_image(sample_image, max_pixels=exact - 1)


def test_normalize_image_writes_rgb_png(sample_image: Path, tmp_path: Path) -> None:
    out = normalize_image(sample_image, tmp_path / "norm.png")
    assert out.exists()
    assert inspect_image(out).format == "PNG"


def test_normalize_is_deterministic(sample_image: Path, tmp_path: Path) -> None:
    a = sha256_file(normalize_image(sample_image, tmp_path / "a.png"))
    b = sha256_file(normalize_image(sample_image, tmp_path / "b.png"))
    assert a == b


def test_inspect_audio_reports_duration_and_rate(sample_waveform: Path) -> None:
    info = inspect_audio(sample_waveform)
    assert isinstance(info, AudioInfo)
    assert info.duration_seconds > 0
    assert info.sample_rate > 0
    assert info.channels >= 1
    assert len(info.sha256) == 64


def test_inspect_audio_measures_the_real_duration(tmp_path: Path) -> None:
    path = tmp_path / "two-seconds.wav"
    rate = 8000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 1000) * (rate * 2))
    assert inspect_audio(path).duration_seconds == pytest.approx(2.0, abs=0.05)


def test_inspect_audio_rejects_a_non_audio_file(tmp_path: Path) -> None:
    bad = tmp_path / "notaudio.wav"
    bad.write_bytes(b"nope")
    with pytest.raises(AppError):
        inspect_audio(bad)


def test_sha256_file_matches_a_known_digest(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "x.bin"
    path.write_bytes(b"hello world")
    assert sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_face_result_is_a_value(sample_image: Path) -> None:
    """Detection is injected, so the offline suite needs no vision model."""
    result = FaceResult(face_count=1, has_mouth=True)
    assert result.face_count == 1 and result.has_mouth is True
