"""Container export and output verification (T042).

`video_path` becomes visible to the operator only after everything here passes.
That ordering is the point: a file that exists is not the same as a file that
contains a video stream, an audio stream, audible speech, and two streams that
agree in length. Showing an unverified render would let a silent or truncated
output look like a success.

All ffmpeg invocations use argument vectors, never shell strings. Filenames come
from operator uploads, so a shell string would be command injection.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from errors import AppError, CodecError, ExportError

SILENCE_FLOOR = 1e-4
"""RMS below which a track counts as silent. An application threshold for the
non-silence check, not a model capability."""


@dataclass(frozen=True)
class MediaProbe:
    has_video: bool
    has_audio: bool
    video_seconds: float
    audio_seconds: float
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    non_silent: bool
    probe: MediaProbe


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - core dependency
        raise AppError(f"ffmpeg support unavailable: {exc}") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_ffmpeg_args(
    *,
    frames_dir: Path | str,
    audio_path: Path | str,
    out_path: Path | str,
    frame_rate: float,
    pattern: str = "%06d.png",
) -> list[str]:
    """The mux command, as a list. Never joined into a shell string."""
    return [
        _ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{frame_rate:g}",
        "-i",
        str(Path(frames_dir) / pattern),
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        # Explicit container. The export writes through a temporary file whose
        # suffix ffmpeg cannot infer a format from, so leaving this out makes the
        # muxer fail on a path that works when the name happens to end in .mp4.
        "-f",
        "mp4",
        str(out_path),
    ]


def export_mp4(
    *,
    frames_dir: Path | str,
    audio_path: Path | str,
    out_path: Path | str,
    frame_rate: float,
    pattern: str = "%06d.png",
) -> Path:
    """Mux frames and audio into an MP4, writing through a temporary file."""
    frames = Path(frames_dir)
    destination = Path(out_path)

    if not frames.is_dir() or not any(frames.glob("*.png")):
        raise ExportError("no frames were produced, so there is nothing to export")
    if not Path(audio_path).is_file():
        raise ExportError("the generated audio track is missing")

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.stem, suffix=".part"
    )
    os.close(handle)
    temporary_path = Path(temporary)

    try:
        result = subprocess.run(  # noqa: S603 - argument vector, never a shell string
            build_ffmpeg_args(
                frames_dir=frames,
                audio_path=audio_path,
                out_path=temporary_path,
                frame_rate=frame_rate,
                pattern=pattern,
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or temporary_path.stat().st_size == 0:
            raise CodecError(f"the encoder failed: {result.stderr.strip()[:400]}")
        os.replace(temporary_path, destination)
    finally:
        # A failed export must leave no .part behind for the disk monitor to
        # trip over, and no half-written file at the destination.
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)

    return destination


def probe_media(path: Path | str) -> MediaProbe:
    """Read stream facts from ffmpeg's report.

    imageio-ffmpeg ships ffmpeg but not ffprobe, so this parses `ffmpeg -i`,
    which writes its stream summary to stderr.
    """
    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        raise ExportError(f"no output file was produced: {target.name}")

    result = subprocess.run(  # noqa: S603 - argument vector
        [_ffmpeg_exe(), "-hide_banner", "-i", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    text = result.stderr
    if "Invalid data" in text or "Duration" not in text:
        raise CodecError(f"{target.name} is not a readable media container")

    seconds = 0.0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Duration:"):
            stamp = stripped.split("Duration:", 1)[1].split(",", 1)[0].strip()
            if stamp and stamp != "N/A":
                hours, minutes, secs = stamp.split(":")
                seconds = int(hours) * 3600 + int(minutes) * 60 + float(secs)

    has_video = "Video:" in text
    has_audio = "Audio:" in text

    width = height = sample_rate = None
    for line in text.splitlines():
        if "Video:" in line:
            for part in line.split(","):
                token = part.strip().split(" ")[0]
                if "x" in token and token.replace("x", "").isdigit():
                    width, height = (int(v) for v in token.split("x"))
                    break
        if "Audio:" in line:
            for part in line.split(","):
                part = part.strip()
                if part.endswith("Hz"):
                    sample_rate = int(part.split()[0])

    return MediaProbe(
        has_video=has_video,
        has_audio=has_audio,
        # The container reports one duration; both streams are muxed with
        # -shortest, so it describes each of them.
        video_seconds=seconds if has_video else 0.0,
        audio_seconds=seconds if has_audio else 0.0,
        width=width,
        height=height,
        sample_rate=sample_rate,
    )


def _decode_to_wav(path: Path, destination: Path) -> Path:
    result = subprocess.run(  # noqa: S603 - argument vector
        [
            _ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-f",
            "wav",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CodecError(f"could not decode the audio track: {result.stderr.strip()[:300]}")
    return destination


def measure_rms(path: Path | str) -> float:
    """Root-mean-square amplitude of a track's audio, in 0.0-1.0."""
    import array
    import math

    with tempfile.TemporaryDirectory() as tmp:
        wav = _decode_to_wav(Path(path), Path(tmp) / "probe.wav")
        with wave.open(str(wav), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            width = handle.getsampwidth()

    if not frames or width != 2:
        return 0.0

    samples = array.array("h")
    samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
    if not samples:
        return 0.0
    total = sum(float(s) * float(s) for s in samples)
    return math.sqrt(total / len(samples)) / 32768.0


def verify_output(
    path: Path | str, *, expected_seconds: float, frame_rate: float
) -> VerificationResult:
    """Gate publication. Every check here must pass before a video is shown."""
    target = Path(path)
    probe = probe_media(target)

    if not probe.has_video:
        raise ExportError("the output has no video stream")
    if not probe.has_audio:
        raise ExportError("the output has no audio stream")

    # Tolerance scales with the profile's frame rate; it is not a constant.
    tolerance = max(1.0 / frame_rate, 0.05)

    if abs(probe.video_seconds - probe.audio_seconds) > tolerance:
        raise ExportError(
            "video and audio duration disagree by more than one frame "
            f"({probe.video_seconds:.3f}s vs {probe.audio_seconds:.3f}s)"
        )
    if abs(probe.video_seconds - expected_seconds) > tolerance:
        raise ExportError(
            f"the output duration is {probe.video_seconds:.3f}s but "
            f"{expected_seconds:.3f}s was requested"
        )

    non_silent = measure_rms(target) > SILENCE_FLOOR
    if not non_silent:
        raise ExportError(
            "the generated speech track is silent. The voice never spoke, so the "
            "result is not usable."
        )

    return VerificationResult(ok=True, non_silent=True, probe=probe)


def publish_atomically(staged: Path | str, destination: Path | str) -> Path:
    """Move a verified file into place in one step.

    `os.replace` is atomic within a filesystem, so no reader ever observes a
    partially written published file. Falls back to copy+replace across devices.
    """
    source, target = Path(staged), Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, target)
    except OSError:
        temporary = target.with_suffix(target.suffix + ".part")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        source.unlink(missing_ok=True)
    return target


def export_video_only(
    *,
    frames_dir: Path | str,
    out_path: Path | str,
    frame_rate: float,
    pattern: str = "%06d.png",
) -> Path:
    """Encode the decoded picture stream with no audio track.

    Retained as the `decoded_video` artifact. It is deliberately NOT a copy of
    the final MP4: an artifact identical to another one records nothing, and this
    is the bundle's largest file to duplicate.
    """
    frames = Path(frames_dir)
    destination = Path(out_path)
    if not frames.is_dir() or not any(frames.glob("*.png")):
        raise ExportError("no frames were produced, so there is nothing to encode")

    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(  # noqa: S603 - argument vector, never a shell string
        [
            _ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{frame_rate:g}",
            "-i",
            str(frames / pattern),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mp4",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CodecError(f"the encoder failed: {result.stderr.strip()[:400]}")
    return destination
