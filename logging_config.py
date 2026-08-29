"""Redacted structured logging (T018).

The redaction is a filter on the logging pipeline rather than a helper call sites
must remember, because "remember to sanitize" fails exactly once and then the
prompt is in the log forever.

Never logged: repository tokens, raw prompts and scripts, absolute upload paths,
and any derived voice representation.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from errors import sanitize

# Field names whose VALUES are dropped entirely rather than sanitized. Sanitizing
# a prompt would still leave the prompt; these are replaced wholesale.
_DROPPED_FIELDS = frozenset(
    {
        "prompt",
        "motion_prompt",
        "speech_script",
        "script",
        "rendered",
        "dialogue",
        "dialogue_segments",
        "transcript",
        "reference_transcript",
        "token",
        "hf_token",
        "authorization",
        "api_key",
        "voice",
        "derived_voice",
        "waveform",
        "audio_bytes",
        "upload_path",
        "absolute_path",
        "source_path",
    }
)

_REDACTED = "[redacted]"


def _scrub(value: Any, *, key: str | None = None) -> Any:
    if key is not None and key.lower() in _DROPPED_FIELDS:
        return _REDACTED
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, dict):
        return {k: _scrub(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


class RedactionFilter(logging.Filter):
    """Scrubs the message and every structured field on the way out."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _scrub(record.msg)
        if isinstance(record.args, dict):
            record.args = _scrub(record.args)
        elif record.args:
            record.args = tuple(_scrub(a) for a in record.args)
        for key, value in list(vars(record).items()):
            if key in _RESERVED:
                continue
            setattr(record, key, _scrub(value, key=key))
        return True


_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO, *, stream: Any = None) -> None:
    """Install the redacting handler as the only root handler."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactionFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
