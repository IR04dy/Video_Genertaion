"""T024: consent. Fresh, server-bound, and reset on every change."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from domain import ConsentAttestation
from errors import ConsentError
from execution import build_consent, verify_consent

SHA_A = "a" * 64
SHA_B = "b" * 64


def test_a_true_submission_creates_the_record() -> None:
    request_id = uuid.uuid4()
    consent = build_consent(request_id=request_id, audio_sha256=SHA_A, confirmed=True)
    assert consent.confirmed is True
    assert consent.request_id == request_id
    assert consent.reference_audio_sha256 == SHA_A


def test_a_false_submission_creates_nothing() -> None:
    with pytest.raises(ConsentError):
        build_consent(request_id=uuid.uuid4(), audio_sha256=SHA_A, confirmed=False)


def test_the_record_cannot_be_constructed_as_false() -> None:
    """Belt and braces: the type itself refuses, not just the factory."""
    with pytest.raises(ValidationError):
        ConsentAttestation(
            request_id=uuid.uuid4(),
            reference_audio_sha256=SHA_A,
            confirmed=False,
            confirmed_at=datetime.now(UTC),
        )


def test_timestamp_is_server_generated_and_utc() -> None:
    consent = build_consent(request_id=uuid.uuid4(), audio_sha256=SHA_A, confirmed=True)
    assert consent.confirmed_at.tzinfo is not None
    assert consent.confirmed_at.utcoffset().total_seconds() == 0


def test_verify_accepts_a_matching_request_and_digest() -> None:
    request_id = uuid.uuid4()
    consent = build_consent(request_id=request_id, audio_sha256=SHA_A, confirmed=True)
    verify_consent(consent, request_id=request_id, audio_sha256=SHA_A)


def test_verify_rejects_a_digest_from_different_audio() -> None:
    """The whole point: consent for recording A never authorizes recording B."""
    request_id = uuid.uuid4()
    consent = build_consent(request_id=request_id, audio_sha256=SHA_A, confirmed=True)
    with pytest.raises(ConsentError, match="audio"):
        verify_consent(consent, request_id=request_id, audio_sha256=SHA_B)


def test_verify_rejects_consent_from_another_request() -> None:
    """No bundle or history record can hydrate consent for a later request."""
    consent = build_consent(request_id=uuid.uuid4(), audio_sha256=SHA_A, confirmed=True)
    with pytest.raises(ConsentError, match="request"):
        verify_consent(consent, request_id=uuid.uuid4(), audio_sha256=SHA_A)


def test_verify_rejects_a_missing_attestation() -> None:
    with pytest.raises(ConsentError):
        verify_consent(None, request_id=uuid.uuid4(), audio_sha256=SHA_A)


def test_digest_must_be_a_real_sha256() -> None:
    with pytest.raises((ConsentError, ValidationError)):
        build_consent(request_id=uuid.uuid4(), audio_sha256="short", confirmed=True)


def test_consent_is_frozen_after_construction() -> None:
    consent = build_consent(request_id=uuid.uuid4(), audio_sha256=SHA_A, confirmed=True)
    with pytest.raises(ValidationError):
        consent.reference_audio_sha256 = SHA_B  # type: ignore[misc]
