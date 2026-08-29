"""Typed domain records (T015), from `specs/001-generate-image-video/data-model.md`.

One rule governs this module: **no model-specific value may appear here.** No
duration, frame rate, resolution, sample rate, language, reference limit, or
token capacity is defined, defaulted, or bounded by this file. Those are measured
per adapter and live in `ModelProfile`, which this module describes but never
populates. `tests/unit/test_profile_agnostic.py` re-runs the suite against a
fixture profile with entirely different values to keep that honest.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SHA256_PATTERN = re.compile(r"^[A-Fa-f0-9]{64}$")


class _Record(BaseModel):
    """Frozen, strict base. Domain records are values, not mutable state."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class ModelRole(StrEnum):
    VIDEO = "video"
    VOICE = "voice"
    LIP_SYNC = "lip_sync"


class ProviderMode(StrEnum):
    NATIVE = "native"
    DEDICATED = "dedicated"


class ModelState(StrEnum):
    INSPECTING = "inspecting"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    READY = "ready"
    INCOMPATIBLE = "incompatible"
    FAILED = "failed"
    DELETING = "deleting"


class DownloadState(StrEnum):
    QUEUED = "queued"
    INSPECTING = "inspecting"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RequestState(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    PLANNING = "planning"
    RUNNING = "running"
    EXPORTING = "exporting"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATES


_TERMINAL_STATES = frozenset({RequestState.COMPLETE, RequestState.FAILED, RequestState.CANCELLED})


class StageKind(StrEnum):
    VALIDATE = "validate"
    PREPARE_REFERENCES = "prepare_references"
    ASSEMBLE_PROMPT = "assemble_prompt"
    PLAN_DURATION = "plan_duration"
    LOAD_MODEL = "load_model"
    GENERATE = "generate"
    DECODE = "decode"
    EXPORT = "export"
    VERIFY = "verify"
    METADATA = "metadata"
    PUBLISH = "publish"


class DeviceKind(StrEnum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class BundleAvailability(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    CORRUPT = "corrupt"
    UNSAFE = "unsafe"


class ArtifactKind(StrEnum):
    ORIGINAL_IMAGE = "original_image"
    REFERENCE_AUDIO = "reference_audio"
    DERIVED_VOICE = "derived_voice"
    ASSEMBLED_PROMPT = "assembled_prompt"
    DECODED_VIDEO = "decoded_video"
    DECODED_AUDIO = "decoded_audio"
    FINAL_MP4 = "final_mp4"
    METADATA = "metadata"


class OffloadMode(StrEnum):
    NONE = "none"
    MODEL_CPU = "model_cpu"
    SEQUENTIAL_CPU = "sequential_cpu"
    LAYER_WISE = "layer_wise"


class ReferenceLifecycle(StrEnum):
    STAGED = "staged"
    RETAINED = "retained"
    DISCARDED_WITH_FAILED_STAGE = "discarded_with_failed_stage"


class ErrorCode(StrEnum):
    VALIDATION = "validation"
    CONSENT = "consent"
    FACE = "face"
    REFERENCE = "reference"
    LANGUAGE = "language"
    DURATION = "duration"
    MODEL_URL = "model_url"
    MODEL_ACCESS = "model_access"
    MODEL_DOWNLOAD = "model_download"
    MODEL_INCOMPATIBLE = "model_incompatible"
    INVENTORY = "inventory"
    MODEL_LOAD = "model_load"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    OOM = "oom"
    HOST_MEMORY = "host_memory"
    DISK = "disk"
    GENERATION = "generation"
    EXPORT = "export"
    CODEC = "codec"
    HISTORY = "history"
    CANCELLED = "cancelled"
    FILESYSTEM = "filesystem"
    INTERNAL = "internal"


Sha256 = Annotated[str, Field(pattern=r"^[A-Fa-f0-9]{64}$")]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegInt = Annotated[int, Field(ge=0)]


# --------------------------------------------------------------------------
# Measured capability records
# --------------------------------------------------------------------------


class DurationRange(_Record):
    """A profile's measured supported output duration. No default values here."""

    min_seconds: PositiveFloat
    max_seconds: PositiveFloat
    default_seconds: PositiveFloat

    @model_validator(mode="after")
    def _check_bounds(self) -> DurationRange:
        if self.min_seconds > self.max_seconds:
            raise ValueError("min_seconds must not exceed max_seconds")
        if not (self.min_seconds <= self.default_seconds <= self.max_seconds):
            raise ValueError("default_seconds must fall inside [min_seconds, max_seconds]")
        return self

    def clamp(self, seconds: float) -> float:
        return min(max(seconds, self.min_seconds), self.max_seconds)

    def contains(self, seconds: float) -> bool:
        return self.min_seconds <= seconds <= self.max_seconds


class Resolution(_Record):
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]


class AudioOutput(_Record):
    sample_rate: Annotated[int, Field(gt=0)]
    channels: Annotated[int, Field(gt=0)]


class ReferenceLimits(_Record):
    """Accepted reference kinds with counts, and refused kinds with reasons."""

    accepted: dict[str, Annotated[int, Field(gt=0)]]
    rejected: dict[str, str] = Field(default_factory=dict)
    audio_clip_seconds: DurationRange | None = None


class MemoryProfile(_Record):
    offload_mode: OffloadMode
    quantization: str | None
    expected_peak_reserved_bytes: NonNegInt
    expected_host_resident_bytes: NonNegInt


class SpeakingRate(_Record):
    language: str
    rate: PositiveFloat


class ModelProfile(_Record):
    """Validated capability manifest produced by one installed reviewed adapter.

    Every capability field is a measurement belonging to this profile. Reading one
    is how the rest of the application learns what the model can do; hardcoding
    one anywhere else is an invariant violation.

    License identifiers, terms, URLs, and acknowledgement state are absent by
    design, not by omission — see `test_profile_carries_no_license_field`.
    """

    adapter_key: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    roles: set[ModelRole] = Field(min_length=1)
    native_capabilities: set[ModelRole] = Field(default_factory=set)
    pipeline_class: str = Field(min_length=1)
    supported_devices: set[DeviceKind] = Field(min_length=1)
    dtype_policy: dict[DeviceKind, dict[str, Any]] = Field(default_factory=dict)
    memory_profiles: list[MemoryProfile] = Field(default_factory=list)

    duration_range_seconds: DurationRange
    frame_rate: PositiveFloat
    resolutions: list[Resolution] = Field(min_length=1)
    audio_output: AudioOutput
    dialogue_languages: list[str] = Field(min_length=1)
    speaking_rates: dict[str, PositiveFloat] = Field(default_factory=dict)
    reference_limits: ReferenceLimits
    prompt_capacity_tokens: Annotated[int, Field(gt=0)]
    dialogue_tag_form: str = Field(min_length=1)

    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    weight_policy: dict[str, Any] = Field(default_factory=dict)
    auxiliary_sources: list[dict[str, Any]] = Field(default_factory=list)
    resource_profile: MemoryProfile | None = None

    @model_validator(mode="after")
    def _check_roles(self) -> ModelProfile:
        if not self.native_capabilities <= self.roles:
            raise ValueError("native_capabilities must be a subset of roles")
        native_extras = self.native_capabilities & {ModelRole.VOICE, ModelRole.LIP_SYNC}
        if native_extras and ModelRole.VIDEO not in self.roles:
            raise ValueError(
                "a profile claiming native voice or lip sync must also claim the video role"
            )
        return self

    def speaking_rate_for(self, language: str) -> SpeakingRate | None:
        """The measured rate for a language, or None when the profile has none."""
        rate = self.speaking_rates.get(language)
        return None if rate is None else SpeakingRate(language=language, rate=rate)


# --------------------------------------------------------------------------
# Runtime and request records
# --------------------------------------------------------------------------


class RuntimeProfile(_Record):
    name: str = Field(min_length=1)
    device: DeviceKind
    dtype: str = Field(min_length=1)
    offload_mode: OffloadMode
    quantization: str | None = None
    max_reserved_bytes: NonNegInt | None = None
    max_host_resident_bytes: NonNegInt | None = None
    minimum_free_headroom_bytes: NonNegInt | None = None
    warnings: list[str] = Field(default_factory=list)


class ConsentAttestation(_Record):
    """Only a true, current submission creates this record."""

    request_id: uuid.UUID
    reference_audio_sha256: Sha256
    confirmed: Literal[True]
    confirmed_at: datetime


class VoiceOrigin(_Record):
    bundle_id: uuid.UUID
    artifact_relative_path: str
    artifact_sha256: Sha256


class DialogueSegment(_Record):
    language: str = Field(min_length=1)
    text: str = Field(min_length=1)


class MotionTruncation(_Record):
    original_length: NonNegInt
    retained_length: NonNegInt
    discarded_length: NonNegInt

    @model_validator(mode="after")
    def _check_sum(self) -> MotionTruncation:
        if self.retained_length + self.discarded_length != self.original_length:
            raise ValueError("retained_length + discarded_length must equal original_length")
        return self


class AssembledPrompt(_Record):
    """Only `motion_text` may be truncated. Dialogue is never altered."""

    motion_text: str
    dialogue_segments: list[DialogueSegment] = Field(min_length=1)
    rendered: str = Field(min_length=1)
    token_count: NonNegInt
    token_capacity: Annotated[int, Field(gt=0)]
    motion_truncation: MotionTruncation | None = None
    structuring_version: str = Field(min_length=1)

    @property
    def over_capacity(self) -> bool:
        """Whether the prompt exceeds what the model can hold.

        Recorded, never rejected. Speech is the output's content: refusing a long
        script would terminate a request for its length, which the spec forbids.
        Motion gives way first; if the script alone still exceeds capacity, the
        adapter is told the truth and decides.
        """
        return self.token_count > self.token_capacity


class DurationDecision(_Record):
    """Duration is an INPUT to joint generation, never a measurement of speech.

    There is no pre-generation fit check and no rejection for script length. The
    suggestion is advisory; the operator may choose anything the profile supports.
    """

    suggested_duration_seconds: PositiveFloat
    speaking_rate_used: SpeakingRate | None
    requested_duration_seconds: float | None
    operator_overrode: bool
    effective_duration_seconds: PositiveFloat
    effective_num_frames: Annotated[int, Field(gt=0)]
    frame_rate: PositiveFloat
    resolution: Resolution
    audio_sample_rate: Annotated[int, Field(gt=0)]
    overrides: list[str] = Field(default_factory=list)
    profile_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_frames(self) -> DurationDecision:
        expected = round(self.effective_duration_seconds * self.frame_rate)
        if abs(self.effective_num_frames - expected) > 1:
            raise ValueError(
                "effective_num_frames must follow effective_duration_seconds x frame_rate"
            )
        return self


class ReferenceSet(_Record):
    """One or more image references plus exactly one audio timbre anchor.

    Video references are rejected outright: a short video reference costs tens of
    thousands of tokens on its own and would blow the accelerator ceiling before
    generation starts. The rejection is a profile-declared refusal, not a bug.

    The application imposes NO maximum on `image_paths`. The only bound is the
    profile's measured `reference_limits`, and exceeding it is a `reference`
    error rather than a silent trim.
    """

    request_id: uuid.UUID
    image_paths: list[str] = Field(min_length=1)
    audio_path: str = Field(min_length=1)
    audio_role: Literal["timbre_anchor"] = "timbre_anchor"
    source_paths: dict[str, str] = Field(default_factory=dict)
    measured: dict[str, Any] = Field(default_factory=dict)
    speaker_result: dict[str, Any] = Field(default_factory=dict)
    transcript: str | None = None
    consent: ConsentAttestation
    origin: VoiceOrigin | None = None
    derived_artifact_path: str | None = None
    profile_limits: dict[str, Any] = Field(default_factory=dict)
    rejected_kinds: list[str] = Field(default_factory=list)
    lifecycle: ReferenceLifecycle = ReferenceLifecycle.STAGED

    @model_validator(mode="after")
    def _check_consent_binds_to_this_request(self) -> ReferenceSet:
        if self.consent.request_id != self.request_id:
            raise ValueError("consent must be bound to this request_id")
        return self


class ArtifactRecord(_Record):
    kind: ArtifactKind
    relative_path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size_bytes: NonNegInt
    sha256: Sha256
    created_by_stage: StageKind

    @field_validator("relative_path")
    @classmethod
    def _check_relative(cls, value: str) -> str:
        from storage import is_safe_relative  # local import avoids a cycle

        if not is_safe_relative(value):
            raise ValueError(f"unsafe artifact path: {value!r}")
        return value


class MemorySnapshot(_Record):
    available: bool
    device_name: str | None = None
    allocated_bytes: NonNegInt | None = None
    reserved_bytes: NonNegInt | None = None
    peak_allocated_bytes: NonNegInt | None = None
    peak_reserved_bytes: NonNegInt | None = None
    free_bytes: NonNegInt | None = None
    total_bytes: NonNegInt | None = None
    host_resident_bytes: NonNegInt | None = None
    reserved_gate_passed: bool | None = None
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _check_reason(self) -> MemorySnapshot:
        if not self.available and not self.unavailable_reason:
            raise ValueError("unavailable_reason is required when available is False")
        return self


class ProgressEvent(_Record):
    request_id: uuid.UUID
    phase: str = Field(min_length=1)
    fraction: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    message: str = ""
    memory: MemorySnapshot | None = None
    timestamp: datetime


class ErrorDetail(_Record):
    code: ErrorCode
    message: str
    retryable: bool
    suggestions: list[str] = Field(default_factory=list)


class GeneratedOutput(_Record):
    """The single joint audio/video result. No separate speech artifact exists."""

    request_id: uuid.UUID
    decoded_video_path: str
    decoded_audio_path: str
    final_mp4_path: str | None = None
    frame_count: Annotated[int, Field(gt=0)]
    frame_rate: PositiveFloat
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    audio_sample_rate: Annotated[int, Field(gt=0)]
    audio_channels: Annotated[int, Field(gt=0)]
    measured_video_seconds: PositiveFloat
    measured_audio_seconds: PositiveFloat
    script_sha256: Sha256
    language: str
    profile_id: str
    non_silent: bool
    lifecycle: ReferenceLifecycle = ReferenceLifecycle.STAGED


class GenerationResult(_Record):
    request_id: uuid.UUID
    state: RequestState
    video_path: str | None
    bundle_path: str | None
    manifest_path: str | None
    artifact_inventory: list[ArtifactRecord] = Field(default_factory=list)
    retained_bytes: NonNegInt = 0
    execution_plan: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float
    memory_by_stage: dict[str, MemorySnapshot] = Field(default_factory=dict)
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def _check_terminal(self) -> GenerationResult:
        if not self.state.is_terminal:
            raise ValueError(f"{self.state.value} is not a terminal state")
        if self.state is RequestState.COMPLETE and self.error is not None:
            raise ValueError("a complete result must not carry an error")
        if self.state is not RequestState.COMPLETE and self.error is None:
            raise ValueError(f"a {self.state.value} result must carry an error")
        if self.state is not RequestState.COMPLETE and (
            self.video_path or self.bundle_path or self.manifest_path
        ):
            raise ValueError("a non-complete result must publish nothing")
        return self
