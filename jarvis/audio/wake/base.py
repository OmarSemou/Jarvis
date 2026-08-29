"""Wake-word contracts independent of OpenWakeWord and microphone hardware."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class WakeWordErrorCode(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_MISSING = "model_missing"
    FEATURE_MODEL_MISSING = "feature_model_missing"
    MODEL_LOAD_FAILED = "model_load_failed"
    INFERENCE_FAILED = "inference_failed"
    INVALID_AUDIO = "invalid_audio"


@dataclass(frozen=True, slots=True)
class WakeWordFailure:
    code: WakeWordErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class WakeWordDetection:
    detected: bool
    score: float
    provider: str
    phrase: str
    error: WakeWordFailure | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("wake-word score must be from 0 to 1")
        if self.error is not None and self.detected:
            raise ValueError("failed wake-word inference cannot be detected")


class WakeWordProvider(Protocol):
    name: str
    phrase: str
    threshold: float

    def readiness_error(self) -> WakeWordFailure | None: ...

    def warmup(self) -> WakeWordFailure | None: ...

    def process(self, pcm16: bytes) -> WakeWordDetection: ...

    def reset(self) -> None: ...
