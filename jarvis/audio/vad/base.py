"""VAD contracts independent of Silero/OpenWakeWord implementation types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class VADErrorCode(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_MISSING = "model_missing"
    MODEL_LOAD_FAILED = "model_load_failed"
    INFERENCE_FAILED = "inference_failed"


@dataclass(frozen=True, slots=True)
class VADFailure:
    code: VADErrorCode
    message: str


class VADProvider(Protocol):
    name: str

    def readiness_error(self) -> VADFailure | None: ...

    def warmup(self) -> VADFailure | None: ...

    def score(self, pcm16: bytes) -> float: ...

    def reset(self) -> None: ...
