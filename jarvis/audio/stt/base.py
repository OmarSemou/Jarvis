"""Provider-neutral speech-to-text contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class TranscriptionErrorCode(StrEnum):
    EXECUTABLE_MISSING = "executable_missing"
    MODEL_MISSING = "model_missing"
    AUDIO_MISSING = "audio_missing"
    INVALID_AUDIO = "invalid_audio"
    TIMEOUT = "timeout"
    PROCESS_FAILED = "process_failed"
    OUTPUT_MISSING = "output_missing"
    EMPTY_TRANSCRIPT = "empty_transcript"


@dataclass(frozen=True, slots=True)
class TranscriptionFailure:
    code: TranscriptionErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    success: bool
    text: str
    provider: str
    elapsed_seconds: float
    audio_duration_seconds: float | None = None
    detected_language: str | None = None
    error: TranscriptionFailure | None = None

    @property
    def real_time_factor(self) -> float | None:
        if not self.audio_duration_seconds or self.audio_duration_seconds <= 0:
            return None
        return self.elapsed_seconds / self.audio_duration_seconds


class STTProvider(Protocol):
    name: str

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe one local audio file without leaking provider objects."""

    def readiness_error(self) -> TranscriptionFailure | None:
        """Return a structured local setup failure without running inference."""
