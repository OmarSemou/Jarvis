"""Provider-neutral local speech-to-text interfaces and adapters."""

from .base import (
    STTProvider,
    TranscriptionErrorCode,
    TranscriptionFailure,
    TranscriptionResult,
)
from .whisper_cpp import (
    WHISPER_CPP_VERSION,
    WhisperBenchmarkResult,
    WhisperCppSTT,
    WhisperCppSettings,
    WhisperTimings,
)

__all__ = [
    "STTProvider",
    "TranscriptionErrorCode",
    "TranscriptionFailure",
    "TranscriptionResult",
    "WHISPER_CPP_VERSION",
    "WhisperBenchmarkResult",
    "WhisperCppSTT",
    "WhisperCppSettings",
    "WhisperTimings",
]
