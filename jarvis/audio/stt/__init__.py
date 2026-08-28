"""Provider-neutral local speech-to-text interfaces and adapters."""

from .base import (
    STTProvider,
    TranscriptionErrorCode,
    TranscriptionFailure,
    TranscriptionResult,
)
from .whisper_cpp import WhisperCppSTT, WhisperCppSettings

__all__ = [
    "STTProvider",
    "TranscriptionErrorCode",
    "TranscriptionFailure",
    "TranscriptionResult",
    "WhisperCppSTT",
    "WhisperCppSettings",
]
