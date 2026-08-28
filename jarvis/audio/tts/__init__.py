"""Provider-neutral local speech synthesis and playback."""

from .base import (
    SpeechSynthesisResult,
    SynthesizedAudio,
    SynthesisErrorCode,
    SynthesisFailure,
    TTSProvider,
)
from .playback import AudioPlaybackService, PlaybackResult
from .service import SpeechOutputResult, TTSService, TTSStatus

__all__ = [
    "SpeechSynthesisResult",
    "SynthesizedAudio",
    "SynthesisErrorCode",
    "SynthesisFailure",
    "TTSProvider",
    "AudioPlaybackService",
    "PlaybackResult",
    "SpeechOutputResult",
    "TTSService",
    "TTSStatus",
]
