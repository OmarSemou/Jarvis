"""Provider-neutral local speech synthesis and playback."""

from .base import (
    SpeechAudioChunk,
    SpeechSynthesisResult,
    SynthesizedAudio,
    SynthesisErrorCode,
    SynthesisFailure,
    SynthesisStreamError,
    TTSProvider,
)
from .chunks import SpeechChunk, SpeechChunker, SpeechChunkerSettings
from .playback import AudioPlaybackService, PlaybackResult
from .pipeline import (
    SpeechPipelineMetrics,
    SpeechPipelineResult,
    SpeechPipelineSettings,
    SpeechSessionHandle,
)
from .service import SpeechOutputResult, TTSService, TTSStatus

__all__ = [
    "SpeechSynthesisResult",
    "SpeechAudioChunk",
    "SpeechChunk",
    "SpeechChunker",
    "SpeechChunkerSettings",
    "SynthesizedAudio",
    "SynthesisErrorCode",
    "SynthesisFailure",
    "SynthesisStreamError",
    "TTSProvider",
    "AudioPlaybackService",
    "PlaybackResult",
    "SpeechPipelineMetrics",
    "SpeechPipelineResult",
    "SpeechPipelineSettings",
    "SpeechSessionHandle",
    "SpeechOutputResult",
    "TTSService",
    "TTSStatus",
]
