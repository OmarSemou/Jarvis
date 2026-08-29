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
from .profiles import (
    BMO_PROFILE,
    FENRIR_PROFILE,
    VOICE_PROFILES,
    VoiceProfile,
    profile_for_selection,
    resolve_voice_profile,
)

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
    "VoiceProfile",
    "FENRIR_PROFILE",
    "BMO_PROFILE",
    "VOICE_PROFILES",
    "resolve_voice_profile",
    "profile_for_selection",
]
