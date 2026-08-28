"""Local audio capture contracts with no import-time device access."""

from .devices import (
    AudioDevice,
    MicrophoneDeviceService,
    MicrophoneError,
    MicrophoneStatus,
)
from .recorder import AudioRecording, PushToTalkRecorder, RecordingError
from .service import VoiceInputError, VoiceInputOutcome, VoiceInputService

__all__ = [
    "AudioDevice",
    "AudioRecording",
    "MicrophoneDeviceService",
    "MicrophoneError",
    "MicrophoneStatus",
    "PushToTalkRecorder",
    "RecordingError",
    "VoiceInputError",
    "VoiceInputOutcome",
    "VoiceInputService",
]
