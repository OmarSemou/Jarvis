"""Provider-neutral contracts for fully local text-to-speech."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Iterator
from threading import Event
from typing import Protocol, runtime_checkable


class SynthesisErrorCode(StrEnum):
    DISABLED = "disabled"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_MISSING = "model_missing"
    VOICE_MISSING = "voice_missing"
    INVALID_VOICE = "invalid_voice"
    INVALID_TEXT = "invalid_text"
    INVALID_SPEED = "invalid_speed"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    MODEL_LOAD_FAILED = "model_load_failed"
    SYNTHESIS_FAILED = "synthesis_failed"
    EMPTY_AUDIO = "empty_audio"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class SynthesisFailure:
    code: SynthesisErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    """Interleaved signed little-endian PCM16, independent of any provider."""

    pcm16: bytes
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if not self.pcm16:
            raise ValueError("pcm16 audio must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in {1, 2}:
            raise ValueError("channels must be 1 or 2")
        if len(self.pcm16) % (2 * self.channels):
            raise ValueError("pcm16 audio must contain complete interleaved frames")

    @property
    def frame_count(self) -> int:
        return len(self.pcm16) // (2 * self.channels)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate


@dataclass(frozen=True, slots=True)
class SpeechSynthesisResult:
    success: bool
    provider: str
    voice: str
    elapsed_seconds: float
    audio: SynthesizedAudio | None = None
    first_audio_seconds: float | None = None
    error: SynthesisFailure | None = None

    def __post_init__(self) -> None:
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must not be negative")
        if self.first_audio_seconds is not None and self.first_audio_seconds < 0:
            raise ValueError("first_audio_seconds must not be negative")
        if self.success and (self.audio is None or self.error is not None):
            raise ValueError("successful synthesis requires audio and no error")
        if not self.success and (self.audio is not None or self.error is None):
            raise ValueError("failed synthesis requires one structured error and no audio")

    @property
    def speech_duration_seconds(self) -> float | None:
        return self.audio.duration_seconds if self.audio is not None else None

    @property
    def real_time_factor(self) -> float | None:
        duration = self.speech_duration_seconds
        if duration is None or duration <= 0:
            return None
        return self.elapsed_seconds / duration


@dataclass(frozen=True, slots=True)
class SpeechAudioChunk:
    """One ordered provider output for a Jarvis semantic speech chunk."""

    audio: SynthesizedAudio
    sequence: int
    final: bool = False

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("audio chunk sequence cannot be negative")


class SynthesisStreamError(RuntimeError):
    """Structured provider failure raised while iterating local audio."""

    def __init__(self, failure: SynthesisFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class TTSProvider(Protocol):
    name: str
    available_voices: tuple[str, ...]

    def readiness_error(self, voice: str) -> SynthesisFailure | None: ...

    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed: float = 1.0,
        language: str = "en",
    ) -> SpeechSynthesisResult: ...


@runtime_checkable
class StreamingTTSProvider(TTSProvider, Protocol):
    def synthesize_stream(
        self,
        text: str,
        *,
        voice: str,
        speed: float = 1.0,
        language: str = "en",
        cancellation: Event | None = None,
    ) -> Iterator[SpeechAudioChunk]: ...
