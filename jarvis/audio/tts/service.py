"""Application-level orchestration for local synthesis and playback."""

from __future__ import annotations

from dataclasses import dataclass

from .base import (
    SpeechSynthesisResult,
    SynthesizedAudio,
    SynthesisErrorCode,
    SynthesisFailure,
    TTSProvider,
)
from .playback import AudioPlaybackService, PlaybackHandle, PlaybackResult
from .text import prepare_text_for_speech


DEFAULT_PROVIDER_VOICES = {
    "kokoro": "am_fenrir",
    "piper": "en_US-joe-medium",
}


@dataclass(frozen=True, slots=True)
class TTSStatus:
    enabled: bool
    provider: str
    voice: str
    speed: float
    language: str
    ready: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SpeechOutputResult:
    success: bool
    synthesis: SpeechSynthesisResult
    playback: PlaybackResult | None = None

    @property
    def error_message(self) -> str | None:
        if self.synthesis.error is not None:
            return self.synthesis.error.message
        if self.playback is not None and self.playback.error is not None:
            return self.playback.error.message
        return None


class TTSService:
    """Mutable session settings around explicit allowlisted providers."""

    def __init__(
        self,
        providers: dict[str, TTSProvider],
        playback: AudioPlaybackService,
        *,
        enabled: bool = False,
        provider: str = "kokoro",
        voice: str = "am_fenrir",
        speed: float = 1.0,
        language: str = "en",
    ) -> None:
        if set(providers) != set(DEFAULT_PROVIDER_VOICES):
            raise ValueError("TTS providers must be exactly: kokoro, piper")
        self.providers = dict(providers)
        self.playback = playback
        self.enabled = enabled
        self.provider = provider
        self.voice = voice
        self.speed = speed
        self.language = language
        self._warmup_attempted = False
        self._warmup_failure: SynthesisFailure | None = None
        self._validate_selection(provider, voice)

    def _validate_selection(self, provider: str, voice: str) -> None:
        selected = self.providers.get(provider)
        if selected is None:
            raise ValueError("TTS provider must be one of: kokoro, piper")
        if voice not in selected.available_voices:
            raise ValueError(
                f"Voice '{voice}' is not available for {provider}. "
                f"Allowed: {', '.join(selected.available_voices)}"
            )

    def status(self) -> TTSStatus:
        failure = self.providers[self.provider].readiness_error(self.voice)
        return TTSStatus(
            self.enabled,
            self.provider,
            self.voice,
            self.speed,
            self.language,
            failure is None,
            "ready" if failure is None else failure.message,
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def set_provider(self, provider: str) -> None:
        normalized = provider.strip().casefold()
        if normalized not in self.providers:
            raise ValueError("TTS provider must be one of: kokoro, piper")
        self.provider = normalized
        self.voice = DEFAULT_PROVIDER_VOICES[normalized]

    def set_voice(self, voice: str) -> None:
        normalized = voice.strip()
        self._validate_selection(self.provider, normalized)
        self.voice = normalized

    def synthesize(self, text: str) -> SpeechSynthesisResult:
        if not self.enabled:
            return self.disabled_result().synthesis
        provider = self.providers[self.provider]
        return provider.synthesize(
            prepare_text_for_speech(text),
            voice=self.voice,
            speed=self.speed,
            language=self.language,
        )

    def warmup(self, text: str = "Hi.") -> SynthesisFailure | None:
        """Load the selected provider once; synthesized audio is never played."""

        if self._warmup_attempted:
            return self._warmup_failure
        self._warmup_attempted = True
        result = self.synthesize(text)
        self._warmup_failure = result.error
        return self._warmup_failure

    def start_playback(self, audio: SynthesizedAudio) -> PlaybackHandle:
        return self.playback.start(audio)

    def speak(self, text: str) -> SpeechOutputResult:
        synthesis = self.synthesize(text)
        if not synthesis.success or synthesis.audio is None:
            return SpeechOutputResult(False, synthesis)
        playback = self.playback.play(synthesis.audio)
        return SpeechOutputResult(playback.success, synthesis, playback)

    def disabled_result(self) -> SpeechOutputResult:
        result = SpeechSynthesisResult(
            False,
            self.provider,
            self.voice,
            0.0,
            error=SynthesisFailure(
                SynthesisErrorCode.DISABLED,
                "Voice output is disabled.",
            ),
        )
        return SpeechOutputResult(False, result)

    def stop(self) -> None:
        self.playback.stop()

    cancel = stop
