"""Application-level orchestration for local synthesis and playback."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from threading import Lock
from time import perf_counter

from .base import (
    SpeechSynthesisResult,
    SynthesizedAudio,
    SynthesisErrorCode,
    SynthesisFailure,
    TTSProvider,
)
from .playback import AudioPlaybackService, PlaybackHandle, PlaybackResult
from .chunks import SpeechChunker
from .pipeline import (
    SpeechPipeline,
    SpeechPipelineSettings,
    SpeechSessionHandle,
)
from .text import prepare_text_for_speech
from .profiles import VoiceProfile, profile_for_selection, resolve_voice_profile


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
        profile: str | None = None,
        chunker: SpeechChunker | None = None,
        pipeline_settings: SpeechPipelineSettings = SpeechPipelineSettings(),
    ) -> None:
        if not set(DEFAULT_PROVIDER_VOICES).issubset(providers):
            raise ValueError("TTS providers must include: kokoro, piper")
        self.providers = dict(providers)
        self.playback = playback
        self.enabled = enabled
        self.provider = provider
        self.voice = voice
        self.speed = speed
        self.language = language
        self.chunker = chunker or SpeechChunker()
        self.pipeline = SpeechPipeline(playback, settings=pipeline_settings)
        self._generation_ids = count(1)
        self._session_lock = Lock()
        self._active_session: SpeechSessionHandle | None = None
        self._warmup_attempted = False
        self._warmup_failure: SynthesisFailure | None = None
        self._profile_id = profile_for_selection(provider, voice)
        try:
            self._validate_selection(provider, voice)
        except ValueError:
            if profile is not None and resolve_voice_profile(profile).id == "bmo":
                raise ValueError(
                    "The original BMO voice model is unavailable. Restore the "
                    "preserved BMO voice asset before selecting this profile."
                ) from None
            raise
        if profile is not None:
            self.set_profile(profile)

    @property
    def profile(self) -> VoiceProfile | None:
        """The semantic profile, or ``None`` for a legacy custom selection."""

        if self._profile_id is None:
            return None
        return resolve_voice_profile(self._profile_id)

    @property
    def profile_id(self) -> str | None:
        return self._profile_id

    def _validate_selection(self, provider: str, voice: str) -> None:
        selected = self.providers.get(provider)
        if selected is None:
            allowed = ", ".join(sorted(self.providers))
            raise ValueError(f"TTS provider must be one of: {allowed}")
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
        if normalized not in self.providers or normalized not in DEFAULT_PROVIDER_VOICES:
            allowed = ", ".join(sorted(self.providers))
            raise ValueError(f"TTS provider must be one of: {allowed}")
        default_voice = DEFAULT_PROVIDER_VOICES[normalized]
        self._validate_selection(normalized, default_voice)
        self.stop()
        self.provider = normalized
        self.voice = default_voice
        self._profile_id = profile_for_selection(self.provider, self.voice)
        self._warmup_attempted = False
        self._warmup_failure = None

    def set_voice(self, voice: str) -> None:
        normalized = voice.strip()
        self._validate_selection(self.provider, normalized)
        self.stop()
        self.voice = normalized
        self._profile_id = profile_for_selection(self.provider, self.voice)
        self._warmup_attempted = False
        self._warmup_failure = None

    def set_profile(self, profile_id: str) -> None:
        """Switch semantic voice profiles and stop any prior session safely."""

        profile = resolve_voice_profile(profile_id)
        provider = self.providers.get(profile.provider)
        if provider is None or profile.provider_voice not in provider.available_voices:
            if profile.id == "bmo":
                raise ValueError(
                    "The original BMO voice model is unavailable. Restore the "
                    "preserved BMO voice asset before selecting this profile."
                )
            raise ValueError(
                f"Voice profile '{profile.id}' is unavailable from the configured providers."
            )
        self.stop()
        self.provider = profile.provider
        self.voice = profile.provider_voice
        self._profile_id = profile.id
        self._warmup_attempted = False
        self._warmup_failure = None

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

    def start_speech(
        self,
        text: str,
        *,
        assistant_text_ready: float | None = None,
    ) -> SpeechSessionHandle:
        """Begin bounded semantic synthesis and playback for one response."""

        ready_at = perf_counter() if assistant_text_ready is None else assistant_text_ready
        speech_text = prepare_text_for_speech(text)
        chunks = self.chunker.chunk(speech_text)
        failure = None
        if not self.enabled:
            failure = SynthesisFailure(
                SynthesisErrorCode.DISABLED,
                "Voice output is disabled.",
            )

        with self._session_lock:
            previous = self._active_session
            if previous is not None and not previous.done:
                previous.stop()
                previous.wait(1.0)
            handle = self.pipeline.start(
                self.providers[self.provider],
                chunks,
                generation_id=next(self._generation_ids),
                voice=self.voice,
                speed=self.speed,
                language=self.language,
                assistant_text_ready=ready_at,
                initial_failure=failure,
            )
            self._active_session = handle
            return handle

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
        with self._session_lock:
            active = self._active_session
        if active is not None:
            active.stop()
        self.playback.stop()

    cancel = stop
