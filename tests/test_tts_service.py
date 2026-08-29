from jarvis.audio.tts.base import (
    SpeechSynthesisResult,
    SynthesizedAudio,
    SynthesisErrorCode,
    SynthesisFailure,
)
from jarvis.audio.tts.playback import PlaybackErrorCode, PlaybackFailure, PlaybackResult
from jarvis.audio.tts.service import TTSService


class Provider:
    def __init__(self, name, voices, *, failure=None):
        self.name = name
        self.available_voices = voices
        self.failure = failure
        self.calls = []

    def readiness_error(self, _voice):
        return self.failure

    def synthesize(self, text, *, voice, speed, language):
        self.calls.append((text, voice, speed, language))
        if self.failure:
            return SpeechSynthesisResult(False, self.name, voice, 0.01, error=self.failure)
        return SpeechSynthesisResult(
            True,
            self.name,
            voice,
            0.01,
            SynthesizedAudio(b"\x00\x00" * 100, 10_000),
            0.01,
        )


class Playback:
    def __init__(self, result=None):
        self.result = result or PlaybackResult(True, "Speakers")
        self.audio = []
        self.stopped = 0

    def play(self, audio):
        self.audio.append(audio)
        return self.result

    def stop(self):
        self.stopped += 1


def _service(*, playback=None, failure=None):
    return TTSService(
        {
            "kokoro": Provider("kokoro", ("am_fenrir",), failure=failure),
            "piper": Provider("piper", ("en_US-joe-medium",)),
        },
        playback or Playback(),
        enabled=True,
    )


def test_service_synthesizes_then_plays_and_supports_session_switches():
    service = _service()
    result = service.speak("Hey.")
    assert result.success
    assert len(service.playback.audio) == 1

    service.set_provider("piper")
    assert service.voice == "en_US-joe-medium"
    service.set_voice("en_US-joe-medium")
    assert service.status().ready


def test_service_preserves_display_text_but_sends_speech_safe_text_to_provider():
    service = _service()
    display_text = "Use the **Power Stroke**."

    result = service.speak(display_text)

    assert result.success
    assert display_text == "Use the **Power Stroke**."
    assert service.providers["kokoro"].calls[0][0] == "Use the Power Stroke."
    assert "*" not in service.providers["kokoro"].calls[0][0]


def test_disabled_service_never_synthesizes_or_plays():
    service = _service()
    service.set_enabled(False)
    result = service.speak("Hey.")

    assert not result.success
    assert result.synthesis.error.code is SynthesisErrorCode.DISABLED
    assert not service.providers["kokoro"].calls
    assert not service.playback.audio


def test_service_does_not_play_failed_synthesis():
    failure = SynthesisFailure(SynthesisErrorCode.SYNTHESIS_FAILED, "no audio")
    service = _service(failure=failure)
    result = service.speak("Hey.")
    assert not result.success
    assert result.error_message == "no audio"
    assert not service.playback.audio


def test_service_preserves_structured_playback_error_and_cancel_boundary():
    playback = Playback(
        PlaybackResult(
            False,
            error=PlaybackFailure(PlaybackErrorCode.PLAYBACK_FAILED, "speaker gone"),
        )
    )
    service = _service(playback=playback)
    result = service.speak("Hey.")
    assert not result.success
    assert result.error_message == "speaker gone"
    service.cancel()
    assert playback.stopped == 1


def test_service_rejects_unknown_provider_and_cross_provider_voice():
    service = _service()
    for action in (
        lambda: service.set_provider("cloud"),
        lambda: service.set_voice("en_US-joe-medium"),
    ):
        try:
            action()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid TTS selection was accepted")


def test_warmup_synthesizes_once_without_playback_and_normal_speech_still_plays():
    service = _service()

    assert service.warmup() is None
    assert service.warmup() is None
    assert len(service.providers["kokoro"].calls) == 1
    assert service.playback.audio == []

    result = service.speak("Ready.")
    assert result.success
    assert len(service.providers["kokoro"].calls) == 2
    assert len(service.playback.audio) == 1


def test_failed_warmup_is_cached_and_never_plays_audio():
    failure = SynthesisFailure(SynthesisErrorCode.MODEL_LOAD_FAILED, "cold failure")
    service = _service(failure=failure)

    assert service.warmup().message == "cold failure"
    assert service.warmup().message == "cold failure"
    assert len(service.providers["kokoro"].calls) == 1
    assert service.playback.audio == []
