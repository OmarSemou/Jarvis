from threading import Event

import pytest

from jarvis.audio.tts.base import (
    SpeechAudioChunk,
    SynthesisErrorCode,
    SynthesisStreamError,
    SynthesizedAudio,
)
from jarvis.audio.tts.kokoro import KOKORO_VOICES, KokoroSettings, KokoroTTS


class Engine:
    def __init__(self, result=None, error=None):
        self.result = result or ([0.0, 0.5, -0.5], 24_000)
        self.error = error
        self.calls = []

    def create(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if self.error:
            raise self.error
        return self.result


def _provider(tmp_path, *, engine=None, package=True, create_files=True):
    model = tmp_path / "kokoro.onnx"
    voices = tmp_path / "voices.bin"
    if create_files:
        model.write_bytes(b"model")
        voices.write_bytes(b"voices")
    engine = engine or Engine()
    provider = KokoroTTS(
        KokoroSettings(model.resolve(), voices.resolve()),
        engine_loader=lambda *_paths: engine,
        package_probe=lambda _name: package,
    )
    return provider, engine, model, voices


def test_kokoro_returns_provider_neutral_pcm16_and_caches_model(tmp_path):
    provider, engine, _, _ = _provider(tmp_path)

    first = provider.synthesize("Hey.", voice="am_michael")
    second = provider.synthesize("There.", voice="am_michael")

    assert first.success
    assert isinstance(first.audio, SynthesizedAudio)
    assert first.audio.sample_rate == 24_000
    assert first.audio.channels == 1
    assert len(engine.calls) == 2
    assert second.success


def test_kokoro_missing_package_model_and_voice_bundle_fail_cleanly(tmp_path):
    provider, _, model, voices = _provider(tmp_path, package=False)
    assert provider.synthesize("Hey.", voice="am_michael").error.code is SynthesisErrorCode.PROVIDER_UNAVAILABLE

    provider, _, model, voices = _provider(tmp_path)
    model.unlink()
    assert provider.synthesize("Hey.", voice="am_michael").error.code is SynthesisErrorCode.MODEL_MISSING
    model.write_bytes(b"model")
    voices.unlink()
    assert provider.synthesize("Hey.", voice="am_michael").error.code is SynthesisErrorCode.VOICE_MISSING


def test_kokoro_rejects_invalid_voice_and_handles_synthesis_failure(tmp_path):
    provider, _, _, _ = _provider(tmp_path)
    invalid = provider.synthesize("Hey.", voice="../../voice")
    assert invalid.error.code is SynthesisErrorCode.INVALID_VOICE

    provider, _, _, _ = _provider(tmp_path, engine=Engine(error=RuntimeError("bad inference")))
    failed = provider.synthesize("Hey.", voice=KOKORO_VOICES[0])
    assert failed.error.code is SynthesisErrorCode.SYNTHESIS_FAILED
    assert "bad inference" in failed.error.message


def test_kokoro_rejects_non_english_language(tmp_path):
    provider, _, _, _ = _provider(tmp_path)
    result = provider.synthesize("Hej.", voice="am_michael", language="da")
    assert result.error.code is SynthesisErrorCode.UNSUPPORTED_LANGUAGE


def test_kokoro_async_stream_is_adapted_to_ordered_provider_neutral_pcm(tmp_path):
    class StreamingEngine(Engine):
        async def create_stream(self, text, **kwargs):
            self.calls.append((text, kwargs))
            yield [0.0, 0.25], 24_000
            yield [-0.25, 0.5], 24_000

    provider, engine, _, _ = _provider(tmp_path, engine=StreamingEngine())

    chunks = list(provider.synthesize_stream("Streaming.", voice="am_fenrir"))

    assert all(isinstance(chunk, SpeechAudioChunk) for chunk in chunks)
    assert [chunk.sequence for chunk in chunks] == [0, 1]
    assert all(chunk.audio.sample_rate == 24_000 for chunk in chunks)
    assert engine.calls[0][0] == "Streaming."


def test_kokoro_stream_cancellation_discards_remaining_audio(tmp_path):
    cancellation = Event()

    class StreamingEngine(Engine):
        async def create_stream(self, _text, **_kwargs):
            yield [0.1], 24_000
            yield [0.2], 24_000

    provider, _, _, _ = _provider(tmp_path, engine=StreamingEngine())
    stream = provider.synthesize_stream(
        "Cancel me.", voice="am_fenrir", cancellation=cancellation
    )

    first = next(stream)
    cancellation.set()

    assert first.sequence == 0
    assert list(stream) == []


def test_kokoro_stream_error_is_structured(tmp_path):
    class BrokenStreamingEngine(Engine):
        async def create_stream(self, _text, **_kwargs):
            raise RuntimeError("stream broke")
            yield

    provider, _, _, _ = _provider(tmp_path, engine=BrokenStreamingEngine())

    with pytest.raises(SynthesisStreamError) as raised:
        list(provider.synthesize_stream("Broken.", voice="am_fenrir"))

    assert raised.value.failure.code is SynthesisErrorCode.SYNTHESIS_FAILED
    assert "stream broke" in raised.value.failure.message
