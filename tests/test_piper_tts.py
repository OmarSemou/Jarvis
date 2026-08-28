from dataclasses import dataclass

from jarvis.audio.tts.base import SynthesisErrorCode, SynthesizedAudio
from jarvis.audio.tts.piper import PIPER_VOICES, PiperSettings, PiperTTS


@dataclass
class Chunk:
    audio_int16_bytes: bytes = b"\x00\x00" * 100
    sample_rate: int = 22_050
    sample_width: int = 2
    sample_channels: int = 1


class Engine:
    def __init__(self, chunks=None, error=None):
        self.chunks = [Chunk()] if chunks is None else chunks
        self.error = error

    def synthesize(self, _text, *, syn_config):
        if self.error:
            raise self.error
        return iter(self.chunks)


def _provider(tmp_path, *, engine=None, package=True):
    files = {}
    for voice in PIPER_VOICES:
        model = tmp_path / f"{voice}.onnx"
        config = tmp_path / f"{voice}.onnx.json"
        model.write_bytes(b"model")
        config.write_text("{}", encoding="utf-8")
        files[voice] = (model.resolve(), config.resolve())
    engine = engine or Engine()
    configs = []
    provider = PiperTTS(
        PiperSettings(files),
        engine_loader=lambda *_paths: engine,
        synthesis_config_factory=lambda **values: configs.append(values) or values,
        package_probe=lambda _name: package,
    )
    return provider, files, configs


def test_piper_returns_same_provider_neutral_pcm16_interface(tmp_path):
    provider, _, configs = _provider(tmp_path)
    result = provider.synthesize("Alright.", voice="en_US-joe-medium", speed=1.25)

    assert result.success
    assert isinstance(result.audio, SynthesizedAudio)
    assert result.audio.sample_rate == 22_050
    assert result.first_audio_seconds is not None
    assert configs == [{"length_scale": 0.8}]


def test_piper_missing_package_model_and_config_fail_cleanly(tmp_path):
    provider, files, _ = _provider(tmp_path, package=False)
    assert provider.synthesize("Hi", voice=PIPER_VOICES[0]).error.code is SynthesisErrorCode.PROVIDER_UNAVAILABLE

    provider, files, _ = _provider(tmp_path)
    files[PIPER_VOICES[0]][0].unlink()
    assert provider.synthesize("Hi", voice=PIPER_VOICES[0]).error.code is SynthesisErrorCode.MODEL_MISSING
    files[PIPER_VOICES[0]][0].write_bytes(b"model")
    files[PIPER_VOICES[0]][1].unlink()
    assert provider.synthesize("Hi", voice=PIPER_VOICES[0]).error.code is SynthesisErrorCode.VOICE_MISSING


def test_piper_rejects_invalid_voice_and_handles_synthesis_failure(tmp_path):
    provider, _, _ = _provider(tmp_path)
    assert provider.synthesize("Hi", voice="arbitrary").error.code is SynthesisErrorCode.INVALID_VOICE

    provider, _, _ = _provider(tmp_path, engine=Engine(error=RuntimeError("broken")))
    result = provider.synthesize("Hi", voice=PIPER_VOICES[0])
    assert result.error.code is SynthesisErrorCode.SYNTHESIS_FAILED
    assert "broken" in result.error.message


def test_piper_rejects_mixed_audio_formats(tmp_path):
    provider, _, _ = _provider(
        tmp_path,
        engine=Engine(chunks=[Chunk(), Chunk(sample_rate=16_000)]),
    )
    result = provider.synthesize("Hi", voice=PIPER_VOICES[0])
    assert result.error.code is SynthesisErrorCode.SYNTHESIS_FAILED
