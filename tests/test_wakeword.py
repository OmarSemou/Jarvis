from types import SimpleNamespace

import pytest

from jarvis.audio.wake.base import WakeWordErrorCode
from jarvis.audio.wake.openwakeword import OpenWakeWord, OpenWakeWordSettings


class Engine:
    def __init__(self, scores=(0.2, 0.8)):
        self.scores = iter(scores)
        self.inputs = []
        self.resets = 0

    def predict(self, samples):
        self.inputs.append(samples)
        return {"hey_jarvis": next(self.scores)}

    def reset(self):
        self.resets += 1


def _settings(tmp_path, threshold=0.5):
    paths = [tmp_path / name for name in ("wake.onnx", "mel.onnx", "embedding.onnx")]
    for path in paths:
        path.write_bytes(b"model")
    return OpenWakeWordSettings(*(path.resolve() for path in paths), threshold)


def test_wake_adapter_returns_provider_neutral_detection_and_applies_threshold(tmp_path):
    engine = Engine()
    provider = OpenWakeWord(
        _settings(tmp_path),
        engine_loader=lambda *_paths: engine,
        package_probe=lambda _name: True,
    )

    first = provider.process(b"\x00\x00" * 1_280)
    second = provider.process(b"\x00\x00" * 1_280)

    assert not first.detected and second.detected
    assert second.provider == "openwakeword"
    assert second.phrase == "Hey Jarvis"
    assert second.score == 0.8
    assert provider._engine is engine


def test_wake_adapter_is_lazy_and_warmup_resets_without_device_access(tmp_path):
    loads = []
    engine = Engine((0.0,))
    provider = OpenWakeWord(
        _settings(tmp_path),
        engine_loader=lambda *paths: loads.append(paths) or engine,
        package_probe=lambda _name: True,
    )

    assert provider.readiness_error() is None
    assert loads == []
    assert provider.warmup() is None
    assert len(loads) == 1
    assert engine.resets == 1


def test_wake_missing_package_or_model_fails_without_loading(tmp_path):
    settings = _settings(tmp_path)
    settings.model_path.unlink()
    provider = OpenWakeWord(settings, package_probe=lambda _name: False)

    failure = provider.readiness_error()

    assert failure.code is WakeWordErrorCode.PROVIDER_UNAVAILABLE
    assert provider._engine is None


@pytest.mark.parametrize("threshold", [0.0, 1.0, float("inf")])
def test_wake_threshold_validation(threshold, tmp_path):
    with pytest.raises(ValueError, match="threshold"):
        _settings(tmp_path, threshold)


def test_invalid_wake_audio_is_structured_and_never_loads_model(tmp_path):
    provider = OpenWakeWord(
        _settings(tmp_path),
        engine_loader=lambda *_: (_ for _ in ()).throw(AssertionError("loaded")),
        package_probe=lambda _name: True,
    )

    detection = provider.process(b"\x00")

    assert not detection.detected
    assert detection.error.code is WakeWordErrorCode.INVALID_AUDIO
    assert provider._engine is None
