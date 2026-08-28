import wave

import pytest

from jarvis.audio.devices import MicrophoneDeviceService
from jarvis.audio.formats import inspect_wav
from jarvis.audio.recorder import PushToTalkRecorder, RecordingError
from jarvis.audio.service import VoiceInputService
from jarvis.audio.stt.base import TranscriptionResult


class FakeStream:
    def __init__(self, callback, chunks):
        self.callback = callback
        self.chunks = chunks
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True
        for chunk in self.chunks:
            self.callback(chunk, len(chunk) // 2, None, None)

    def stop(self):
        self.stopped = True

    def abort(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeSoundDevice:
    def __init__(self, chunks, supported_rate=48_000):
        self.default = type("Default", (), {"device": (3, -1)})()
        self.chunks = chunks
        self.supported_rate = supported_rate
        self.stream = None
        self.settings_checks = []

    def query_devices(self):
        return [
            {"name": "Output", "max_input_channels": 0, "default_samplerate": 48_000},
            {"name": "Other output", "max_input_channels": 0, "default_samplerate": 48_000},
            {"name": "Unused", "max_input_channels": 0, "default_samplerate": 48_000},
            {"name": "Test Mic", "max_input_channels": 1, "default_samplerate": self.supported_rate},
        ]

    def check_input_settings(self, **kwargs):
        self.settings_checks.append(kwargs)
        if kwargs["samplerate"] != self.supported_rate:
            raise RuntimeError("unsupported rate")

    def RawInputStream(self, **kwargs):
        self.stream = FakeStream(kwargs["callback"], self.chunks)
        return self.stream


def make_recorder(tmp_path, chunks, supported_rate=48_000):
    module = FakeSoundDevice(chunks, supported_rate)
    devices = MicrophoneDeviceService(module_loader=lambda: module)
    recorder = PushToTalkRecorder(devices, tmp_path / "data" / "recordings")
    return recorder, module


def test_clean_start_stop_produces_private_mono_pcm16_16khz_wav(tmp_path):
    samples = b"\x00\x00\x10\x00" * 240
    recorder, module = make_recorder(tmp_path, [samples])

    session = recorder.start()
    recording = recorder.stop()

    assert session.capture_sample_rate == 48_000
    assert module.stream.started and module.stream.stopped and module.stream.closed
    assert recording.path.is_relative_to((tmp_path / "data" / "recordings").resolve())
    info = inspect_wav(recording.path)
    assert (info.channels, info.sample_width, info.sample_rate) == (1, 2, 16_000)
    assert info.frame_count == 160


def test_direct_16khz_capture_does_not_require_resampling(tmp_path):
    recorder, _module = make_recorder(tmp_path, [b"\x01\x00" * 160], supported_rate=16_000)
    recorder.start()
    recording = recorder.stop()

    with wave.open(str(recording.path), "rb") as wav_file:
        assert wav_file.readframes(160) == b"\x01\x00" * 160


def test_empty_recording_fails_without_creating_wav(tmp_path):
    recorder, _module = make_recorder(tmp_path, [])
    recorder.start()
    with pytest.raises(RecordingError, match="empty"):
        recorder.stop()
    assert not (tmp_path / "data").exists()


class FakeSTT:
    name = "fake-stt"

    def __init__(self, result):
        self.result = result
        self.paths = []

    def readiness_error(self):
        return None

    def transcribe(self, path):
        self.paths.append(path)
        assert path.is_file()
        return self.result


def test_recording_is_deleted_after_transcription_by_default(tmp_path):
    recorder, _module = make_recorder(tmp_path, [b"\x00\x00" * 160], supported_rate=16_000)
    stt = FakeSTT(TranscriptionResult(True, "hello", "fake-stt", 0.1, 0.01))
    voice = VoiceInputService(recorder.devices, recorder, stt, retain_recordings=False)

    voice.start()
    outcome = voice.finish()

    assert outcome.transcription.text == "hello"
    assert not stt.paths[0].exists()
    assert outcome.retained_recording is None


def test_recording_is_deleted_after_failed_transcription_by_default(tmp_path):
    recorder, _module = make_recorder(tmp_path, [b"\x00\x00" * 160], supported_rate=16_000)
    stt = FakeSTT(TranscriptionResult(False, "", "fake-stt", 0.1, 0.01))
    voice = VoiceInputService(recorder.devices, recorder, stt, retain_recordings=False)

    voice.start()
    outcome = voice.finish()

    assert not outcome.transcription.success
    assert not stt.paths[0].exists()


def test_recording_is_retained_only_when_explicitly_configured(tmp_path):
    recorder, _module = make_recorder(tmp_path, [b"\x00\x00" * 160], supported_rate=16_000)
    stt = FakeSTT(TranscriptionResult(True, "hello", "fake-stt", 0.1, 0.01))
    voice = VoiceInputService(recorder.devices, recorder, stt, retain_recordings=True)

    voice.start()
    outcome = voice.finish()

    assert outcome.retained_recording is not None
    assert outcome.retained_recording.is_file()
