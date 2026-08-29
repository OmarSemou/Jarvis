from pathlib import Path

from jarvis.audio.service import VoiceInputService
from jarvis.audio.stt.base import TranscriptionResult


class Recorder:
    def __init__(self, recordings_dir: Path):
        self.recordings_dir = recordings_dir

    def cancel(self):
        pass


class STT:
    name = "mock"

    def __init__(self):
        self.paths = []

    def readiness_error(self):
        return None

    def transcribe(self, path):
        self.paths.append(path)
        assert path.is_file()
        return TranscriptionResult(True, "hello", self.name, 0.1, 0.2)


def test_continuous_pcm_is_temporary_and_deleted_after_stt(tmp_path):
    stt = STT()
    service = VoiceInputService(object(), Recorder(tmp_path / "recordings"), stt)

    outcome = service.transcribe_pcm16(b"\x01\x00" * 3_200)

    assert outcome.transcription.text == "hello"
    assert outcome.retained_recording is None
    assert len(stt.paths) == 1
    assert not stt.paths[0].exists()


def test_continuous_pcm_can_only_be_retained_by_existing_explicit_setting(tmp_path):
    stt = STT()
    service = VoiceInputService(
        object(), Recorder(tmp_path / "recordings"), stt, retain_recordings=True
    )

    outcome = service.transcribe_pcm16(b"\x01\x00" * 1_600)

    assert outcome.retained_recording is not None
    assert outcome.retained_recording.is_file()
