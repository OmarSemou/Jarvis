"""Push-to-talk orchestration and private recording lifecycle."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .devices import MicrophoneDeviceService
from .formats import TARGET_SAMPLE_RATE, write_pcm16_mono_wav
from .recorder import PushToTalkRecorder, RecordingSession
from .stt.base import STTProvider, TranscriptionResult


class VoiceInputError(RuntimeError):
    """Raised for a clean, user-facing voice input failure."""


@dataclass(frozen=True, slots=True)
class VoiceInputOutcome:
    transcription: TranscriptionResult
    retained_recording: Path | None = None
    cleanup_warning: str | None = None


class VoiceInputService:
    """Coordinate one recording and transcription without involving the LLM."""

    def __init__(
        self,
        devices: MicrophoneDeviceService,
        recorder: PushToTalkRecorder,
        stt: STTProvider,
        *,
        retain_recordings: bool = False,
    ) -> None:
        self.devices = devices
        self.recorder = recorder
        self.stt = stt
        self.retain_recordings = retain_recordings

    def start(self) -> RecordingSession:
        readiness_error = self.stt.readiness_error()
        if readiness_error is not None:
            raise VoiceInputError(readiness_error.message)
        return self.recorder.start()

    def finish(self) -> VoiceInputOutcome:
        recording = self.recorder.stop()
        return self._transcribe_temporary(recording.path)

    def transcribe_pcm16(self, pcm16: bytes) -> VoiceInputOutcome:
        """Transcribe one in-memory 16 kHz utterance through a private temp WAV."""

        readiness_error = self.stt.readiness_error()
        if readiness_error is not None:
            raise VoiceInputError(readiness_error.message)
        if not pcm16 or len(pcm16) % 2:
            raise VoiceInputError("Captured voice audio is empty or incomplete.")
        self.recorder.recordings_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                prefix="voice-auto-",
                suffix=".wav",
                dir=self.recorder.recordings_dir,
                delete=False,
            ) as temporary:
                path = Path(temporary.name).resolve()
            try:
                write_pcm16_mono_wav(path, pcm16, TARGET_SAMPLE_RATE)
            except Exception:
                path.unlink(missing_ok=True)
                raise
        except Exception as exc:
            raise VoiceInputError(f"Could not create a private temporary WAV: {exc}") from exc
        return self._transcribe_temporary(path)

    def _transcribe_temporary(self, path: Path) -> VoiceInputOutcome:
        cleanup_warning: str | None = None
        try:
            transcription = self.stt.transcribe(path)
        finally:
            if not self.retain_recordings:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    cleanup_warning = (
                        f"Private recording cleanup failed for {path}: {exc}"
                    )
        retained = path if self.retain_recordings else None
        return VoiceInputOutcome(transcription, retained, cleanup_warning)

    def cancel(self) -> None:
        self.recorder.cancel()
