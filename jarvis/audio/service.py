"""Push-to-talk orchestration and private recording lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .devices import MicrophoneDeviceService
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
        cleanup_warning: str | None = None
        try:
            transcription = self.stt.transcribe(recording.path)
        finally:
            if not self.retain_recordings:
                try:
                    recording.path.unlink(missing_ok=True)
                except OSError as exc:
                    cleanup_warning = (
                        f"Private recording cleanup failed for {recording.path}: {exc}"
                    )
        retained = recording.path if self.retain_recordings else None
        return VoiceInputOutcome(transcription, retained, cleanup_warning)

    def cancel(self) -> None:
        self.recorder.cancel()
