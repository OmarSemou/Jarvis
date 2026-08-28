"""Explicit push-to-talk microphone recording with no import-time effects."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .devices import AudioDevice, MicrophoneDeviceService, MicrophoneError
from .formats import TARGET_SAMPLE_RATE, resample_pcm16_mono, write_pcm16_mono_wav


class RecordingError(RuntimeError):
    """Raised for expected recording lifecycle and device failures."""


@dataclass(frozen=True, slots=True)
class RecordingSession:
    device: AudioDevice
    capture_sample_rate: int


@dataclass(frozen=True, slots=True)
class AudioRecording:
    path: Path
    duration_seconds: float
    sample_rate: int = TARGET_SAMPLE_RATE


class PushToTalkRecorder:
    """Capture mono PCM16 audio until the caller explicitly stops it."""

    def __init__(
        self,
        devices: MicrophoneDeviceService,
        recordings_dir: Path,
        *,
        preferred_sample_rate: int | None = None,
    ) -> None:
        self.devices = devices
        self.recordings_dir = recordings_dir.resolve()
        self.preferred_sample_rate = preferred_sample_rate
        self._stream: Any | None = None
        self._chunks: list[bytes] = []
        self._statuses: list[str] = []
        self._session: RecordingSession | None = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _capture_rate(self, module: Any, device: AudioDevice) -> int:
        candidates = [self.preferred_sample_rate, TARGET_SAMPLE_RATE, device.default_sample_rate]
        seen: set[int] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                module.check_input_settings(
                    device=device.index,
                    samplerate=candidate,
                    channels=1,
                    dtype="int16",
                )
            except Exception:
                continue
            return candidate
        raise RecordingError(
            f"Microphone '{device.name}' has no supported mono PCM16 capture rate."
        )

    def start(self) -> RecordingSession:
        if self.is_recording:
            raise RecordingError("A microphone recording is already active.")
        try:
            module = self.devices.backend()
            device = self.devices.selected_input()
            sample_rate = self._capture_rate(module, device)
        except MicrophoneError as exc:
            raise RecordingError(str(exc)) from exc

        self._chunks = []
        self._statuses = []

        def callback(indata: Any, _frames: int, _time_info: Any, status: Any) -> None:
            if status:
                self._statuses.append(str(status))
            self._chunks.append(bytes(indata))

        stream = None
        try:
            stream = module.RawInputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                device=device.index,
                callback=callback,
            )
            stream.start()
        except Exception as exc:
            self._chunks = []
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            raise RecordingError(f"Could not start microphone recording: {exc}") from exc

        self._stream = stream
        self._session = RecordingSession(device, sample_rate)
        return self._session

    def stop(self) -> AudioRecording:
        stream = self._stream
        session = self._session
        if stream is None or session is None:
            raise RecordingError("No microphone recording is active.")
        self._stream = None
        self._session = None
        stop_error: Exception | None = None
        try:
            stream.stop()
        except Exception as exc:
            stop_error = exc
        finally:
            try:
                stream.close()
            except Exception as exc:
                stop_error = stop_error or exc
        if stop_error is not None:
            self._chunks = []
            raise RecordingError(
                f"Could not stop microphone recording cleanly: {stop_error}"
            ) from stop_error

        raw = b"".join(self._chunks)
        self._chunks = []
        if self._statuses:
            detail = "; ".join(dict.fromkeys(self._statuses))
            self._statuses = []
            raise RecordingError(f"Microphone recording reported an error: {detail}")
        if not raw:
            raise RecordingError("The microphone recording was empty.")

        try:
            pcm16 = resample_pcm16_mono(raw, session.capture_sample_rate)
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix="voice-",
                suffix=".wav",
                dir=self.recordings_dir,
                delete=False,
            ) as temporary:
                path = Path(temporary.name).resolve()
            try:
                write_pcm16_mono_wav(path, pcm16)
            except Exception:
                path.unlink(missing_ok=True)
                raise
        except Exception as exc:
            raise RecordingError(f"Could not create a 16 kHz WAV recording: {exc}") from exc

        duration = len(pcm16) / (TARGET_SAMPLE_RATE * 2)
        return AudioRecording(path, duration)

    def cancel(self) -> None:
        stream = self._stream
        self._stream = None
        self._session = None
        self._chunks = []
        self._statuses = []
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            try:
                stream.stop()
            except Exception:
                pass
        finally:
            try:
                stream.close()
            except Exception:
                pass
