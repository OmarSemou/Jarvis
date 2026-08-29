"""Provider-neutral PCM16 playback through a lazily loaded local audio device."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any

from .base import SynthesizedAudio


class PlaybackErrorCode(StrEnum):
    BACKEND_UNAVAILABLE = "backend_unavailable"
    DEVICE_MISSING = "device_missing"
    PLAYBACK_FAILED = "playback_failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class PlaybackFailure:
    code: PlaybackErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class PlaybackResult:
    success: bool
    device: str | None = None
    error: PlaybackFailure | None = None


class PlaybackHandle:
    """Observable handle for one background playback operation."""

    def __init__(self, service: "AudioPlaybackService") -> None:
        self._service = service
        self._started = Event()
        self._done = Event()
        self._cancelled = Event()
        self._result: PlaybackResult | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None

    @property
    def started(self) -> bool:
        return self._started.is_set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def started_at(self) -> float | None:
        return self._started_at

    @property
    def finished_at(self) -> float | None:
        return self._finished_at

    def wait_started(self, timeout_seconds: float | None = None) -> bool:
        return self._started.wait(timeout_seconds)

    def wait(self, timeout_seconds: float | None = None) -> PlaybackResult | None:
        return self._result if self._done.wait(timeout_seconds) else None

    def stop(self) -> None:
        self._cancelled.set()
        self._service.stop()

    cancel = stop

    def _mark_started(self) -> None:
        self._started_at = perf_counter()
        self._started.set()

    def _finish(self, result: PlaybackResult) -> None:
        self._result = result
        self._finished_at = perf_counter()
        self._done.set()


@dataclass(frozen=True, slots=True)
class OutputDevice:
    index: int
    name: str
    max_output_channels: int
    default_sample_rate: int
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class SpeakerStatus:
    available: bool
    selected: OutputDevice | None
    configured: int | str | None
    detail: str


class SpeakerError(RuntimeError):
    """Base class for expected local output-device failures."""


class SpeakerUnavailableError(SpeakerError):
    """Raised when sounddevice or a requested output is unavailable."""


class SpeakerBackendUnavailableError(SpeakerError):
    """Raised when the local sounddevice backend is not installed."""


SoundDeviceLoader = Callable[[], Any]


def load_sounddevice() -> Any:
    try:
        return importlib.import_module("sounddevice")
    except ImportError as exc:
        raise SpeakerBackendUnavailableError(
            "Speaker support is not installed. Install the project requirements with "
            ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc


def _default_output_index(module: Any) -> int | None:
    value = getattr(getattr(module, "default", None), "device", None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = value[1] if len(value) > 1 else None
    elif not isinstance(value, int):
        try:
            value = value[1]
        except (IndexError, KeyError, TypeError):
            value = None
    return value if isinstance(value, int) and value >= 0 else None


class AudioPlaybackService:
    """Synchronously play PCM16 or expose a cancellable background handle."""

    def __init__(
        self,
        configured_device: int | str | None = None,
        *,
        module_loader: SoundDeviceLoader = load_sounddevice,
    ) -> None:
        self.configured_device = configured_device
        self._module_loader = module_loader
        self._stream: Any | None = None
        self._stream_lock = Lock()
        self._stop_requested = Event()
        self._active_handle: PlaybackHandle | None = None

    def backend(self) -> Any:
        return self._module_loader()

    def list_outputs(self) -> tuple[OutputDevice, ...]:
        module = self.backend()
        try:
            raw_devices = module.query_devices()
        except Exception as exc:
            raise SpeakerUnavailableError(f"Could not query speaker devices: {exc}") from exc
        default_index = _default_output_index(module)
        devices: list[OutputDevice] = []
        for index, raw in enumerate(raw_devices):
            try:
                channels = int(raw.get("max_output_channels", 0))
            except (AttributeError, TypeError, ValueError):
                continue
            if channels < 1:
                continue
            name = str(raw.get("name", f"Output {index}")).strip() or f"Output {index}"
            try:
                rate = int(round(float(raw.get("default_samplerate", 0))))
            except (TypeError, ValueError):
                rate = 0
            devices.append(OutputDevice(index, name, channels, rate, index == default_index))
        return tuple(devices)

    def selected_output(self) -> OutputDevice:
        devices = self.list_outputs()
        if not devices:
            raise SpeakerUnavailableError("No speaker output devices are available.")
        requested = self.configured_device
        if requested is None or (
            isinstance(requested, str) and requested.casefold() == "default"
        ):
            selected = next((device for device in devices if device.is_default), None)
            if selected is None:
                raise SpeakerUnavailableError(
                    "No default speaker is configured. Use /speaker use with an index or name."
                )
            return selected
        if isinstance(requested, int) or (
            isinstance(requested, str) and requested.isdecimal()
        ):
            index = int(requested)
            selected = next((device for device in devices if device.index == index), None)
            if selected is None:
                raise SpeakerUnavailableError(
                    f"Configured speaker index {index} is unavailable or disconnected."
                )
            return selected
        needle = str(requested).strip().casefold()
        exact = [device for device in devices if device.name.casefold() == needle]
        matches = exact or [device for device in devices if needle in device.name.casefold()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise SpeakerUnavailableError(
                f"Configured speaker '{requested}' is unavailable or disconnected."
            )
        raise SpeakerUnavailableError(
            f"Configured speaker name '{requested}' is ambiguous; use its numeric index."
        )

    def status(self) -> SpeakerStatus:
        try:
            selected = self.selected_output()
        except SpeakerError as exc:
            return SpeakerStatus(False, None, self.configured_device, str(exc))
        selection = (
            "default output"
            if self.configured_device in (None, "default")
            else "configured output"
        )
        return SpeakerStatus(True, selected, self.configured_device, selection)

    def play(
        self,
        audio: SynthesizedAudio,
        *,
        _handle: PlaybackHandle | None = None,
    ) -> PlaybackResult:
        return self.play_sequence((audio,), _handle=_handle)

    def play_sequence(
        self,
        audio_chunks: Iterable[SynthesizedAudio],
        *,
        _handle: PlaybackHandle | None = None,
        on_chunk_played: Callable[[SynthesizedAudio], None] | None = None,
    ) -> PlaybackResult:
        """Play a lazy ordered PCM sequence through one continuous stream."""

        self._stop_requested.clear()
        try:
            if _handle is not None and _handle._cancelled.is_set():
                return self._interrupted_result()
            chunks = iter(audio_chunks)
            try:
                first = next(chunks)
            except StopIteration:
                return PlaybackResult(
                    False,
                    error=PlaybackFailure(
                        PlaybackErrorCode.PLAYBACK_FAILED,
                        "No synthesized audio was available for playback.",
                    ),
                )
            device = self.selected_output()
            if device.max_output_channels < first.channels:
                raise SpeakerUnavailableError(
                    f"Speaker '{device.name}' does not support {first.channels} channels."
                )
            if self._stop_requested.is_set() or (
                _handle is not None and _handle._cancelled.is_set()
            ):
                return self._interrupted_result(device.name)
            module = self.backend()
            stream = module.RawOutputStream(
                samplerate=first.sample_rate,
                channels=first.channels,
                dtype="int16",
                device=device.index,
            )
            with self._stream_lock:
                self._stream = stream
            if self._stop_requested.is_set() or (
                _handle is not None and _handle._cancelled.is_set()
            ):
                return self._interrupted_result(device.name)
            stream.start()
            if _handle is not None:
                _handle._mark_started()
            for audio in chain((first,), chunks):
                if (
                    audio.sample_rate != first.sample_rate
                    or audio.channels != first.channels
                ):
                    raise ValueError(
                        "Synthesized audio format changed during queued playback."
                    )
                frame_bytes = 2 * audio.channels
                chunk_bytes = max(
                    frame_bytes,
                    round(audio.sample_rate * 0.02) * frame_bytes,
                )
                for offset in range(0, len(audio.pcm16), chunk_bytes):
                    if self._stop_requested.is_set() or (
                        _handle is not None and _handle._cancelled.is_set()
                    ):
                        return self._interrupted_result(device.name)
                    stream.write(audio.pcm16[offset : offset + chunk_bytes])
                if on_chunk_played is not None:
                    on_chunk_played(audio)
            stream.stop()
            if self._stop_requested.is_set():
                return self._interrupted_result(device.name)
            return PlaybackResult(True, device.name)
        except SpeakerBackendUnavailableError as exc:
            return PlaybackResult(
                False,
                error=PlaybackFailure(PlaybackErrorCode.BACKEND_UNAVAILABLE, str(exc)),
            )
        except SpeakerError as exc:
            return PlaybackResult(
                False,
                error=PlaybackFailure(PlaybackErrorCode.DEVICE_MISSING, str(exc)),
            )
        except Exception as exc:
            if self._stop_requested.is_set():
                return self._interrupted_result()
            return PlaybackResult(
                False,
                error=PlaybackFailure(
                    PlaybackErrorCode.PLAYBACK_FAILED,
                    f"Local audio playback failed: {exc}",
                ),
            )
        finally:
            with self._stream_lock:
                stream = self._stream
                self._stream = None
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    @staticmethod
    def _interrupted_result(device: str | None = None) -> PlaybackResult:
        return PlaybackResult(
            False,
            device,
            PlaybackFailure(
                PlaybackErrorCode.INTERRUPTED,
                "Local speech playback was interrupted.",
            ),
        )

    def start(self, audio: SynthesizedAudio) -> PlaybackHandle:
        """Play in a background thread so the coordinator can monitor the mic."""

        return self.start_sequence((audio,))

    def start_sequence(
        self,
        audio_chunks: Iterable[SynthesizedAudio],
        *,
        on_chunk_played: Callable[[SynthesizedAudio], None] | None = None,
    ) -> PlaybackHandle:
        """Consume a lazy sequence in one cancellable background stream."""

        with self._stream_lock:
            if self._active_handle is not None and not self._active_handle.done:
                raise SpeakerError("Another speech playback operation is already active.")
            handle = PlaybackHandle(self)
            self._active_handle = handle

        def worker() -> None:
            try:
                result = self.play_sequence(
                    audio_chunks,
                    _handle=handle,
                    on_chunk_played=on_chunk_played,
                )
            finally:
                with self._stream_lock:
                    if self._active_handle is handle:
                        self._active_handle = None
            handle._finish(result)

        Thread(target=worker, name="jarvis-tts-playback", daemon=True).start()
        return handle

    def stop(self) -> None:
        self._stop_requested.set()
        with self._stream_lock:
            stream = self._stream
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            try:
                stream.stop()
            except Exception:
                pass

    cancel = stop


def format_speaker_list(devices: Sequence[OutputDevice]) -> str:
    if not devices:
        return "No speaker output devices are available."
    lines = ["Speaker outputs"]
    for device in devices:
        marker = "*" if device.is_default else " "
        rate = f", {device.default_sample_rate} Hz" if device.default_sample_rate else ""
        lines.append(
            f"{marker} {device.index}: {device.name} "
            f"({device.max_output_channels} output channel(s){rate})"
        )
    lines.append("* = system default output")
    return "\n".join(lines)


def format_speaker_status(status: SpeakerStatus) -> str:
    if not status.available or status.selected is None:
        return f"Speaker unavailable\n{status.detail}"
    device = status.selected
    configured = "default" if status.configured is None else str(status.configured)
    return (
        "Speaker ready\n"
        f"Device: {device.index}: {device.name}\n"
        f"Selection: {configured} ({status.detail})\n"
        f"Default sample rate: {device.default_sample_rate or 'unknown'} Hz"
    )
