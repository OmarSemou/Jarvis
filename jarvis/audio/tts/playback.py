"""Provider-neutral PCM16 playback through a lazily loaded local audio device."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .base import SynthesizedAudio


class PlaybackErrorCode(StrEnum):
    BACKEND_UNAVAILABLE = "backend_unavailable"
    DEVICE_MISSING = "device_missing"
    PLAYBACK_FAILED = "playback_failed"


@dataclass(frozen=True, slots=True)
class PlaybackFailure:
    code: PlaybackErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class PlaybackResult:
    success: bool
    device: str | None = None
    error: PlaybackFailure | None = None


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
    """Select an output and synchronously play in-memory PCM16 audio.

    ``stop`` is deliberately public and idempotent so a future conversational
    coordinator can add interruption without changing the provider contracts.
    """

    def __init__(
        self,
        configured_device: int | str | None = None,
        *,
        module_loader: SoundDeviceLoader = load_sounddevice,
    ) -> None:
        self.configured_device = configured_device
        self._module_loader = module_loader
        self._stream: Any | None = None

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

    def play(self, audio: SynthesizedAudio) -> PlaybackResult:
        try:
            device = self.selected_output()
            if device.max_output_channels < audio.channels:
                raise SpeakerUnavailableError(
                    f"Speaker '{device.name}' does not support {audio.channels} channels."
                )
            module = self.backend()
            stream = module.RawOutputStream(
                samplerate=audio.sample_rate,
                channels=audio.channels,
                dtype="int16",
                device=device.index,
            )
            self._stream = stream
            stream.start()
            stream.write(audio.pcm16)
            stream.stop()
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
            return PlaybackResult(
                False,
                error=PlaybackFailure(
                    PlaybackErrorCode.PLAYBACK_FAILED,
                    f"Local audio playback failed: {exc}",
                ),
            )
        finally:
            stream = self._stream
            self._stream = None
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    def stop(self) -> None:
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
