"""Read-only microphone discovery and deterministic device selection.

``sounddevice`` is imported only when an explicit device operation is requested.
Importing this module never probes or opens audio hardware.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class MicrophoneError(RuntimeError):
    """Base class for clean microphone/device failures."""


class SoundDeviceUnavailableError(MicrophoneError):
    """Raised when the local capture dependency is unavailable."""


class MicrophoneUnavailableError(MicrophoneError):
    """Raised when no usable configured input device can be found."""


@dataclass(frozen=True, slots=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: int
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class MicrophoneStatus:
    available: bool
    selected: AudioDevice | None
    configured: int | str | None
    detail: str


SoundDeviceLoader = Callable[[], Any]


def load_sounddevice() -> Any:
    """Load the optional capture backend without touching a device."""

    try:
        return importlib.import_module("sounddevice")
    except (ImportError, OSError) as exc:
        raise SoundDeviceUnavailableError(
            "Microphone support is not installed. Install the project requirements with "
            ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc


def _default_input_index(module: Any) -> int | None:
    value = getattr(getattr(module, "default", None), "device", None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = value[0] if value else None
    elif not isinstance(value, int):
        try:
            value = value[0]
        except (IndexError, KeyError, TypeError):
            value = None
    if isinstance(value, int) and value >= 0:
        return value
    return None


class MicrophoneDeviceService:
    """List and select input devices without opening a recording stream."""

    def __init__(
        self,
        configured_device: int | str | None = None,
        *,
        module_loader: SoundDeviceLoader = load_sounddevice,
    ) -> None:
        self.configured_device = configured_device
        self._module_loader = module_loader

    def backend(self) -> Any:
        return self._module_loader()

    def list_inputs(self) -> tuple[AudioDevice, ...]:
        module = self.backend()
        try:
            raw_devices = module.query_devices()
        except Exception as exc:
            raise MicrophoneUnavailableError(f"Could not query microphone devices: {exc}") from exc

        default_index = _default_input_index(module)
        devices: list[AudioDevice] = []
        for index, raw in enumerate(raw_devices):
            try:
                channels = int(raw.get("max_input_channels", 0))
            except (AttributeError, TypeError, ValueError):
                continue
            if channels < 1:
                continue
            name = str(raw.get("name", f"Input {index}")).strip() or f"Input {index}"
            try:
                rate = int(round(float(raw.get("default_samplerate", 0))))
            except (TypeError, ValueError):
                rate = 0
            devices.append(AudioDevice(index, name, channels, rate, index == default_index))
        return tuple(devices)

    def selected_input(self) -> AudioDevice:
        devices = self.list_inputs()
        if not devices:
            raise MicrophoneUnavailableError("No microphone input devices are available.")

        requested = self.configured_device
        if requested is None or (isinstance(requested, str) and requested.casefold() == "default"):
            selected = next((device for device in devices if device.is_default), None)
            if selected is None:
                raise MicrophoneUnavailableError(
                    "No default microphone is configured. Set 'input_device' to an index or name."
                )
            return selected

        if isinstance(requested, int) or (isinstance(requested, str) and requested.isdecimal()):
            index = int(requested)
            selected = next((device for device in devices if device.index == index), None)
            if selected is None:
                raise MicrophoneUnavailableError(
                    f"Configured microphone index {index} is unavailable or disconnected."
                )
            return selected

        needle = str(requested).strip().casefold()
        exact = [device for device in devices if device.name.casefold() == needle]
        matches = exact or [device for device in devices if needle in device.name.casefold()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise MicrophoneUnavailableError(
                f"Configured microphone '{requested}' is unavailable or disconnected."
            )
        raise MicrophoneUnavailableError(
            f"Configured microphone name '{requested}' is ambiguous; use its numeric index."
        )

    def status(self) -> MicrophoneStatus:
        try:
            selected = self.selected_input()
        except MicrophoneError as exc:
            return MicrophoneStatus(False, None, self.configured_device, str(exc))
        selection = "default input" if self.configured_device in (None, "default") else "configured input"
        return MicrophoneStatus(True, selected, self.configured_device, selection)


def format_device_list(devices: Sequence[AudioDevice]) -> str:
    if not devices:
        return "No microphone input devices are available."
    lines = ["Microphone inputs"]
    for device in devices:
        marker = "*" if device.is_default else " "
        rate = f", {device.default_sample_rate} Hz" if device.default_sample_rate else ""
        lines.append(
            f"{marker} {device.index}: {device.name} "
            f"({device.max_input_channels} input channel(s){rate})"
        )
    lines.append("* = system default input")
    return "\n".join(lines)


def format_microphone_status(status: MicrophoneStatus) -> str:
    if not status.available or status.selected is None:
        return f"Microphone unavailable\n{status.detail}"
    device = status.selected
    configured = "default" if status.configured is None else str(status.configured)
    return (
        "Microphone ready\n"
        f"Device: {device.index}: {device.name}\n"
        f"Selection: {configured} ({status.detail})\n"
        f"Default sample rate: {device.default_sample_rate or 'unknown'} Hz"
    )
