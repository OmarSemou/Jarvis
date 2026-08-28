from types import SimpleNamespace

import pytest

from jarvis.audio.devices import (
    MicrophoneDeviceService,
    MicrophoneUnavailableError,
    format_device_list,
    format_microphone_status,
)


class FakeSoundDevice:
    def __init__(self, devices, default_input=0):
        self._devices = devices
        self.default = SimpleNamespace(device=(default_input, -1))
        self.query_count = 0

    def query_devices(self):
        self.query_count += 1
        return self._devices


DEVICES = [
    {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48_000},
    {"name": "USB Microphone", "max_input_channels": 2, "default_samplerate": 48_000},
    {"name": "Webcam Mic", "max_input_channels": 1, "default_samplerate": 44_100},
]


def service(configured=None, default_input=1):
    module = FakeSoundDevice(DEVICES, default_input)
    return MicrophoneDeviceService(configured, module_loader=lambda: module), module


def test_lists_only_input_devices_and_marks_default():
    selected_service, selected_module = service()
    assert selected_module.query_count == 0
    devices = selected_service.list_inputs()

    assert [device.index for device in devices] == [1, 2]
    assert devices[0].is_default is True
    assert selected_module.query_count == 1
    rendered = format_device_list(devices)
    assert "USB Microphone" in rendered
    assert "* = system default input" in rendered


@pytest.mark.parametrize(
    ("configured", "expected_index"),
    [(None, 1), (2, 2), ("2", 2), ("Webcam", 2), ("USB Microphone", 1)],
)
def test_configured_device_selection(configured, expected_index):
    selected, _module = service(configured)
    assert selected.selected_input().index == expected_index


def test_no_input_device_is_reported_cleanly():
    module = FakeSoundDevice([DEVICES[0]], default_input=-1)
    selected = MicrophoneDeviceService(module_loader=lambda: module)

    with pytest.raises(MicrophoneUnavailableError, match="No microphone"):
        selected.selected_input()
    assert "Microphone unavailable" in format_microphone_status(selected.status())


def test_disconnected_configured_device_is_reported_cleanly():
    selected, _module = service(99)
    with pytest.raises(MicrophoneUnavailableError, match="unavailable or disconnected"):
        selected.selected_input()
