from types import SimpleNamespace

from jarvis.audio.devices import AudioDevice
from jarvis.audio.realtime import (
    RealtimeAudioFrame,
    RollingAudioFrameBuffer,
    SoundDeviceRealtimeInput,
)


class Stream:
    def __init__(self, **settings):
        self.settings = settings
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def abort(self):
        self.started = False

    def close(self):
        self.closed = True


class Module:
    def __init__(self):
        self.stream = None

    def check_input_settings(self, **_settings):
        pass

    def RawInputStream(self, **settings):
        self.stream = Stream(**settings)
        return self.stream


class Devices:
    def __init__(self, module):
        self.module = module

    def backend(self):
        return self.module

    def selected_input(self):
        return AudioDevice(2, "Mock Mic", 1, 16_000, True)


def test_realtime_input_opens_only_on_start_and_emits_fixed_local_frames(tmp_path):
    module = Module()
    source = SoundDeviceRealtimeInput(Devices(module), frame_duration_ms=30)
    before = list(tmp_path.iterdir())

    assert module.stream is None
    source.start()
    assert module.stream.started
    callback = module.stream.settings["callback"]
    callback(b"\x01\x00" * 480, 480, None, None)
    frame = source.read()

    assert frame.sample_rate == 16_000
    assert frame.duration_seconds == 0.03
    assert len(frame.pcm16) == 960
    assert frame.sequence == 0
    assert list(tmp_path.iterdir()) == before
    source.stop()
    assert module.stream.closed


def test_realtime_drain_discards_buffered_room_audio_without_returning_it():
    module = Module()
    source = SoundDeviceRealtimeInput(Devices(module), frame_duration_ms=30)
    source.start()
    callback = module.stream.settings["callback"]
    callback(b"\x01\x00" * 480, 480, None, None)
    callback(b"\x02\x00" * 480, 480, None, None)

    assert source.drain() == 2
    assert source._raw_queue.empty()
    assert source._normalized == bytearray()
    source.stop()


def test_wake_barge_rolling_buffer_is_bounded_resettable_and_memory_only(tmp_path):
    buffer = RollingAudioFrameBuffer(max_duration_ms=320, frame_duration_ms=80)
    before = list(tmp_path.iterdir())

    for index in range(10):
        buffer.append(
            RealtimeAudioFrame(
                bytes([index, 0]) * 1_280,
                (index + 1) * 0.08,
                0.08,
                sequence=index,
            )
        )

    frames = buffer.snapshot()
    assert buffer.max_frames == 4
    assert len(buffer) == 4
    assert [item.sequence for item in frames] == [6, 7, 8, 9]
    assert buffer.duration_ms == 320
    assert list(tmp_path.iterdir()) == before

    buffer.clear()
    assert buffer.snapshot() == ()
