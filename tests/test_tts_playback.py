from types import SimpleNamespace
from threading import Event

from jarvis.audio.tts.base import SynthesizedAudio
from jarvis.audio.tts.playback import (
    AudioPlaybackService,
    PlaybackErrorCode,
    SpeakerBackendUnavailableError,
    format_speaker_list,
    format_speaker_status,
)


class Stream:
    def __init__(self, **settings):
        self.settings = settings
        self.events = []

    def start(self):
        self.events.append("start")

    def write(self, data):
        self.events.append(("write", data))

    def stop(self):
        self.events.append("stop")

    def abort(self):
        self.events.append("abort")

    def close(self):
        self.events.append("close")


class Module:
    def __init__(self, devices=None, stream_error=None):
        self.default = SimpleNamespace(device=(0, 1))
        self.devices = devices if devices is not None else [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speakers", "max_output_channels": 2, "default_samplerate": 48_000},
        ]
        self.stream_error = stream_error
        self.streams = []

    def query_devices(self):
        return self.devices

    def RawOutputStream(self, **settings):
        if self.stream_error:
            raise self.stream_error
        stream = Stream(**settings)
        self.streams.append(stream)
        return stream


def test_output_listing_and_selection_are_read_only():
    module = Module()
    playback = AudioPlaybackService(module_loader=lambda: module)

    devices = playback.list_outputs()
    status = playback.status()

    assert len(devices) == 1
    assert devices[0].name == "Speakers"
    assert status.available and status.selected.is_default
    assert not module.streams
    assert "Speaker outputs" in format_speaker_list(devices)
    assert "Speaker ready" in format_speaker_status(status)


def test_missing_speaker_fails_without_opening_stream():
    module = Module(devices=[])
    playback = AudioPlaybackService(module_loader=lambda: module)
    result = playback.play(SynthesizedAudio(b"\x00\x00", 24_000))

    assert not result.success
    assert result.error.code is PlaybackErrorCode.DEVICE_MISSING
    assert not module.streams


def test_missing_sounddevice_backend_has_distinct_structured_error():
    def unavailable():
        raise SpeakerBackendUnavailableError("sounddevice missing")

    result = AudioPlaybackService(module_loader=unavailable).play(
        SynthesizedAudio(b"\x00\x00", 24_000)
    )
    assert result.error.code is PlaybackErrorCode.BACKEND_UNAVAILABLE


def test_pcm16_playback_is_synchronous_in_memory_and_closes_stream():
    module = Module()
    playback = AudioPlaybackService(module_loader=lambda: module)
    audio = SynthesizedAudio(b"\x01\x00" * 10, 24_000)

    result = playback.play(audio)

    assert result.success
    stream = module.streams[0]
    assert stream.settings == {
        "samplerate": 24_000,
        "channels": 1,
        "dtype": "int16",
        "device": 1,
    }
    assert stream.events == ["start", ("write", audio.pcm16), "stop", "close"]


def test_playback_failure_is_structured_and_stream_cleanup_is_attempted():
    playback = AudioPlaybackService(
        module_loader=lambda: Module(stream_error=RuntimeError("backend failed"))
    )
    result = playback.play(SynthesizedAudio(b"\x00\x00", 24_000))
    assert result.error.code is PlaybackErrorCode.PLAYBACK_FAILED
    assert "backend failed" in result.error.message


def test_stop_and_cancel_are_safe_and_idempotent():
    playback = AudioPlaybackService(module_loader=lambda: Module())
    playback.stop()
    stream = Stream()
    playback._stream = stream
    playback.cancel()
    playback.cancel()
    assert stream.events == ["abort", "abort"]


def test_ambiguous_or_missing_configured_speaker_fails_closed():
    module = Module(
        devices=[
            {"name": "USB Audio A", "max_output_channels": 2},
            {"name": "USB Audio B", "max_output_channels": 2},
        ]
    )
    assert not AudioPlaybackService("USB", module_loader=lambda: module).status().available
    assert not AudioPlaybackService(99, module_loader=lambda: module).status().available


def test_background_playback_can_be_interrupted_and_reports_structured_result():
    release = Event()

    class BlockingStream(Stream):
        def write(self, data):
            self.events.append(("write", data))
            release.wait(2)

        def abort(self):
            self.events.append("abort")
            release.set()

    class BlockingModule(Module):
        def RawOutputStream(self, **settings):
            stream = BlockingStream(**settings)
            self.streams.append(stream)
            return stream

    module = BlockingModule()
    playback = AudioPlaybackService(module_loader=lambda: module)
    handle = playback.start(SynthesizedAudio(b"\x00\x00" * 2_000, 24_000))

    assert handle.wait_started(1)
    assert not handle.done
    handle.cancel()
    result = handle.wait(2)

    assert result is not None and not result.success
    assert result.error.code is PlaybackErrorCode.INTERRUPTED
    assert handle.finished_at is not None
    assert "abort" in module.streams[0].events


def test_background_playback_cancel_before_stream_start_never_plays_audio():
    entered = Event()
    release = Event()

    class SlowModule(Module):
        def query_devices(self):
            entered.set()
            release.wait(2)
            return self.devices

    module = SlowModule()
    playback = AudioPlaybackService(module_loader=lambda: module)
    handle = playback.start(SynthesizedAudio(b"\x00\x00" * 2_000, 24_000))
    assert entered.wait(1)

    handle.cancel()
    release.set()
    result = handle.wait(2)

    assert result is not None and result.error.code is PlaybackErrorCode.INTERRUPTED
    assert module.streams == []


def test_background_playback_is_chunked_for_bounded_cancel_latency():
    module = Module()
    playback = AudioPlaybackService(module_loader=lambda: module)
    audio = SynthesizedAudio(b"\x00\x00" * 2_000, 10_000)

    result = playback.play(audio)

    writes = [event[1] for event in module.streams[0].events if isinstance(event, tuple)]
    assert result.success
    assert len(writes) == 10
    assert all(len(chunk) <= 400 for chunk in writes)
    assert b"".join(writes) == audio.pcm16


def test_lazy_audio_sequence_uses_one_stream_and_preserves_exact_order():
    module = Module()
    playback = AudioPlaybackService(module_loader=lambda: module)
    first = SynthesizedAudio(b"\x01\x00" * 10, 24_000)
    second = SynthesizedAudio(b"\x02\x00" * 10, 24_000)
    played = []

    handle = playback.start_sequence(
        iter((first, second)),
        on_chunk_played=played.append,
    )
    result = handle.wait(1)

    assert result is not None and result.success
    assert len(module.streams) == 1
    writes = [event[1] for event in module.streams[0].events if isinstance(event, tuple)]
    assert writes == [first.pcm16, second.pcm16]
    assert played == [first, second]
