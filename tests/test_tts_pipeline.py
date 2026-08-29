from collections import deque
from threading import Event, Lock
from time import monotonic
from types import SimpleNamespace

from jarvis.audio.tts.base import (
    SpeechAudioChunk,
    SpeechSynthesisResult,
    SynthesizedAudio,
    SynthesisErrorCode,
    SynthesisFailure,
    SynthesisStreamError,
)
from jarvis.audio.tts.chunks import SpeechChunker
from jarvis.audio.tts.pipeline import SpeechPipelineSettings
from jarvis.audio.tts.playback import AudioPlaybackService, PlaybackErrorCode
from jarvis.audio.tts.service import TTSService


class Stream:
    def __init__(self, events, *, block_first=None):
        self.events = events
        self.block_first = block_first
        self.writes = 0

    def start(self):
        self.events.append("start")

    def write(self, data):
        self.writes += 1
        self.events.append(bytes(data))
        if self.block_first is not None and self.writes == 1:
            self.block_first.wait(2)

    def stop(self):
        self.events.append("stop")

    def abort(self):
        self.events.append("abort")
        if self.block_first is not None:
            self.block_first.set()

    def close(self):
        self.events.append("close")


class Module:
    def __init__(self, *, block_first=None):
        self.default = SimpleNamespace(device=(0, 0))
        self.events = []
        self.streams = []
        self.block_first = block_first

    def query_devices(self):
        return [
            {
                "name": "Mock speakers",
                "max_output_channels": 1,
                "default_samplerate": 1_000,
            }
        ]

    def RawOutputStream(self, **_settings):
        stream = Stream(self.events, block_first=self.block_first)
        self.streams.append(stream)
        return stream


class FullProvider:
    available_voices = ("voice",)

    def __init__(self, name="piper"):
        self.name = name
        self.calls = []

    def readiness_error(self, _voice):
        return None

    def synthesize(self, text, *, voice, speed, language):
        self.calls.append(text)
        value = len(self.calls)
        return SpeechSynthesisResult(
            True,
            self.name,
            voice,
            0.01,
            SynthesizedAudio(bytes((value, 0)) * 20, 1_000),
        )


class ControlledStreamingProvider(FullProvider):
    def __init__(self, *, release_later=None, late_audio=None):
        super().__init__("kokoro")
        self.release_later = release_later
        self.late_audio = late_audio
        self.first_ready = Event()
        self.later_entered = Event()
        self.generated = []

    def synthesize_stream(
        self, text, *, voice, speed, language, cancellation=None
    ):
        self.calls.append(text)
        value = len(self.calls)
        if value > 1 and self.release_later is not None:
            self.later_entered.set()
            self.release_later.wait(2)
        audio = SynthesizedAudio(bytes((value, 0)) * 20, 1_000)
        self.generated.append((text, value))
        if value == 1:
            self.first_ready.set()
        if self.late_audio is not None and value > 1:
            self.late_audio.set()
        yield SpeechAudioChunk(audio, 0, True)


def service(provider, module, *, queue_size=2):
    piper = FullProvider()
    return TTSService(
        {"kokoro": provider, "piper": piper},
        AudioPlaybackService(module_loader=lambda: module),
        enabled=True,
        voice="voice",
        chunker=SpeechChunker(),
        pipeline_settings=SpeechPipelineSettings(queue_size=queue_size),
    )


def test_first_audio_starts_before_later_sentence_synthesis_finishes():
    release_later = Event()
    provider = ControlledStreamingProvider(release_later=release_later)
    module = Module()
    runtime = service(provider, module)

    handle = runtime.start_speech("First sentence. Second sentence.")

    assert handle.wait_started(1)
    assert provider.later_entered.wait(1)
    assert bytes((1, 0)) * 20 in module.events
    assert not handle.done
    release_later.set()
    result = handle.wait(2)

    assert result is not None and result.success
    assert provider.calls == ["First sentence.", "Second sentence."]
    writes = [event for event in module.events if isinstance(event, bytes)]
    assert writes == [bytes((1, 0)) * 20, bytes((2, 0)) * 20]
    assert result.metrics.queued_chunks == result.metrics.played_chunks == 2
    assert result.metrics.tts_first_chunk is not None
    assert result.metrics.tts_total_generation is not None
    assert result.metrics.first_audio_started < result.metrics.generation_finished


def test_bounded_queue_applies_backpressure_without_overlapping_playback():
    release_playback = Event()
    module = Module(block_first=release_playback)

    class ManyChunkProvider(FullProvider):
        def __init__(self):
            super().__init__("kokoro")
            self.generated = 0
            self.lock = Lock()

        def synthesize_stream(self, text, *, voice, speed, language, cancellation=None):
            del text, voice, speed, language
            for value in range(1, 7):
                with self.lock:
                    self.generated += 1
                yield SpeechAudioChunk(
                    SynthesizedAudio(bytes((value, 0)) * 20, 1_000),
                    value - 1,
                    value == 6,
                )

    provider = ManyChunkProvider()
    runtime = service(provider, module, queue_size=1)
    handle = runtime.start_speech("One semantic sentence.")

    assert handle.wait_started(1)
    deadline = monotonic() + 1
    while handle._queue.qsize() < 1 and monotonic() < deadline:
        Event().wait(0.01)
    assert handle._queue.maxsize == 1
    assert handle._queue.qsize() == 1
    assert provider.generated <= 3

    release_playback.set()
    result = handle.wait(2)
    assert result is not None and result.success
    assert result.metrics.played_chunks == 6
    assert len(module.streams) == 1


def test_cancel_clears_queue_discards_late_audio_and_next_generation_is_clean():
    release_playback = Event()
    release_later = Event()
    late_audio = Event()
    provider = ControlledStreamingProvider(
        release_later=release_later,
        late_audio=late_audio,
    )
    module = Module(block_first=release_playback)
    runtime = service(provider, module, queue_size=1)
    first = runtime.start_speech("Old first. Old late.")

    assert first.wait_started(1)
    assert provider.later_entered.wait(1)
    first.stop()
    first.stop()
    cancelled = first.wait(1)

    assert cancelled is not None and not cancelled.success
    assert cancelled.playback.error.code is PlaybackErrorCode.INTERRUPTED
    assert first._queue.qsize() <= 1

    release_later.set()
    assert late_audio.wait(1)
    Event().wait(0.05)
    old_writes = [event for event in module.events if isinstance(event, bytes)]
    assert bytes((2, 0)) * 20 not in old_writes

    # A fresh response uses a fresh monotonic identity and is unaffected by
    # provider work that returned late for the cancelled generation.
    module.block_first = None
    second = runtime.start_speech("New response.")
    completed = second.wait(2)
    assert completed is not None and completed.success
    assert second.generation_id > first.generation_id
    assert bytes((3, 0)) * 20 in module.events
    assert bytes((2, 0)) * 20 not in module.events


def test_piper_style_provider_falls_back_to_one_audio_chunk_per_sentence():
    provider = FullProvider("kokoro")
    module = Module()
    runtime = service(provider, module)

    result = runtime.start_speech("One. Two.").wait(2)

    assert result is not None and result.success
    assert provider.calls == ["One.", "Two."]
    assert result.metrics.semantic_chunks == 2
    assert result.metrics.played_chunks == 2


def test_pipeline_applies_markdown_sanitizer_before_semantic_chunking():
    provider = FullProvider("kokoro")
    module = Module()
    runtime = service(provider, module)

    result = runtime.start_speech("1. **Power Stroke**: The gases expand.").wait(2)

    assert result is not None and result.success
    assert provider.calls == ["1. Power Stroke: The gases expand."]
    assert "*" not in provider.calls[0]


def test_first_and_later_stream_failures_are_structured_without_replay():
    class FailingProvider(FullProvider):
        def __init__(self, fail_on):
            super().__init__("kokoro")
            self.fail_on = fail_on

        def synthesize_stream(self, text, *, voice, speed, language, cancellation=None):
            del voice, speed, language, cancellation
            self.calls.append(text)
            if len(self.calls) == self.fail_on:
                raise SynthesisStreamError(
                    SynthesisFailure(
                        SynthesisErrorCode.SYNTHESIS_FAILED,
                        "expected local failure",
                    )
                )
            yield SpeechAudioChunk(
                SynthesizedAudio(bytes((len(self.calls), 0)) * 20, 1_000),
                0,
                True,
            )

    first_module = Module()
    first = service(FailingProvider(1), first_module).start_speech("Fails.").wait(2)
    assert first is not None and not first.success
    assert first.synthesis_error.code is SynthesisErrorCode.SYNTHESIS_FAILED
    assert first_module.streams == []

    later_module = Module()
    later = service(FailingProvider(2), later_module).start_speech("Plays. Fails.").wait(2)
    assert later is not None and not later.success
    assert later.metrics.played_chunks == 1
    writes = [event for event in later_module.events if isinstance(event, bytes)]
    assert writes == [bytes((1, 0)) * 20]
