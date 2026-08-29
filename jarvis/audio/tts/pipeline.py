"""Generation-aware bounded local speech synthesis and playback pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Callable

from .base import (
    SpeechAudioChunk,
    StreamingTTSProvider,
    SynthesizedAudio,
    SynthesisErrorCode,
    SynthesisFailure,
    SynthesisStreamError,
    TTSProvider,
)
from .chunks import SpeechChunk
from .playback import AudioPlaybackService, PlaybackHandle, PlaybackResult


Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class SpeechPipelineSettings:
    queue_size: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.queue_size <= 3:
            raise ValueError("speech audio queue size must be from 1 to 3")


@dataclass(frozen=True, slots=True)
class SpeechPipelineMetrics:
    generation_id: int
    assistant_text_ready: float
    first_chunk_ready: float | None
    first_audio_started: float | None
    generation_finished: float | None
    queued_chunks: int
    played_chunks: int
    semantic_chunks: int
    speech_duration_seconds: float

    @property
    def tts_first_chunk(self) -> float | None:
        if self.first_chunk_ready is None:
            return None
        return max(0.0, self.first_chunk_ready - self.assistant_text_ready)

    @property
    def tts_total_generation(self) -> float | None:
        if self.generation_finished is None:
            return None
        return max(0.0, self.generation_finished - self.assistant_text_ready)


@dataclass(frozen=True, slots=True)
class SpeechPipelineResult:
    success: bool
    provider: str
    voice: str
    generation_id: int
    playback: PlaybackResult | None
    metrics: SpeechPipelineMetrics
    synthesis_error: SynthesisFailure | None = None

    @property
    def error_message(self) -> str | None:
        if self.synthesis_error is not None:
            return self.synthesis_error.message
        if self.playback is not None and self.playback.error is not None:
            return self.playback.error.message
        return None


@dataclass(frozen=True, slots=True)
class _QueuedAudio:
    generation_id: int
    semantic_sequence: int
    provider_sequence: int
    audio: SynthesizedAudio


@dataclass(frozen=True, slots=True)
class _EndOfGeneration:
    generation_id: int


QueueItem = _QueuedAudio | _EndOfGeneration


class SpeechSessionHandle:
    """Cancellable response generation whose late audio is generation-gated."""

    def __init__(
        self,
        *,
        generation_id: int,
        provider_name: str,
        voice: str,
        assistant_text_ready: float,
        semantic_chunks: int,
        audio_queue: Queue[QueueItem],
        cancellation: Event,
        clock: Clock,
    ) -> None:
        self.generation_id = generation_id
        self.provider_name = provider_name
        self.voice = voice
        self._assistant_text_ready = assistant_text_ready
        self._semantic_chunks = semantic_chunks
        self._queue = audio_queue
        self._cancellation = cancellation
        self._clock = clock
        self._lock = Lock()
        self._attached = Event()
        self._playback: PlaybackHandle | None = None
        self._first_chunk_ready: float | None = None
        self._generation_finished: float | None = None
        self._queued_chunks = 0
        self._played_chunks = 0
        self._speech_duration_seconds = 0.0
        self._synthesis_error: SynthesisFailure | None = None

    @property
    def started(self) -> bool:
        return self._playback is not None and self._playback.started

    @property
    def done(self) -> bool:
        return self._playback is not None and self._playback.done

    @property
    def started_at(self) -> float | None:
        return self._playback.started_at if self._playback is not None else None

    @property
    def finished_at(self) -> float | None:
        return self._playback.finished_at if self._playback is not None else None

    @property
    def cancelled(self) -> bool:
        return self._cancellation.is_set()

    @property
    def metrics(self) -> SpeechPipelineMetrics:
        with self._lock:
            return SpeechPipelineMetrics(
                self.generation_id,
                self._assistant_text_ready,
                self._first_chunk_ready,
                self.started_at,
                self._generation_finished,
                self._queued_chunks,
                self._played_chunks,
                self._semantic_chunks,
                self._speech_duration_seconds,
            )

    def _attach(self, playback: PlaybackHandle) -> None:
        self._playback = playback
        self._attached.set()
        if self.cancelled:
            playback.stop()

    def _mark_chunk_ready(self) -> None:
        with self._lock:
            if self._first_chunk_ready is None:
                self._first_chunk_ready = self._clock()

    def _mark_queued(self, audio: SynthesizedAudio) -> None:
        with self._lock:
            self._queued_chunks += 1
            self._speech_duration_seconds += audio.duration_seconds

    def _mark_played(self, _audio: SynthesizedAudio) -> None:
        with self._lock:
            self._played_chunks += 1

    def _finish_generation(self, error: SynthesisFailure | None = None) -> None:
        with self._lock:
            if error is not None:
                self._synthesis_error = error
            if self._generation_finished is None:
                self._generation_finished = self._clock()

    def wait_started(self, timeout_seconds: float | None = None) -> bool:
        if not self._attached.wait(timeout_seconds):
            return False
        assert self._playback is not None
        return self._playback.wait_started(timeout_seconds)

    def wait(self, timeout_seconds: float | None = None) -> SpeechPipelineResult | None:
        if not self._attached.wait(timeout_seconds):
            return None
        assert self._playback is not None
        playback = self._playback.wait(timeout_seconds)
        if playback is None:
            return None
        with self._lock:
            synthesis_error = self._synthesis_error
        return SpeechPipelineResult(
            playback.success and synthesis_error is None,
            self.provider_name,
            self.voice,
            self.generation_id,
            playback,
            self.metrics,
            synthesis_error,
        )

    def stop(self) -> None:
        if not self._cancellation.is_set():
            self._cancellation.set()
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
            try:
                self._queue.put_nowait(_EndOfGeneration(self.generation_id))
            except Full:
                pass
        if self._playback is not None:
            self._playback.stop()

    cancel = stop


class SpeechPipeline:
    """Produce a small lookahead while one continuous playback consumes it."""

    def __init__(
        self,
        playback: AudioPlaybackService,
        *,
        settings: SpeechPipelineSettings = SpeechPipelineSettings(),
        clock: Clock = perf_counter,
    ) -> None:
        self.playback = playback
        self.settings = settings
        self.clock = clock

    @staticmethod
    def _fallback_stream(
        provider: TTSProvider,
        chunk: SpeechChunk,
        *,
        voice: str,
        speed: float,
        language: str,
    ) -> Iterator[SpeechAudioChunk]:
        result = provider.synthesize(
            chunk.text,
            voice=voice,
            speed=speed,
            language=language,
        )
        if not result.success or result.audio is None:
            raise SynthesisStreamError(
                result.error
                or SynthesisFailure(
                    SynthesisErrorCode.SYNTHESIS_FAILED,
                    f"{provider.name} synthesis failed.",
                )
            )
        yield SpeechAudioChunk(result.audio, 0, True)

    def start(
        self,
        provider: TTSProvider,
        chunks: Iterable[SpeechChunk],
        *,
        generation_id: int,
        voice: str,
        speed: float,
        language: str,
        assistant_text_ready: float,
        initial_failure: SynthesisFailure | None = None,
    ) -> SpeechSessionHandle:
        semantic_chunks = tuple(chunks)
        audio_queue: Queue[QueueItem] = Queue(maxsize=self.settings.queue_size)
        cancellation = Event()
        handle = SpeechSessionHandle(
            generation_id=generation_id,
            provider_name=provider.name,
            voice=voice,
            assistant_text_ready=assistant_text_ready,
            semantic_chunks=len(semantic_chunks),
            audio_queue=audio_queue,
            cancellation=cancellation,
            clock=self.clock,
        )

        def put(item: QueueItem) -> bool:
            while not cancellation.is_set():
                try:
                    audio_queue.put(item, timeout=0.05)
                    return True
                except Full:
                    continue
            return False

        def producer() -> None:
            failure = initial_failure
            try:
                if failure is not None:
                    return
                if not semantic_chunks:
                    failure = SynthesisFailure(
                        SynthesisErrorCode.INVALID_TEXT,
                        "Speech text is empty.",
                    )
                    return
                for semantic in semantic_chunks:
                    if cancellation.is_set():
                        break
                    if isinstance(provider, StreamingTTSProvider):
                        stream = provider.synthesize_stream(
                            semantic.text,
                            voice=voice,
                            speed=speed,
                            language=language,
                            cancellation=cancellation,
                        )
                    else:
                        stream = self._fallback_stream(
                            provider,
                            semantic,
                            voice=voice,
                            speed=speed,
                            language=language,
                        )
                    try:
                        for provider_chunk in stream:
                            if cancellation.is_set():
                                break
                            queued = _QueuedAudio(
                                generation_id,
                                semantic.sequence,
                                provider_chunk.sequence,
                                provider_chunk.audio,
                            )
                            handle._mark_chunk_ready()
                            if not put(queued):
                                break
                            handle._mark_queued(provider_chunk.audio)
                    finally:
                        close = getattr(stream, "close", None)
                        if callable(close):
                            close()
                    if cancellation.is_set():
                        break
            except SynthesisStreamError as exc:
                failure = exc.failure
            except Exception as exc:
                failure = SynthesisFailure(
                    SynthesisErrorCode.SYNTHESIS_FAILED,
                    f"Local streaming synthesis failed: {exc}",
                )
            finally:
                handle._finish_generation(failure)
                if not cancellation.is_set():
                    put(_EndOfGeneration(generation_id))

        def queued_audio() -> Iterator[SynthesizedAudio]:
            while not cancellation.is_set():
                try:
                    item = audio_queue.get(timeout=0.05)
                except Empty:
                    continue
                if item.generation_id != generation_id:
                    continue
                if isinstance(item, _EndOfGeneration):
                    return
                yield item.audio

        Thread(
            target=producer,
            name=f"jarvis-tts-producer-{generation_id}",
            daemon=True,
        ).start()
        try:
            playback = self.playback.start_sequence(
                queued_audio(),
                on_chunk_played=handle._mark_played,
            )
        except Exception:
            handle.stop()
            raise
        handle._attach(playback)
        return handle
