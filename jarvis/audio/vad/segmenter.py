"""Deterministic speech-start/end policy over provider-neutral VAD scores."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from jarvis.audio.realtime import RealtimeAudioFrame, RealtimeAudioSource

from .base import VADProvider


class UtteranceEndReason(StrEnum):
    COMPLETE = "complete"
    MAX_DURATION = "max_duration"
    NO_SPEECH = "no_speech"


@dataclass(frozen=True, slots=True)
class UtteranceCapture:
    reason: UtteranceEndReason
    pcm16: bytes
    speech_started_at: float | None
    speech_ended_at: float | None
    end_detected_at: float | None
    handoff_frame_gap_seconds: float | None = None
    handoff_sequence_gap: int | None = None

    @property
    def has_speech(self) -> bool:
        return bool(self.pcm16) and self.speech_started_at is not None

    @property
    def duration_seconds(self) -> float:
        if self.speech_started_at is None or self.speech_ended_at is None:
            return 0.0
        return max(0.0, self.speech_ended_at - self.speech_started_at)


@dataclass(frozen=True, slots=True)
class VADSegmenterSettings:
    threshold: float = 0.5
    trailing_silence_ms: int = 640
    max_utterance_seconds: float = 18.0
    min_speech_ms: int = 240
    listen_timeout_seconds: float = 8.0
    pre_roll_ms: int = 240

    def __post_init__(self) -> None:
        if not 0.05 <= self.threshold <= 0.99:
            raise ValueError("VAD threshold must be from 0.05 to 0.99")
        if not 300 <= self.trailing_silence_ms <= 2_000:
            raise ValueError("trailing silence must be from 300 to 2000 ms")
        if not 3 <= self.max_utterance_seconds <= 60:
            raise ValueError("maximum utterance must be from 3 to 60 seconds")
        if not 60 <= self.min_speech_ms <= 2_000:
            raise ValueError("minimum speech must be from 60 to 2000 ms")
        if not 1 <= self.listen_timeout_seconds <= 60:
            raise ValueError("listen timeout must be from 1 to 60 seconds")
        if not 0 <= self.pre_roll_ms <= 1_000:
            raise ValueError("pre-roll must be from 0 to 1000 ms")


class VADSegmenter:
    def __init__(self, provider: VADProvider, settings: VADSegmenterSettings) -> None:
        self.provider = provider
        self.settings = settings

    def capture(
        self,
        source: RealtimeAudioSource,
        *,
        initial_frames: tuple[RealtimeAudioFrame, ...] = (),
        candidate_frames: tuple[RealtimeAudioFrame, ...] = (),
        speech_already_started: bool = False,
        speech_start_timeout_seconds: float | None = None,
    ) -> UtteranceCapture:
        """Capture one utterance from a continuous source.

        ``initial_frames`` are passive bounded pre-roll. ``candidate_frames``
        are the tail of an already-consumed stream and pass through the normal
        VAD start gate before new source frames. A candidate tail can bridge a
        wake transition, but is deliberately kept shorter than the configured
        speech minimum by the coordinator so old playback cannot confirm a
        command by itself.
        """

        self.provider.reset()
        frame_ms = source.frame_duration_ms
        pre_roll_frames = max(1, math.ceil(self.settings.pre_roll_ms / frame_ms))
        initial_frames = tuple(initial_frames)
        candidate_frames = tuple(candidate_frames)
        if speech_already_started and candidate_frames:
            raise ValueError(
                "candidate_frames cannot be combined with speech_already_started"
            )
        start_timeout = (
            self.settings.listen_timeout_seconds
            if speech_start_timeout_seconds is None
            else speech_start_timeout_seconds
        )
        if not 0.1 <= start_timeout <= 60:
            raise ValueError("speech-start timeout must be from 0.1 to 60 seconds")
        pre_roll_seed = () if speech_already_started else initial_frames[-pre_roll_frames:]
        pre_roll: deque[RealtimeAudioFrame] = deque(
            pre_roll_seed,
            maxlen=pre_roll_frames,
        )
        pending: list[RealtimeAudioFrame] = []
        output: list[RealtimeAudioFrame] = (
            list(initial_frames) if speech_already_started else []
        )
        speech_started = speech_already_started and bool(initial_frames)
        speech_start = (
            initial_frames[0].captured_at - initial_frames[0].duration_seconds
            if speech_started
            else None
        )
        speech_end = initial_frames[-1].captured_at if speech_started else None
        last_voiced_count = len(output)
        voiced_ms = (
            sum(frame.duration_seconds * 1000 for frame in initial_frames)
            if speech_started
            else 0.0
        )
        silence_ms = 0.0
        listening_ms = 0.0
        speech_elapsed = (
            sum(frame.duration_seconds for frame in initial_frames)
            if speech_started
            else 0.0
        )
        candidates: deque[RealtimeAudioFrame] = deque(candidate_frames)
        handoff_tail = (
            candidate_frames[-1]
            if candidate_frames
            else initial_frames[-1]
            if initial_frames
            else None
        )
        first_live_frame_seen = False
        handoff_frame_gap_seconds: float | None = None
        handoff_sequence_gap: int | None = None

        while True:
            if candidates:
                frame = candidates.popleft()
                live_frame = False
            else:
                frame = source.read(timeout_seconds=1.0)
                live_frame = True
                listening_ms += frame.duration_seconds * 1000
                if not first_live_frame_seen:
                    first_live_frame_seen = True
                    if handoff_tail is not None:
                        frame_start = frame.captured_at - frame.duration_seconds
                        handoff_frame_gap_seconds = max(
                            0.0, frame_start - handoff_tail.captured_at
                        )
                        if (
                            handoff_tail.sequence is not None
                            and frame.sequence is not None
                        ):
                            handoff_sequence_gap = max(
                                0, frame.sequence - handoff_tail.sequence - 1
                            )
            score = self.provider.score(frame.pcm16)
            is_speech = score >= self.settings.threshold

            if not speech_started:
                if is_speech:
                    pending.append(frame)
                    voiced_ms += frame.duration_seconds * 1000
                    if voiced_ms >= self.settings.min_speech_ms:
                        speech_started = True
                        speech_start = pending[0].captured_at - pending[0].duration_seconds
                        output.extend(pre_roll)
                        output.extend(pending)
                        speech_end = pending[-1].captured_at
                        last_voiced_count = len(output)
                        speech_elapsed = sum(item.duration_seconds for item in pending)
                        pending.clear()
                else:
                    for item in pending:
                        pre_roll.append(item)
                    pending.clear()
                    voiced_ms = 0.0
                    pre_roll.append(frame)
                if live_frame and listening_ms >= start_timeout * 1000:
                    return UtteranceCapture(
                        UtteranceEndReason.NO_SPEECH,
                        b"",
                        None,
                        None,
                        frame.captured_at,
                        handoff_frame_gap_seconds,
                        handoff_sequence_gap,
                    )
                continue

            output.append(frame)
            speech_elapsed += frame.duration_seconds
            if is_speech:
                speech_end = frame.captured_at
                last_voiced_count = len(output)
                silence_ms = 0.0
            else:
                silence_ms += frame.duration_seconds * 1000

            if speech_elapsed >= self.settings.max_utterance_seconds:
                final = output[:last_voiced_count] or output
                ended = speech_end or frame.captured_at
                return UtteranceCapture(
                    UtteranceEndReason.MAX_DURATION,
                    b"".join(item.pcm16 for item in final),
                    speech_start,
                    ended,
                    frame.captured_at,
                    handoff_frame_gap_seconds,
                    handoff_sequence_gap,
                )
            if silence_ms >= self.settings.trailing_silence_ms:
                final = output[:last_voiced_count]
                return UtteranceCapture(
                    UtteranceEndReason.COMPLETE,
                    b"".join(item.pcm16 for item in final),
                    speech_start,
                    speech_end,
                    frame.captured_at,
                    handoff_frame_gap_seconds,
                    handoff_sequence_gap,
                )


class SpeechStartGate:
    """Detect sustained speech during playback after an explicit echo guard."""

    def __init__(
        self,
        provider: VADProvider,
        *,
        threshold: float,
        min_speech_ms: int,
        suppression_ms: int,
    ) -> None:
        if not 0.05 <= threshold <= 0.99:
            raise ValueError("barge-in threshold must be from 0.05 to 0.99")
        if not 60 <= min_speech_ms <= 1_000:
            raise ValueError("barge-in minimum speech must be from 60 to 1000 ms")
        if not 0 <= suppression_ms <= 3_000:
            raise ValueError("barge-in suppression must be from 0 to 3000 ms")
        self.provider = provider
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.suppression_ms = suppression_ms
        self._candidate: list[RealtimeAudioFrame] = []
        self._voiced_ms = 0.0
        self._armed = suppression_ms == 0
        self.provider.reset()

    def process(self, frame: RealtimeAudioFrame, playback_elapsed_ms: float) -> bool:
        if playback_elapsed_ms < self.suppression_ms:
            self._candidate.clear()
            self._voiced_ms = 0.0
            return False
        if not self._armed:
            self.provider.reset()
            self._armed = True
        if self.provider.score(frame.pcm16) >= self.threshold:
            self._candidate.append(frame)
            self._voiced_ms += frame.duration_seconds * 1000
            return self._voiced_ms >= self.min_speech_ms
        self._candidate.clear()
        self._voiced_ms = 0.0
        return False

    def take_candidate_frames(self) -> tuple[RealtimeAudioFrame, ...]:
        frames = tuple(self._candidate)
        self._candidate.clear()
        self._voiced_ms = 0.0
        return frames
