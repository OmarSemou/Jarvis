"""Responsive local wake/VAD/STT/conversation/TTS coordination."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Any, Protocol

from jarvis.audio.realtime import (
    RealtimeAudioFrame,
    RealtimeAudioSource,
    RealtimeAudioTimeoutError,
    RollingAudioFrameBuffer,
)
from jarvis.audio.service import VoiceInputError, VoiceInputService
from jarvis.audio.tts.playback import PlaybackErrorCode
from jarvis.audio.tts.service import TTSService
from jarvis.audio.vad.base import VADProvider
from jarvis.audio.vad.segmenter import (
    SpeechStartGate,
    UtteranceCapture,
    UtteranceEndReason,
    VADSegmenter,
)
from jarvis.audio.wake.base import WakeWordProvider
from jarvis.core.conversation import ConversationService
from jarvis.llm.base import LLMError

from .commands import (
    LocalVoiceCommandExecutor,
    LocalVoiceCommandRouter,
    is_no_speech_transcript,
    is_wake_only_transcript,
)
from .latency import LatencyHistory, VoiceLatencyTracker, format_latency
from .state import VoiceInteractionState, VoiceStateMachine


class VoiceCoordinatorError(RuntimeError):
    """A clean, user-facing continuous voice-mode failure."""


OutputFunction = Callable[[str], None]
Clock = Callable[[], float]
WAKE_SCORE_REPORT_SECONDS = 1.0
PLAYBACK_CANCEL_TARGET_SECONDS = 0.5


class CancellableSpeechHandle(Protocol):
    @property
    def done(self) -> bool: ...

    @property
    def started_at(self) -> float | None: ...

    @property
    def finished_at(self) -> float | None: ...

    def wait_started(self, timeout_seconds: float | None = None) -> bool: ...

    def wait(self, timeout_seconds: float | None = None) -> Any | None: ...

    def stop(self) -> None: ...


class BargeInMode(StrEnum):
    """Explicit playback interruption policies."""

    WAKEWORD = "wakeword"
    VAD_EXPERIMENTAL = "vad_experimental"


@dataclass(frozen=True, slots=True)
class VoiceInterruption:
    capture: UtteranceCapture
    wake_at: float | None


@dataclass(frozen=True, slots=True)
class WakeAudioHandoff:
    initial_frames: tuple[RealtimeAudioFrame, ...]
    candidate_frames: tuple[RealtimeAudioFrame, ...]
    duration_ms: int


@dataclass(frozen=True, slots=True)
class VoiceModeSettings:
    preload_tts: bool = True
    barge_in_enabled: bool = True
    barge_in_mode: BargeInMode = BargeInMode.WAKEWORD
    barge_in_threshold: float = 0.75
    barge_in_suppression_ms: int = 480
    barge_in_min_speech_ms: int = 240
    barge_in_pre_roll_ms: int = 320
    barge_in_command_start_timeout_seconds: float = 1.5
    debug_latency: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.barge_in_mode, BargeInMode):
            try:
                object.__setattr__(self, "barge_in_mode", BargeInMode(self.barge_in_mode))
            except (TypeError, ValueError):
                raise ValueError(
                    "barge-in mode must be one of: wakeword, vad_experimental"
                ) from None
        if not 0.05 <= self.barge_in_threshold <= 0.99:
            raise ValueError("barge-in threshold must be from 0.05 to 0.99")
        if not 0 <= self.barge_in_suppression_ms <= 3_000:
            raise ValueError("barge-in suppression must be from 0 to 3000 ms")
        if not 60 <= self.barge_in_min_speech_ms <= 1_000:
            raise ValueError("barge-in minimum speech must be from 60 to 1000 ms")
        if not 200 <= self.barge_in_pre_roll_ms <= 500:
            raise ValueError("barge-in pre-roll must be from 200 to 500 ms")
        if not 0.5 <= self.barge_in_command_start_timeout_seconds <= 3.0:
            raise ValueError(
                "barge-in command-start timeout must be from 0.5 to 3 seconds"
            )


class VoiceModeCoordinator:
    """Stateful coordinator with no robot authority and no provider-specific types."""

    def __init__(
        self,
        source: RealtimeAudioSource,
        wakeword: WakeWordProvider,
        vad: VADProvider,
        segmenter: VADSegmenter,
        voice_input: VoiceInputService,
        conversation: ConversationService,
        tts: TTSService,
        *,
        local_command_executor: LocalVoiceCommandExecutor | None = None,
        settings: VoiceModeSettings = VoiceModeSettings(),
        state: VoiceStateMachine | None = None,
        latency: LatencyHistory | None = None,
        output_fn: OutputFunction = print,
        clock: Clock = perf_counter,
        speech_activity_sink: Callable[[str, int], None] | None = None,
    ) -> None:
        self.source = source
        self.wakeword = wakeword
        self.vad = vad
        self.segmenter = segmenter
        self.voice_input = voice_input
        self.conversation = conversation
        self.tts = tts
        self.local_command_executor = local_command_executor
        self.local_command_router = LocalVoiceCommandRouter()
        self.settings = settings
        self.state = state or VoiceStateMachine()
        self.latency = latency or LatencyHistory()
        self.output_fn = output_fn
        self.clock = clock
        self.speech_activity_sink = speech_activity_sink
        self._wake_buffer = bytearray()
        self._pending_wake_handoff: WakeAudioHandoff | None = None
        self._shutdown_requested = threading.Event()

    def request_shutdown(self) -> None:
        """Request a clean stop from a UI thread without changing safety state."""

        self._shutdown_requested.set()
        try:
            self.tts.stop()
        finally:
            self.source.stop()

    def _emit_speech_activity(self, kind: str, handle: object) -> None:
        generation_id = getattr(handle, "generation_id", None)
        if self.speech_activity_sink is None or not isinstance(generation_id, int):
            return
        try:
            self.speech_activity_sink(kind, generation_id)
        except Exception:
            # Observation/UI hooks are never part of voice authority.
            pass

    def _reset_detection_state(self) -> None:
        """Clear provider state and partial wake chunks, never microphone audio."""

        self._wake_buffer.clear()
        self.wakeword.reset()
        self.vad.reset()

    def _prepare_handoff(
        self, buffer: RollingAudioFrameBuffer
    ) -> WakeAudioHandoff:
        frames = buffer.snapshot()
        candidate_count = max(
            0,
            math.ceil(
                (
                    self.segmenter.settings.min_speech_ms
                    - self.source.frame_duration_ms
                )
                / self.source.frame_duration_ms
            ),
        )
        candidate_count = min(candidate_count, len(frames))
        if candidate_count:
            initial_frames = frames[:-candidate_count]
            candidate_frames = frames[-candidate_count:]
        else:
            initial_frames = frames
            candidate_frames = ()
        handoff = WakeAudioHandoff(
            initial_frames,
            candidate_frames,
            buffer.duration_ms,
        )
        buffer.clear()
        return handoff

    def _startup(self) -> None:
        for failure in (
            self.wakeword.readiness_error(),
            self.vad.readiness_error(),
            self.voice_input.stt.readiness_error(),
        ):
            if failure is not None:
                raise VoiceCoordinatorError(failure.message)
        if not self.tts.enabled:
            raise VoiceCoordinatorError("Voice output must be enabled for continuous voice mode.")
        tts_status = self.tts.status()
        if not tts_status.ready:
            raise VoiceCoordinatorError(tts_status.detail)
        for failure in (self.wakeword.warmup(), self.vad.warmup()):
            if failure is not None:
                raise VoiceCoordinatorError(failure.message)
        if self.settings.preload_tts:
            failure = self.tts.warmup()
            if failure is not None:
                raise VoiceCoordinatorError(f"TTS warmup failed: {failure.message}")
        self.source.start()

    def _wait_for_wake(self) -> float:
        self._pending_wake_handoff = None
        self._reset_detection_state()
        chunk_bytes = 1_280 * 2
        handoff_buffer = RollingAudioFrameBuffer(
            max_duration_ms=max(
                self.source.frame_duration_ms,
                self.segmenter.settings.pre_roll_ms,
            ),
            frame_duration_ms=self.source.frame_duration_ms,
        )
        score_peak = 0.0
        score_window_started = self.clock()
        while (
            self.state.current is VoiceInteractionState.IDLE
            and not self._shutdown_requested.is_set()
        ):
            try:
                frame = self.source.read(timeout_seconds=1.0)
            except RealtimeAudioTimeoutError:
                continue
            handoff_buffer.append(frame)
            self._wake_buffer.extend(frame.pcm16)
            while len(self._wake_buffer) >= chunk_bytes:
                chunk = bytes(self._wake_buffer[:chunk_bytes])
                del self._wake_buffer[:chunk_bytes]
                detection = self.wakeword.process(chunk)
                if detection.error is not None:
                    raise VoiceCoordinatorError(detection.error.message)
                score_peak = max(score_peak, detection.score)
                now = self.clock()
                if (
                    self.settings.debug_latency
                    and not detection.detected
                    and now - score_window_started >= WAKE_SCORE_REPORT_SECONDS
                ):
                    self.output_fn(
                        f"[WAKE] peak={score_peak:.3f} "
                        f"threshold={self.wakeword.threshold:.3f}"
                    )
                    score_peak = 0.0
                    score_window_started = now
                if detection.detected:
                    detected_at = now
                    self._pending_wake_handoff = self._prepare_handoff(
                        handoff_buffer
                    )
                    self._wake_buffer.clear()
                    self.state.transition(VoiceInteractionState.WAKE_DETECTED)
                    self.output_fn(f"[VOICE] Wake detected (score={detection.score:.3f}).")
                    return detected_at
        if self._shutdown_requested.is_set():
            raise VoiceCoordinatorError("Voice mode was closed by the face window.")
        raise VoiceCoordinatorError("Voice mode left idle unexpectedly.")

    def _capture(self) -> UtteranceCapture:
        self.state.transition(VoiceInteractionState.LISTENING)
        self.output_fn("[VOICE] Listening...")
        handoff = self._pending_wake_handoff
        self._pending_wake_handoff = None
        capture = self.segmenter.capture(
            self.source,
            initial_frames=handoff.initial_frames if handoff is not None else (),
            candidate_frames=handoff.candidate_frames if handoff is not None else (),
        )
        if self.settings.debug_latency and handoff is not None:
            gap = capture.handoff_frame_gap_seconds
            gap_text = "unknown" if gap is None else f"{gap * 1_000:.0f}ms"
            sequence_text = (
                ""
                if capture.handoff_sequence_gap is None
                else f" sequence_gap={capture.handoff_sequence_gap}"
            )
            self.output_fn(
                f"[VOICE] wake_audio preroll={handoff.duration_ms}ms "
                f"first_frame_gap={gap_text}{sequence_text}"
            )
        return capture

    def _record_capture_timing(
        self,
        tracker: VoiceLatencyTracker,
        capture: UtteranceCapture,
        wake_at: float | None,
    ) -> None:
        if wake_at is not None:
            tracker.mark("wake", wake_at)
        if capture.speech_started_at is not None:
            tracker.mark(
                "speech_start",
                max(capture.speech_started_at, wake_at)
                if wake_at is not None
                else capture.speech_started_at,
            )
        if capture.speech_ended_at is not None:
            tracker.mark("speech_end", capture.speech_ended_at)
        if capture.end_detected_at is not None:
            tracker.mark("end_detected", capture.end_detected_at)

    def _wakeword_barge_capture(
        self,
        handle: CancellableSpeechHandle,
        tracker: VoiceLatencyTracker,
    ) -> VoiceInterruption | None:
        """Require the local wake phrase before playback may be cancelled."""

        self._reset_detection_state()
        wake_buffer = bytearray()
        chunk_bytes = 1_280 * 2
        handoff_buffer = RollingAudioFrameBuffer(
            max_duration_ms=self.settings.barge_in_pre_roll_ms,
            frame_duration_ms=self.source.frame_duration_ms,
        )
        while not handle.done:
            try:
                frame = self.source.read(timeout_seconds=0.1)
            except RealtimeAudioTimeoutError:
                if handle.done:
                    break
                continue
            handoff_buffer.append(frame)
            wake_buffer.extend(frame.pcm16)
            while len(wake_buffer) >= chunk_bytes:
                chunk = bytes(wake_buffer[:chunk_bytes])
                del wake_buffer[:chunk_bytes]
                detection = self.wakeword.process(chunk)
                if detection.error is not None:
                    raise VoiceCoordinatorError(detection.error.message)
                if not detection.detected:
                    continue

                wake_at = self.clock()
                tracker.mark("barge_wake", wake_at)
                # Let already-consumed audio bridge the transition without
                # permitting playback-only frames to satisfy the complete VAD
                # start gate. At least one new live frame is always required.
                handoff = self._prepare_handoff(handoff_buffer)
                if self.settings.debug_latency:
                    self.output_fn(
                        "[VOICE] wake_barge_detected source=speaking "
                        f"score={detection.score:.3f} "
                        f"preroll={handoff.duration_ms}ms"
                    )

                # Playback cancellation and VAD handoff share this same
                # already-running microphone stream. No drain, restart, or
                # cancellation wait is allowed between them.
                handle.stop()
                self._emit_speech_activity("cancelled", handle)
                self.state.transition(VoiceInteractionState.INTERRUPTED)
                self.output_fn("[VOICE] Wake detected; listening...")
                self._reset_detection_state()
                self.state.transition(VoiceInteractionState.LISTENING)
                capture = self.segmenter.capture(
                    self.source,
                    initial_frames=handoff.initial_frames,
                    candidate_frames=handoff.candidate_frames,
                    speech_already_started=False,
                    speech_start_timeout_seconds=(
                        self.settings.barge_in_command_start_timeout_seconds
                    ),
                )

                result = handle.wait(PLAYBACK_CANCEL_TARGET_SECONDS)
                if result is None:
                    handle.stop()
                    result = handle.wait(PLAYBACK_CANCEL_TARGET_SECONDS)
                if result is None:
                    raise VoiceCoordinatorError(
                        "Local speech playback did not cancel cleanly after wake detection."
                    )
                cancelled_at = getattr(handle, "finished_at", None) or self.clock()
                tracker.mark("playback_cancelled", cancelled_at)

                if self.settings.debug_latency:
                    cancel_latency = cancelled_at - wake_at
                    self.output_fn(
                        "[VOICE] wake_barge_in source=speaking "
                        f"score={detection.score:.3f} "
                        f"cancel={cancel_latency:.3f}s "
                        f"preroll={handoff.duration_ms}ms"
                    )
                    gap = capture.handoff_frame_gap_seconds
                    gap_text = "unknown" if gap is None else f"{gap * 1_000:.0f}ms"
                    sequence_text = (
                        ""
                        if capture.handoff_sequence_gap is None
                        else f" sequence_gap={capture.handoff_sequence_gap}"
                    )
                    self.output_fn(
                        f"[VOICE] barge_audio first_frame_gap={gap_text}"
                        f"{sequence_text}"
                    )
                    if capture.speech_started_at is not None:
                        self.output_fn(
                            "[VOICE] command_speech_start="
                            f"{max(0.0, capture.speech_started_at - wake_at):.3f}s"
                        )
                return VoiceInterruption(capture, wake_at)
        self._reset_detection_state()
        return None

    def _experimental_vad_barge_capture(
        self,
        handle: CancellableSpeechHandle,
    ) -> VoiceInterruption | None:
        """Retain the former VAD-only behavior behind an explicit opt-in."""

        gate = SpeechStartGate(
            self.vad,
            threshold=self.settings.barge_in_threshold,
            min_speech_ms=self.settings.barge_in_min_speech_ms,
            suppression_ms=self.settings.barge_in_suppression_ms,
        )
        playback_started = handle.started_at or self.clock()
        while not handle.done:
            try:
                frame = self.source.read(timeout_seconds=0.1)
            except RealtimeAudioTimeoutError:
                if handle.done:
                    break
                continue
            elapsed_ms = max(0.0, (self.clock() - playback_started) * 1_000)
            if not gate.process(frame, elapsed_ms):
                continue
            handle.stop()
            self._emit_speech_activity("cancelled", handle)
            handle.wait(1.0)
            self.state.transition(VoiceInteractionState.INTERRUPTED)
            if self.settings.debug_latency:
                self.output_fn("[VOICE] vad_experimental_barge_in")
            else:
                self.output_fn("[VOICE] Experimental speech interruption; listening...")
            self.state.transition(VoiceInteractionState.LISTENING)
            return VoiceInterruption(
                self.segmenter.capture(
                    self.source,
                    initial_frames=gate.take_candidate_frames(),
                    speech_already_started=True,
                ),
                None,
            )
        return None

    def _barge_capture(
        self,
        handle: CancellableSpeechHandle,
        tracker: VoiceLatencyTracker,
    ) -> VoiceInterruption | None:
        if not self.settings.barge_in_enabled:
            handle.wait()
            return None
        if self.settings.barge_in_mode is BargeInMode.WAKEWORD:
            return self._wakeword_barge_capture(handle, tracker)
        return self._experimental_vad_barge_capture(handle)

    def _speak(
        self,
        text: str,
        tracker: VoiceLatencyTracker,
    ) -> VoiceInterruption | None:
        speech_end = tracker.timestamp("speech_end") or 0.0
        assistant_ready = tracker.mark(
            "assistant_text_ready", max(self.clock(), speech_end)
        )
        tracker.mark("tts_start", assistant_ready)
        self.source.drain()
        self._pending_wake_handoff = None
        self._reset_detection_state()
        start_speech = getattr(self.tts, "start_speech", None)
        if callable(start_speech):
            handle = start_speech(text, assistant_text_ready=assistant_ready)
        else:
            # Backward-compatible seam for small injected test doubles. The
            # production TTSService always takes the bounded pipeline above.
            synthesis = self.tts.synthesize(text)
            tracker.mark("tts_end", max(self.clock(), assistant_ready))
            if not synthesis.success or synthesis.audio is None:
                detail = (
                    synthesis.error.message
                    if synthesis.error is not None
                    else "unknown failure"
                )
                self.output_fn(
                    f"Voice output error: {detail}. Text response remains available."
                )
                self.state.transition(VoiceInteractionState.IDLE)
                return None
            tracker.mark("playback_requested", max(self.clock(), assistant_ready))
            handle = self.tts.start_playback(synthesis.audio)

        def has_started() -> bool:
            return bool(
                getattr(
                    handle,
                    "started",
                    getattr(handle, "started_at", None) is not None,
                )
            )

        while not has_started() and not handle.done:
            handle.wait_started(0.1)
        pipeline_metrics = getattr(handle, "metrics", None)
        if pipeline_metrics is not None:
            if pipeline_metrics.first_chunk_ready is not None:
                first_chunk_ready = max(
                    pipeline_metrics.first_chunk_ready, assistant_ready
                )
                tracker.mark(
                    "first_chunk_ready",
                    first_chunk_ready,
                )
                tracker.mark("playback_requested", first_chunk_ready)
            if pipeline_metrics.generation_finished is not None:
                generation_finished = max(
                    pipeline_metrics.generation_finished, assistant_ready
                )
                tracker.mark("tts_end", generation_finished)
                tracker.mark("tts_generation_finished", generation_finished)
            tracker.set_count("queued_chunks", pipeline_metrics.queued_chunks)
            tracker.set_count("played_chunks", pipeline_metrics.played_chunks)

        if not has_started():
            failed = handle.wait(1.0)
            detail = getattr(failed, "error_message", None) or "unknown local speech failure"
            self.output_fn(f"Voice output error: {detail}. Text response remains available.")
            self.state.transition(VoiceInteractionState.IDLE)
            return None

        if handle.started_at is not None:
            self.state.transition(VoiceInteractionState.SPEAKING)
            self._emit_speech_activity("started", handle)
            audio_started = max(handle.started_at, assistant_ready)
            tracker.mark("playback_started", audio_started)
            tracker.mark("first_audio_started", audio_started)

        interruption = self._barge_capture(handle, tracker)
        if interruption is not None:
            return interruption

        result = handle.wait(1.0)
        if result is None:
            handle.stop()
            result = handle.wait(1.0)
        if result is None:
            raise VoiceCoordinatorError("Local speech playback did not stop cleanly.")
        pipeline_metrics = getattr(handle, "metrics", None)
        if pipeline_metrics is not None:
            if pipeline_metrics.generation_finished is not None:
                generation_finished = max(
                    pipeline_metrics.generation_finished, assistant_ready
                )
                tracker.mark("tts_end", generation_finished)
                tracker.mark("tts_generation_finished", generation_finished)
            tracker.set_count("queued_chunks", pipeline_metrics.queued_chunks)
            tracker.set_count("played_chunks", pipeline_metrics.played_chunks)
        playback_result = getattr(result, "playback", result)
        playback_error = getattr(playback_result, "error", None)
        if not result.success and (
            playback_error is None or playback_error.code is not PlaybackErrorCode.INTERRUPTED
        ):
            detail = getattr(result, "error_message", None)
            if detail is None:
                detail = playback_error.message if playback_error is not None else "unknown failure"
            self.output_fn(f"Voice output error: {detail}. Text response remains available.")
        # Audio accumulated during known playback is self-speech until proven
        # otherwise. Barge-in returns earlier with its separately gated frames.
        self.source.drain()
        self._reset_detection_state()
        self._emit_speech_activity("stopped", handle)
        self.state.transition(VoiceInteractionState.IDLE)
        return None

    def _discard_no_speech(self) -> None:
        self._pending_wake_handoff = None
        self._reset_detection_state()
        if self.settings.debug_latency:
            self.output_fn("[VOICE] no_speech_discarded")
        self.state.transition(VoiceInteractionState.IDLE)

    def _process_capture(
        self,
        capture: UtteranceCapture,
        *,
        wake_at: float | None,
    ) -> VoiceInterruption | None:
        tracker = VoiceLatencyTracker(clock=self.clock)
        self._record_capture_timing(tracker, capture, wake_at)
        minimum_speech_seconds = self.segmenter.settings.min_speech_ms / 1_000
        if (
            capture.reason is UtteranceEndReason.NO_SPEECH
            or not capture.has_speech
            or capture.duration_seconds + 1e-9 < minimum_speech_seconds
        ):
            self._discard_no_speech()
            return None

        self.state.transition(VoiceInteractionState.PROCESSING)
        tracker.mark("stt_start")
        outcome = self.voice_input.transcribe_pcm16(capture.pcm16)
        stt_ended_at = tracker.mark("stt_end")
        if outcome.cleanup_warning:
            self.output_fn(f"Privacy warning: {outcome.cleanup_warning}")
        transcription = outcome.transcription
        if not transcription.success:
            detail = (
                transcription.error.message
                if transcription.error is not None
                else "No speech could be transcribed."
            )
            self.output_fn(f"Voice input error: {detail}")
            self.state.transition(VoiceInteractionState.IDLE)
            return None
        if is_no_speech_transcript(transcription.text) or (
            wake_at is not None and is_wake_only_transcript(transcription.text)
        ):
            self._discard_no_speech()
            return None
        self.output_fn(f"You (voice) > {transcription.text}")

        local_command = self.local_command_router.match(transcription.text)
        if local_command is not None:
            if self.local_command_executor is None:
                raise VoiceCoordinatorError(
                    "Deterministic local stop execution is unavailable."
                )
            execution = self.local_command_executor.execute(local_command)
            stopped_at = tracker.mark("local_stop_executed")
            if self.settings.debug_latency:
                self.output_fn(
                    f"[VOICE] local_stop latency={stopped_at - stt_ended_at:.3f}s"
                )
            acknowledgement = "Stopped." if execution.success else "Stop failed safely."
            self.output_fn(f"Jarvis > {acknowledgement}")
            interrupted = self._speak(acknowledgement, tracker)
            metrics = tracker.metrics()
            self.latency.add(metrics)
            if self.settings.debug_latency:
                self.output_fn(format_latency(metrics))
            return interrupted

        tracker.mark("llm_start")
        response = self.conversation.respond(transcription.text)
        tracker.mark("llm_end")
        self.output_fn(f"Jarvis > {response.text}")
        interrupted = self._speak(response.text, tracker)

        metrics = tracker.metrics()
        self.latency.add(metrics)
        if self.settings.debug_latency:
            self.output_fn(format_latency(metrics))
        return interrupted

    def run(self, *, max_interactions: int | None = None) -> int:
        """Run until Ctrl+C/shutdown; ``max_interactions`` is a test seam."""

        if max_interactions is not None and max_interactions < 1:
            raise ValueError("max_interactions must be positive")
        completed = 0
        try:
            self._startup()
            self.output_fn("Listening for wake word...")
            while max_interactions is None or completed < max_interactions:
                if self._shutdown_requested.is_set():
                    return 0
                wake_at = self._wait_for_wake()
                capture = self._capture()
                while True:
                    pending = self._process_capture(capture, wake_at=wake_at)
                    completed += 1
                    if pending is None or (
                        max_interactions is not None and completed >= max_interactions
                    ):
                        break
                    capture = pending.capture
                    wake_at = pending.wake_at
            return 0
        except KeyboardInterrupt:
            self.output_fn("\n[VOICE] Interrupted.")
            return 130
        except (VoiceCoordinatorError, VoiceInputError, LLMError, ValueError) as exc:
            if self._shutdown_requested.is_set():
                return 0
            self.output_fn(f"Voice mode error: {exc}")
            self.state.fail_to_idle()
            return 1
        except Exception as exc:
            self.output_fn(f"Voice mode error: failed safely: {exc}")
            self.state.fail_to_idle()
            return 1
        finally:
            self.tts.stop()
            self.source.stop()
            self.state.shutdown()
