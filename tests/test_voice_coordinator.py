from collections import deque
from time import perf_counter
from types import SimpleNamespace

import pytest

from jarvis.audio.realtime import RealtimeAudioFrame, RealtimeAudioTimeoutError
from jarvis.audio.service import VoiceInputOutcome
from jarvis.audio.stt.base import TranscriptionResult
from jarvis.audio.tts.base import SpeechSynthesisResult, SynthesizedAudio, SynthesisFailure, SynthesisErrorCode
from jarvis.audio.tts.playback import PlaybackResult
from jarvis.audio.tts.service import TTSStatus
from jarvis.audio.vad.segmenter import UtteranceCapture, UtteranceEndReason
from jarvis.audio.vad.segmenter import VADSegmenter, VADSegmenterSettings
from jarvis.audio.voice.coordinator import (
    BargeInMode,
    VoiceModeCoordinator,
    VoiceModeSettings,
)
from jarvis.audio.voice.latency import VoiceLatencyTracker
from jarvis.audio.voice.state import VoiceInteractionState as State
from jarvis.audio.voice.state import VoiceStateMachine
from jarvis.audio.wake.base import WakeWordDetection
from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.llm.base import LLMResponse
from jarvis.robot.controller import create_simulated_controller
from jarvis.robot.intents import RobotAction, RobotIntent
from jarvis.robot.simulator import MotionState
from jarvis.integrations.voice_stop import SafeLocalVoiceCommandExecutor
from jarvis.tools.policy import RobotToolPolicy
from jarvis.tools.registry import RobotToolRegistry
from jarvis.tools.types import ToolCall


class Source:
    frame_duration_ms = 80

    def __init__(self, frames):
        self.frames = deque(frames)
        self.started = 0
        self.stopped = 0
        self.drained = 0

    def start(self):
        self.started += 1

    def read(self, timeout_seconds=1.0):
        return self.frames.popleft()

    def drain(self):
        self.drained += 1
        return 0

    def stop(self):
        self.stopped += 1


class Wake:
    name = "mock-wake"
    phrase = "Hey Jarvis"
    threshold = 0.5

    def __init__(self, scores=()):
        self.calls = 0
        self.warmups = 0
        self.scores = deque(scores)
        self.resets = 0

    def readiness_error(self):
        return None

    def warmup(self):
        self.warmups += 1
        return None

    def process(self, _pcm16):
        self.calls += 1
        score = self.scores.popleft() if self.scores else 0.9
        return WakeWordDetection(
            score >= self.threshold, score, self.name, self.phrase
        )

    def reset(self):
        self.resets += 1


class VAD:
    name = "mock-vad"

    def __init__(self, scores=()):
        self.scores = deque(scores)
        self.warmups = 0
        self.resets = 0

    def readiness_error(self):
        return None

    def warmup(self):
        self.warmups += 1
        return None

    def score(self, _pcm16):
        return self.scores.popleft()

    def reset(self):
        self.resets += 1


class Segmenter:
    def __init__(self, captures):
        self.captures = deque(captures)
        self.calls = []
        self.settings = SimpleNamespace(min_speech_ms=240, pre_roll_ms=240)

    def capture(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return self.captures.popleft()


class STT:
    name = "mock-stt"

    def readiness_error(self):
        return None


class VoiceInput:
    def __init__(self, texts):
        self.stt = STT()
        self.texts = deque(texts)
        self.audio = []

    def transcribe_pcm16(self, pcm16):
        self.audio.append(pcm16)
        text = self.texts.popleft()
        return VoiceInputOutcome(TranscriptionResult(True, text, "mock-stt", 0.1))


class Conversation:
    def __init__(self, replies):
        self.replies = deque(replies)
        self.requests = []

    def respond(self, text):
        self.requests.append(text)
        return LLMResponse(self.replies.popleft(), "mock")


class Handle:
    def __init__(self, *, done=True, started_at=None):
        self._done = done
        self.started_at = started_at if started_at is not None else perf_counter() + 1
        self.stops = 0

    @property
    def done(self):
        return self._done

    def wait_started(self, _timeout=None):
        return True

    def wait(self, _timeout=None):
        return PlaybackResult(True, "mock") if self._done else None

    def stop(self):
        self.stops += 1
        self._done = True


class TTS:
    enabled = True
    provider = "kokoro"
    voice = "am_fenrir"

    def __init__(self, handles, *, warmup_failure=None, synthesis_failure=None):
        self.handles = deque(handles)
        self.warmups = 0
        self.synthesized = []
        self.stops = 0
        self.warmup_failure = warmup_failure
        self.synthesis_failure = synthesis_failure

    def status(self):
        return TTSStatus(True, self.provider, self.voice, 1.0, "en", True, "ready")

    def warmup(self):
        self.warmups += 1
        return self.warmup_failure

    def synthesize(self, text):
        self.synthesized.append(text)
        if self.synthesis_failure:
            return SpeechSynthesisResult(
                False,
                self.provider,
                self.voice,
                0.1,
                error=self.synthesis_failure,
            )
        return SpeechSynthesisResult(
            True,
            self.provider,
            self.voice,
            0.1,
            SynthesizedAudio(b"\x00\x00" * 100, 10_000),
        )

    def start_playback(self, _audio):
        return self.handles.popleft()

    def stop(self):
        self.stops += 1


def frame(timestamp):
    return RealtimeAudioFrame(b"\x00\x00" * 1_280, timestamp, 0.08)


def stream_frame(index, value, *, base=0.0):
    return RealtimeAudioFrame(
        bytes([value, 0]) * 1_280,
        base + (index + 1) * 0.08,
        0.08,
        sequence=index,
    )


def capture(base, text_byte=b"\x01\x00"):
    return UtteranceCapture(
        UtteranceEndReason.COMPLETE,
        text_byte * 1_600,
        base + 0.1,
        base + 0.4,
        base + 1.04,
    )


def coordinator(
    *,
    captures,
    texts,
    replies,
    source_frames=None,
    vad_scores=(),
    handles=None,
    tts=None,
    settings=None,
    local_executor=None,
):
    base = perf_counter()
    source = Source(source_frames or [frame(base)])
    wake = Wake()
    vad = VAD(vad_scores)
    segmenter = Segmenter(captures)
    voice_input = VoiceInput(texts)
    conversation = Conversation(replies)
    tts = tts or TTS(handles or [Handle(started_at=base + 2)])
    output = []
    instance = VoiceModeCoordinator(
        source,
        wake,
        vad,
        segmenter,
        voice_input,
        conversation,
        tts,
        local_command_executor=local_executor,
        settings=settings or VoiceModeSettings(),
        output_fn=output.append,
    )
    return instance, source, wake, vad, segmenter, voice_input, conversation, tts, output


def test_full_voice_turn_uses_normal_conversation_path_and_returns_to_wake_policy():
    base = perf_counter()
    values = coordinator(
        captures=[capture(base)],
        texts=["Look right."],
        replies=["There."],
        source_frames=[frame(base)],
        handles=[Handle(started_at=base + 2)],
        settings=VoiceModeSettings(debug_latency=True),
    )
    instance, source, wake, vad, _segmenter, voice_input, conversation, tts, output = values

    assert instance.run(max_interactions=1) == 0

    assert conversation.requests == ["Look right."]
    assert voice_input.audio
    assert tts.synthesized == ["There."]
    assert wake.calls == 1  # Mock playback completed before the barge monitor read.
    assert wake.warmups == vad.warmups == tts.warmups == 1
    assert wake.resets >= 3 and vad.resets >= 2
    assert source.started == source.stopped == 1
    assert source.drained == 2
    assert "Listening for wake word..." in output
    assert State.IDLE in instance.state.history
    assert instance.state.current is State.SHUTDOWN
    assert any(line.startswith("[LATENCY]") for line in output)


def test_idle_wake_loop_tolerates_one_transient_microphone_timeout():
    base = perf_counter()
    values = coordinator(captures=[], texts=[], replies=[], source_frames=[frame(base)])
    instance, source, *_rest = values
    original_read = source.read
    calls = 0

    def transient_read(timeout_seconds=1.0):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RealtimeAudioTimeoutError("transient")
        return original_read(timeout_seconds)

    source.read = transient_read

    detected_at = instance._wait_for_wake()

    assert detected_at >= base
    assert calls == 2
    assert instance.state.current is State.WAKE_DETECTED


def test_latency_debug_reports_rolling_wake_score_without_logging_audio():
    base = perf_counter()
    values = coordinator(
        captures=[],
        texts=[],
        replies=[],
        source_frames=[frame(base), frame(base + 0.08), frame(base + 0.16)],
        settings=VoiceModeSettings(debug_latency=True),
    )
    instance, _source, *_rest, output = values
    instance.wakeword = Wake((0.2, 0.4, 0.9))
    times = iter((0.0, 0.6, 1.1, 1.2))
    instance.clock = lambda: next(times)

    detected_at = instance._wait_for_wake()

    assert detected_at == 1.2
    assert "[WAKE] peak=0.400 threshold=0.500" in output
    assert "[VOICE] Wake detected (score=0.900)." in output


def test_no_speech_returns_to_idle_without_stt_llm_or_tts():
    base = perf_counter()
    no_speech = UtteranceCapture(UtteranceEndReason.NO_SPEECH, b"", None, None, base)
    values = coordinator(captures=[no_speech], texts=[], replies=[])
    instance, _source, _wake, _vad, _segmenter, voice_input, conversation, tts, output = values

    assert instance.run(max_interactions=1) == 0
    assert voice_input.audio == []
    assert conversation.requests == []
    assert tts.synthesized == []
    assert not any("No speech detected" in line for line in output)


@pytest.mark.parametrize("marker", ("[BLANK_AUDIO]", "[ Silence ]", "   ", ""))
def test_blank_whisper_output_never_reaches_conversation_or_tts(marker):
    base = perf_counter()
    values = coordinator(
        captures=[capture(base)],
        texts=[marker],
        replies=[],
        source_frames=[frame(base)],
        settings=VoiceModeSettings(debug_latency=True),
    )
    instance, _source, _wake, _vad, _segmenter, voice_input, conversation, tts, output = values

    assert instance.run(max_interactions=1) == 0
    assert voice_input.audio
    assert conversation.requests == []
    assert tts.synthesized == []
    assert "[VOICE] no_speech_discarded" in output


def test_insufficient_vad_confirmed_speech_is_rejected_before_stt():
    base = perf_counter()
    too_short = UtteranceCapture(
        UtteranceEndReason.COMPLETE,
        b"\x01\x00" * 100,
        base,
        base + 0.1,
        base + 0.2,
    )
    values = coordinator(
        captures=[too_short],
        texts=[],
        replies=[],
        source_frames=[frame(base)],
        settings=VoiceModeSettings(debug_latency=True),
    )
    instance, _source, _wake, _vad, _segmenter, voice_input, conversation, tts, output = values

    assert instance.run(max_interactions=1) == 0
    assert voice_input.audio == []
    assert conversation.requests == []
    assert tts.synthesized == []
    assert "[VOICE] no_speech_discarded" in output


def test_experimental_vad_barge_in_remains_explicitly_opt_in():
    base = perf_counter()
    first_handle = Handle(done=False, started_at=base + 2)
    second_handle = Handle(done=True, started_at=base + 5)
    values = coordinator(
        captures=[capture(base), capture(base + 2, b"\x02\x00")],
        texts=["Tell me something long.", "Tell me something else."],
        replies=["A longer answer.", "Another answer."],
        source_frames=[frame(base), frame(base + 2)],
        vad_scores=[0.95],
        handles=[first_handle, second_handle],
        settings=VoiceModeSettings(
            barge_in_enabled=True,
            barge_in_mode=BargeInMode.VAD_EXPERIMENTAL,
            barge_in_threshold=0.75,
            barge_in_suppression_ms=0,
            barge_in_min_speech_ms=60,
        ),
    )
    instance, _source, _wake, _vad, segmenter, _voice_input, conversation, _tts, output = values

    assert instance.run(max_interactions=2) == 0

    assert first_handle.stops == 1
    assert conversation.requests == ["Tell me something long.", "Tell me something else."]
    assert segmenter.calls[1][1]["speech_already_started"] is True
    assert State.INTERRUPTED in instance.state.history
    assert any("Experimental speech interruption" in line for line in output)


def test_default_wakeword_mode_ignores_generic_vad_during_playback():
    assert VoiceModeSettings().barge_in_mode is BargeInMode.WAKEWORD
    base = perf_counter()
    values = coordinator(
        captures=[],
        texts=[],
        replies=[],
        source_frames=[frame(base)],
        vad_scores=[0.99],
    )
    instance, _source, wake, vad, *_rest = values
    wake.scores.append(0.1)

    class CompletingHandle(Handle):
        def __init__(self):
            super().__init__(done=False, started_at=base)
            self.checks = 0

        @property
        def done(self):
            self.checks += 1
            return self.checks >= 2

    handle = CompletingHandle()

    interruption = instance._barge_capture(handle, VoiceLatencyTracker())

    assert interruption is None
    assert handle.stops == 0
    assert len(vad.scores) == 1  # Generic VAD was never consulted.


def test_wakeword_barge_in_cancels_and_local_stop_bypasses_conversation():
    base = perf_counter()
    controller = create_simulated_controller()
    assert controller.execute_intent(RobotIntent(RobotAction.MOVE_FORWARD)).success
    first_handle = Handle(done=False, started_at=base + 2)
    acknowledgement_handle = Handle(done=True, started_at=base + 5)
    values = coordinator(
        captures=[capture(base), capture(base + 2, b"\x02\x00")],
        texts=["Tell me something long.", "Hey Jarvis, stop."],
        replies=["A longer answer."],
        source_frames=[frame(base), frame(base + 1)],
        handles=[first_handle, acknowledgement_handle],
        settings=VoiceModeSettings(debug_latency=True),
        local_executor=SafeLocalVoiceCommandExecutor(controller),
    )
    instance, _source, _wake, _vad, segmenter, _voice_input, conversation, tts, output = values

    assert instance.run(max_interactions=2) == 0

    assert first_handle.stops == 1
    assert conversation.requests == ["Tell me something long."]
    assert controller.state.motion is MotionState.STOPPED
    assert tts.synthesized == ["A longer answer.", "Stopped."]
    assert segmenter.calls[1][1]["speech_already_started"] is False
    assert State.INTERRUPTED in instance.state.history
    assert any("[VOICE] wake_barge_in" in line for line in output)
    assert any(
        "[VOICE] wake_barge_in source=speaking score=0.900" in line
        for line in output
    )
    assert any("[VOICE] local_stop" in line for line in output)
    assert instance.latency.items[0].wake_to_playback_cancel is not None
    assert instance.latency.items[0].wake_to_playback_cancel < 0.5
    assert instance.latency.last.stt_to_local_stop is not None


def test_normal_wake_handoff_preserves_immediate_first_command_word():
    base = perf_counter()
    frames = [
        stream_frame(0, 1, base=base),  # wake plus the start of "how"
        stream_frame(1, 2, base=base),  # rest of "how old ..."
        *(stream_frame(index, 0, base=base) for index in range(2, 6)),
    ]
    source = Source(frames)
    wake = Wake((0.9,))
    vad = VAD([0.9, 0.9, 0.0, 0.0, 0.0, 0.0])
    segmenter = VADSegmenter(
        vad,
        VADSegmenterSettings(
            threshold=0.5,
            trailing_silence_ms=320,
            max_utterance_seconds=3,
            min_speech_ms=160,
            listen_timeout_seconds=8,
            pre_roll_ms=240,
        ),
    )
    voice_input = VoiceInput(["Hey Jarvis, how old is the universe?"])
    conversation = Conversation(["About 13.8 billion years."])
    tts = TTS([Handle(done=True)])
    output = []
    instance = VoiceModeCoordinator(
        source,
        wake,
        vad,
        segmenter,
        voice_input,
        conversation,
        tts,
        settings=VoiceModeSettings(debug_latency=True),
        output_fn=output.append,
    )

    wake_at = instance._wait_for_wake()
    utterance = instance._capture()

    assert utterance.pcm16.startswith(frames[0].pcm16 + frames[1].pcm16)
    assert utterance.handoff_frame_gap_seconds == pytest.approx(0.0)
    assert utterance.handoff_sequence_gap == 0
    assert instance._process_capture(utterance, wake_at=wake_at) is None
    assert voice_input.audio == [utterance.pcm16]
    assert conversation.requests == ["Hey Jarvis, how old is the universe?"]
    assert "BMO > About 13.8 billion years." in output
    assert any("[VOICE] wake_audio" in line for line in output)
    assert source.started == source.stopped == 0


def test_playback_start_reset_prevents_stale_wake_state_and_blank_turn():
    base = perf_counter()

    class StaleWake(Wake):
        def __init__(self):
            super().__init__()
            self.stale = True

        def process(self, _pcm16):
            self.calls += 1
            score = 0.9 if self.stale else 0.1
            return WakeWordDetection(
                score >= self.threshold, score, self.name, self.phrase
            )

        def reset(self):
            self.resets += 1
            self.stale = False

    class CompletingHandle(Handle):
        def __init__(self):
            super().__init__(done=False, started_at=base)
            self.checks = 0

        @property
        def done(self):
            self.checks += 1
            return self.checks >= 2

        def wait(self, _timeout=None):
            return PlaybackResult(True, "mock")

    source = Source([frame(base)])
    wake = StaleWake()
    vad = VAD(())
    segmenter = Segmenter([])
    voice_input = VoiceInput([])
    conversation = Conversation([])
    handle = CompletingHandle()
    tts = TTS([handle])
    output = []
    state = VoiceStateMachine(current=State.PROCESSING, history=[State.PROCESSING])
    instance = VoiceModeCoordinator(
        source,
        wake,
        vad,
        segmenter,
        voice_input,
        conversation,
        tts,
        state=state,
        output_fn=output.append,
    )

    assert instance._speak("No wake phrase here.", VoiceLatencyTracker()) is None
    assert handle.stops == 0
    assert wake.calls == 1
    assert wake.resets >= 3
    assert conversation.requests == []
    assert voice_input.audio == []
    assert instance.state.history[-2:] == [State.SPEAKING, State.IDLE]
    assert not any("Wake detected" in line for line in output)


def continuous_barge_coordinator(*, frames, vad_scores, texts):
    controller = create_simulated_controller()
    assert controller.execute_intent(RobotIntent(RobotAction.MOVE_FORWARD)).success
    source = Source(frames)
    wake = Wake((0.9,))
    vad = VAD(vad_scores)
    segmenter = VADSegmenter(
        vad,
        VADSegmenterSettings(
            threshold=0.5,
            trailing_silence_ms=320,
            max_utterance_seconds=3,
            min_speech_ms=160,
            listen_timeout_seconds=8,
            pre_roll_ms=240,
        ),
    )
    voice_input = VoiceInput(texts)
    conversation = Conversation([])
    tts = TTS([Handle(done=True)])
    output = []
    state = VoiceStateMachine(current=State.SPEAKING, history=[State.SPEAKING])
    instance = VoiceModeCoordinator(
        source,
        wake,
        vad,
        segmenter,
        voice_input,
        conversation,
        tts,
        local_command_executor=SafeLocalVoiceCommandExecutor(controller),
        settings=VoiceModeSettings(
            barge_in_pre_roll_ms=320,
            barge_in_command_start_timeout_seconds=1.5,
            debug_latency=True,
        ),
        state=state,
        output_fn=output.append,
    )
    return instance, controller, source, voice_input, conversation, tts, output


def test_continuous_wake_immediate_command_handoff_reaches_local_stop():
    base = perf_counter()
    frames = [
        stream_frame(0, 1, base=base),  # wake plus first STOP phoneme
        stream_frame(1, 2, base=base),  # remaining STOP audio
        *(stream_frame(index, 0, base=base) for index in range(2, 6)),
    ]
    values = continuous_barge_coordinator(
        frames=frames,
        vad_scores=[0.9, 0.9, 0.0, 0.0, 0.0, 0.0],
        texts=["Hey Jarvis, stop."],
    )
    instance, controller, source, voice_input, conversation, tts, output = values
    handle = Handle(done=False)

    interruption = instance._wakeword_barge_capture(
        handle, VoiceLatencyTracker()
    )

    assert interruption is not None
    assert handle.stops == 1
    assert interruption.capture.pcm16.startswith(frames[0].pcm16 + frames[1].pcm16)
    assert interruption.capture.handoff_frame_gap_seconds == pytest.approx(0.0)
    assert interruption.capture.handoff_sequence_gap == 0

    assert instance._process_capture(
        interruption.capture, wake_at=interruption.wake_at
    ) is None
    assert controller.state.motion is MotionState.STOPPED
    assert voice_input.audio == [interruption.capture.pcm16]
    assert conversation.requests == []
    assert tts.synthesized == ["Stopped."]
    assert source.started == source.stopped == 0
    assert any("first_frame_gap=0ms" in line for line in output)
    assert any("sequence_gap=0" in line for line in output)


def test_continuous_wake_short_pause_still_captures_following_stop():
    base = perf_counter()
    frames = [stream_frame(0, 1, base=base)]
    frames.extend(stream_frame(index, 0, base=base) for index in range(1, 7))
    frames.extend(stream_frame(index, 2, base=base) for index in range(7, 9))
    frames.extend(stream_frame(index, 0, base=base) for index in range(9, 13))
    values = continuous_barge_coordinator(
        frames=frames,
        vad_scores=[0.9] + [0.0] * 6 + [0.9, 0.9] + [0.0] * 4,
        texts=["stop"],
    )
    instance, controller, _source, _voice_input, conversation, _tts, _output = values

    interruption = instance._wakeword_barge_capture(
        Handle(done=False), VoiceLatencyTracker()
    )

    assert interruption is not None and interruption.capture.has_speech
    assert frames[7].pcm16 in interruption.capture.pcm16
    instance._process_capture(interruption.capture, wake_at=interruption.wake_at)
    assert controller.state.motion is MotionState.STOPPED
    assert conversation.requests == []


def test_continuous_wake_without_command_times_out_without_stt_or_qwen():
    base = perf_counter()
    frames = [stream_frame(0, 1, base=base)]
    frames.extend(stream_frame(index, 0, base=base) for index in range(1, 20))
    values = continuous_barge_coordinator(
        frames=frames,
        vad_scores=[0.9] + [0.0] * 19,
        texts=[],
    )
    instance, controller, _source, voice_input, conversation, tts, output = values

    interruption = instance._wakeword_barge_capture(
        Handle(done=False), VoiceLatencyTracker()
    )

    assert interruption is not None
    assert interruption.capture.reason is UtteranceEndReason.NO_SPEECH
    assert instance._process_capture(
        interruption.capture, wake_at=interruption.wake_at
    ) is None
    assert controller.state.motion is MotionState.FORWARD
    assert voice_input.audio == []
    assert conversation.requests == []
    assert tts.synthesized == []
    assert instance.state.current is State.IDLE
    assert "[VOICE] no_speech_discarded" in output


def test_wake_only_transcript_is_discarded_before_qwen():
    base = perf_counter()
    values = coordinator(
        captures=[],
        texts=["Hey Jarvis."],
        replies=[],
        settings=VoiceModeSettings(debug_latency=True),
    )
    instance, _source, _wake, _vad, _segmenter, voice_input, conversation, tts, output = values
    instance.state = VoiceStateMachine(current=State.LISTENING, history=[State.LISTENING])

    assert instance._process_capture(capture(base), wake_at=base) is None
    assert voice_input.audio
    assert conversation.requests == []
    assert tts.synthesized == []
    assert "[VOICE] no_speech_discarded" in output


def test_warmup_failure_falls_back_to_safe_textless_error_and_clean_shutdown():
    base = perf_counter()
    failure = SynthesisFailure(SynthesisErrorCode.MODEL_LOAD_FAILED, "model failed")
    tts = TTS([], warmup_failure=failure)
    values = coordinator(captures=[], texts=[], replies=[], source_frames=[frame(base)], tts=tts)
    instance, source, _wake, _vad, _segmenter, _voice_input, conversation, _tts, output = values

    assert instance.run(max_interactions=1) == 1
    assert conversation.requests == []
    assert source.started == 0 and source.stopped == 1
    assert instance.state.history[-3:] == [State.ERROR, State.IDLE, State.SHUTDOWN]
    assert any("TTS warmup failed" in line for line in output)


def test_warm_synthesis_failure_keeps_assistant_text_and_returns_safely():
    base = perf_counter()
    failure = SynthesisFailure(SynthesisErrorCode.SYNTHESIS_FAILED, "voice failed")
    tts = TTS([], synthesis_failure=failure)
    values = coordinator(
        captures=[capture(base)],
        texts=["Hello."],
        replies=["Hey."],
        source_frames=[frame(base)],
        tts=tts,
    )
    instance, _source, _wake, _vad, _segmenter, _voice_input, conversation, _tts, output = values

    assert instance.run(max_interactions=1) == 0
    assert conversation.requests == ["Hello."]
    assert "BMO > Hey." in output
    assert any("voice failed" in line for line in output)


def test_continuous_voice_robot_request_uses_existing_structured_tool_and_safety_path():
    class Provider:
        name = "mock"
        endpoint = "memory://voice-tool"

        def __init__(self):
            self.responses = iter(
                [
                    LLMResponse("", "mock", tool_calls=(ToolCall("look_right", {}),)),
                    LLMResponse("There.", "mock"),
                ]
            )

        def generate(self, _request, *, cancellation=None):
            return next(self.responses)

        def close(self):
            pass

    base = perf_counter()
    controller = create_simulated_controller()
    tools = RobotToolPolicy(RobotToolRegistry(), controller)
    conversation = ConversationService(
        Provider(),
        ConversationSettings(model="mock"),
        system_prompt="Test",
        tool_executor=tools,
    )
    values = coordinator(
        captures=[capture(base)],
        texts=["Look right."],
        replies=[],
        source_frames=[frame(base)],
        handles=[Handle(started_at=base + 2)],
    )
    instance = values[0]
    instance.conversation = conversation

    assert instance.run(max_interactions=1) == 0
    assert controller.state.head.value == "right"
