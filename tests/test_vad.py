from collections import deque

import pytest

from jarvis.audio.realtime import RealtimeAudioFrame
from jarvis.audio.vad.segmenter import (
    SpeechStartGate,
    UtteranceEndReason,
    VADSegmenter,
    VADSegmenterSettings,
)
from jarvis.audio.vad.silero import SileroVAD, SileroVADSettings


class VAD:
    name = "mock-vad"

    def __init__(self, scores):
        self.scores = deque(scores)
        self.resets = 0

    def readiness_error(self):
        return None

    def warmup(self):
        return None

    def score(self, _pcm16):
        return self.scores.popleft()

    def reset(self):
        self.resets += 1


class Source:
    frame_duration_ms = 100

    def __init__(self, count, *, start=0.1):
        self.frames = deque(
            RealtimeAudioFrame(b"\x01\x00" * 1_600, start + index * 0.1, 0.1)
            for index in range(count)
        )

    def read(self, timeout_seconds=1.0):
        return self.frames.popleft()

    def drain(self):
        return 0


def _settings(**overrides):
    values = dict(
        threshold=0.5,
        trailing_silence_ms=300,
        max_utterance_seconds=3,
        min_speech_ms=200,
        listen_timeout_seconds=1,
        pre_roll_ms=100,
    )
    values.update(overrides)
    return VADSegmenterSettings(**values)


def test_vad_detects_sustained_start_and_trailing_silence_end():
    scores = [0.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0]
    segmenter = VADSegmenter(VAD(scores), _settings())

    capture = segmenter.capture(Source(len(scores)))

    assert capture.reason is UtteranceEndReason.COMPLETE
    assert capture.has_speech
    assert capture.duration_seconds == pytest.approx(0.3)
    assert len(capture.pcm16) > 0
    assert capture.end_detected_at - capture.speech_ended_at == pytest.approx(0.3)


def test_vad_rejects_noise_only_and_returns_no_audio():
    segmenter = VADSegmenter(VAD([0.1] * 10), _settings())

    capture = segmenter.capture(Source(10))

    assert capture.reason is UtteranceEndReason.NO_SPEECH
    assert not capture.has_speech
    assert capture.pcm16 == b""


def test_wake_barge_pre_roll_is_bounded_and_preserved_before_new_speech():
    initial = (
        RealtimeAudioFrame(b"\x02\x00" * 1_600, 0.1, 0.1),
        RealtimeAudioFrame(b"\x03\x00" * 1_600, 0.2, 0.1),
    )
    scores = [0.9, 0.9, 0.0, 0.0, 0.0]
    segmenter = VADSegmenter(VAD(scores), _settings(pre_roll_ms=200))

    capture = segmenter.capture(Source(len(scores), start=0.3), initial_frames=initial)

    assert capture.has_speech
    assert capture.pcm16.startswith(initial[0].pcm16 + initial[1].pcm16)


def test_wake_handoff_candidate_and_first_live_frame_confirm_no_pause_speech():
    candidate = RealtimeAudioFrame(
        b"\x02\x00" * 1_600, 0.1, 0.1, sequence=4
    )
    source = Source(4, start=0.2)
    source.frames = deque(
        RealtimeAudioFrame(frame.pcm16, frame.captured_at, 0.1, sequence=5 + index)
        for index, frame in enumerate(source.frames)
    )
    segmenter = VADSegmenter(
        VAD([0.9, 0.9, 0.0, 0.0, 0.0]),
        _settings(min_speech_ms=200, pre_roll_ms=200),
    )

    capture = segmenter.capture(source, candidate_frames=(candidate,))

    assert capture.has_speech
    assert capture.pcm16.startswith(candidate.pcm16)
    assert capture.handoff_frame_gap_seconds == pytest.approx(0.0)
    assert capture.handoff_sequence_gap == 0


def test_handoff_candidates_cannot_confirm_speech_without_a_new_live_frame():
    candidates = (
        RealtimeAudioFrame(b"\x02\x00" * 1_600, 0.1, 0.1),
        RealtimeAudioFrame(b"\x03\x00" * 1_600, 0.2, 0.1),
    )
    segmenter = VADSegmenter(
        VAD([0.9, 0.9] + [0.0] * 5),
        _settings(min_speech_ms=300),
    )

    capture = segmenter.capture(
        Source(5, start=0.3),
        candidate_frames=candidates,
        speech_start_timeout_seconds=0.5,
    )

    assert capture.reason is UtteranceEndReason.NO_SPEECH
    assert not capture.has_speech


def test_vad_enforces_maximum_utterance_duration():
    segmenter = VADSegmenter(VAD([0.9] * 40), _settings(min_speech_ms=100))

    capture = segmenter.capture(Source(40))

    assert capture.reason is UtteranceEndReason.MAX_DURATION
    assert capture.duration_seconds >= 2.9


def test_barge_gate_requires_sustained_speech_after_echo_suppression():
    vad = VAD([0.9, 0.9, 0.9, 0.9])
    gate = SpeechStartGate(
        vad, threshold=0.75, min_speech_ms=200, suppression_ms=500
    )
    frames = tuple(Source(4).frames)

    assert not gate.process(frames[0], 100)
    assert not gate.process(frames[1], 550)
    assert gate.process(frames[2], 650)
    assert len(gate.take_candidate_frames()) == 2


def test_silero_adapter_loads_lazily_and_clamps_score(tmp_path, monkeypatch):
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"model")

    class Engine:
        prediction_buffer = []

        def predict(self, _samples, frame_size):
            assert frame_size == 480
            return 1.2

        def reset_states(self):
            pass

    loads = []
    provider = SileroVAD(
        SileroVADSettings(model.resolve()),
        engine_loader=lambda path: loads.append(path) or Engine(),
        package_probe=lambda _name: True,
    )

    assert provider.readiness_error() is None and loads == []
    assert provider.score(b"\x00\x00" * 480) == 1.0
    assert loads == [model.resolve()]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 1.0},
        {"trailing_silence_ms": 100},
        {"max_utterance_seconds": 2},
        {"min_speech_ms": 20},
    ],
)
def test_vad_setting_ranges(kwargs):
    with pytest.raises(ValueError):
        _settings(**kwargs)
