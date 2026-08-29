import pytest

from jarvis.audio.voice.latency import (
    LatencyHistory,
    VoiceLatencyMetrics,
    VoiceLatencyTracker,
    format_latency,
)


def test_structured_latency_fields_are_calculated_from_real_event_times():
    tracker = VoiceLatencyTracker()
    for name, timestamp in (
        ("wake", 1.0),
        ("speech_start", 1.4),
        ("speech_end", 2.4),
        ("end_detected", 3.0),
        ("stt_start", 3.0),
        ("stt_end", 4.1),
        ("llm_start", 4.1),
        ("llm_end", 4.5),
        ("tts_start", 4.5),
        ("tts_end", 4.7),
        ("playback_requested", 4.7),
        ("playback_started", 4.75),
        ("barge_wake", 5.0),
        ("playback_cancelled", 5.3),
        ("local_stop_executed", 4.11),
    ):
        tracker.mark(name, timestamp)

    metrics = tracker.metrics()

    assert metrics.wake_to_speech_start == pytest.approx(0.4)
    assert metrics.end_detection_delay == pytest.approx(0.6)
    assert metrics.stt == pytest.approx(1.1)
    assert metrics.llm_tools == pytest.approx(0.4)
    assert metrics.tts == pytest.approx(0.2)
    assert metrics.total_response_start == pytest.approx(2.35)
    assert metrics.wake_to_playback_cancel == pytest.approx(0.3)
    assert metrics.stt_to_local_stop == pytest.approx(0.01)
    assert "total response start: 2.35s" in format_latency(metrics)
    assert "wake-to-playback-cancel: 0.30s" in format_latency(metrics)
    assert "STT-to-local-stop: 0.01s" in format_latency(metrics)


def test_latency_history_aggregates_only_available_fields():
    history = LatencyHistory()
    history.add(VoiceLatencyMetrics(stt=1.0, tts=0.2))
    history.add(VoiceLatencyMetrics(stt=1.4, tts=0.4, llm_tools=0.3))

    averages = history.averages()

    assert history.last.stt == 1.4
    assert averages.stt == pytest.approx(1.2)
    assert averages.tts == pytest.approx(0.3)
    assert averages.llm_tools == pytest.approx(0.3)


def test_negative_or_impossible_latency_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        VoiceLatencyMetrics(stt=-0.1)

    tracker = VoiceLatencyTracker()
    tracker.mark("stt_start", 2)
    tracker.mark("stt_end", 1)
    with pytest.raises(ValueError, match="Impossible"):
        tracker.metrics()
