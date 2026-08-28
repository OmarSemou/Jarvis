import pytest

from jarvis.audio.tts.base import SpeechSynthesisResult, SynthesizedAudio


def test_synthesized_audio_reports_frames_duration_and_requires_complete_pcm16():
    audio = SynthesizedAudio(b"\x00\x00" * 24_000, 24_000, 1)

    assert audio.frame_count == 24_000
    assert audio.duration_seconds == 1.0

    with pytest.raises(ValueError, match="complete interleaved frames"):
        SynthesizedAudio(b"\x00\x00", 24_000, 2)


@pytest.mark.parametrize("channels", [0, 3])
def test_synthesized_audio_accepts_only_mono_or_stereo(channels):
    with pytest.raises(ValueError, match="channels"):
        SynthesizedAudio(b"\x00\x00", 24_000, channels)


def test_synthesis_result_enforces_success_failure_invariants():
    with pytest.raises(ValueError, match="successful synthesis requires audio"):
        SpeechSynthesisResult(True, "mock", "voice", 0.1)
    with pytest.raises(ValueError, match="failed synthesis requires"):
        SpeechSynthesisResult(False, "mock", "voice", 0.1)
