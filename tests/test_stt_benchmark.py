from pathlib import Path

import pytest

from jarvis.audio.benchmark import (
    PASS_NAMES,
    BenchmarkPhrase,
    BenchmarkRecording,
    MatchKind,
    cleanup_recordings,
    compare_phrase,
    format_benchmark_report,
    normalize_phrase,
    run_benchmark,
    summarize_model,
)
from jarvis.audio.devices import AudioDevice
from jarvis.audio.formats import write_pcm16_mono_wav
from jarvis.audio.recorder import AudioRecording, RecordingSession
from jarvis.audio.stt.base import (
    TranscriptionErrorCode,
    TranscriptionFailure,
    TranscriptionResult,
)
from jarvis.audio.stt.whisper_cpp import WhisperBenchmarkResult, WhisperTimings
from jarvis.cli import create_stt_provider, stt_benchmark_command
from jarvis.core.config import JarvisConfig
from jarvis.core.paths import JarvisPaths


class FakeBenchmarkProvider:
    def __init__(self, model, transcripts=None, readiness_error=None):
        self.model = model
        self.transcripts = transcripts or {}
        self._readiness_error = readiness_error
        self.calls = []

    def readiness_error(self):
        return self._readiness_error

    def benchmark_transcribe(self, audio_path):
        path = Path(audio_path)
        self.calls.append(path)
        text = self.transcripts.get(path, "Stop.")
        elapsed = (0.5 if self.model == "base" else 1.0) + len(self.calls) / 100
        return WhisperBenchmarkResult(
            TranscriptionResult(True, text, "fake-whisper", elapsed, 2.0),
            WhisperTimings(
                load_seconds=elapsed / 4,
                encode_seconds=elapsed / 2,
                decode_seconds=elapsed / 8,
                total_seconds=elapsed * 0.95,
            ),
        )


def make_recording(tmp_path, name, phrase):
    path = (tmp_path / f"{name}.wav").resolve()
    write_pcm16_mono_wav(path, b"\x00\x00" * 32_000)
    return BenchmarkRecording(phrase, AudioRecording(path, 2.0))


def test_phrase_normalization_is_lightweight_and_preserves_words():
    assert normalize_phrase("  HEJ   Jarvis?!  ") == "hej jarvis"
    assert compare_phrase("Hello Jarvis.", "hello jarvis") is MatchKind.NORMALIZED
    assert compare_phrase("Stop.", "Stop.") is MatchKind.EXACT
    assert compare_phrase("Look left.", "Look right.") is MatchKind.MISS


def test_benchmark_reuses_identical_wav_paths_for_both_models_and_passes(tmp_path):
    first = make_recording(tmp_path, "first", BenchmarkPhrase("en", "Hello Jarvis."))
    second = make_recording(tmp_path, "second", BenchmarkPhrase("da", "Stop."))
    transcripts = {first.audio.path: "hello jarvis", second.audio.path: "Stop."}
    base = FakeBenchmarkProvider("base", transcripts)
    small = FakeBenchmarkProvider("small", transcripts)

    report = run_benchmark((first, second), {"base": base, "small": small})

    expected_paths = [first.audio.path, second.audio.path] * 2
    assert base.calls == expected_paths
    assert small.calls == expected_paths
    assert all(sample.audio_path in expected_paths for sample in report.samples)
    assert len(report.samples) == 8


def test_timing_aggregation_reports_warm_accuracy_by_language(tmp_path):
    english = make_recording(tmp_path, "en", BenchmarkPhrase("en", "Hello Jarvis."))
    danish = make_recording(tmp_path, "da", BenchmarkPhrase("da", "Kig til højre."))
    base = FakeBenchmarkProvider(
        "base",
        {english.audio.path: "hello jarvis", danish.audio.path: "wrong"},
    )
    small = FakeBenchmarkProvider(
        "small",
        {english.audio.path: "Hello Jarvis.", danish.audio.path: "Kig til højre"},
    )

    report = run_benchmark((english, danish), {"base": base, "small": small})
    summary = summarize_model(report, "small")

    assert summary.pass_name == PASS_NAMES[1]
    assert summary.sample_count == 2
    assert summary.mean_seconds == pytest.approx(1.035)
    assert summary.median_seconds == pytest.approx(1.035)
    assert summary.fastest_seconds == pytest.approx(1.03)
    assert summary.slowest_seconds == pytest.approx(1.04)
    assert summary.recognition_successes == 2
    assert (summary.english_successes, summary.english_total) == (1, 1)
    assert (summary.danish_successes, summary.danish_total) == (1, 1)
    rendered = format_benchmark_report(report)
    assert "process-per-command" in rendered
    assert "reloads the model" in rendered


def test_benchmark_recording_cleanup_is_scoped_to_captured_files(tmp_path):
    recording = make_recording(tmp_path, "captured", BenchmarkPhrase("en", "Stop."))
    unrelated = tmp_path / "keep.wav"
    unrelated.write_bytes(b"keep")

    warnings = cleanup_recordings((recording,))

    assert warnings == ()
    assert not recording.audio.path.exists()
    assert unrelated.read_bytes() == b"keep"


class FakeRecorder:
    def __init__(self, root):
        self.root = root
        self.device = AudioDevice(2, "Fake mic", 1, 16_000, True)
        self.devices = type("Devices", (), {"configured_device": None})()
        self.started = 0
        self.stopped = 0
        self.cancelled = 0
        self.created = []

    def start(self):
        self.started += 1
        return RecordingSession(self.device, 16_000)

    def stop(self):
        self.stopped += 1
        path = (self.root / f"benchmark-{self.stopped}.wav").resolve()
        write_pcm16_mono_wav(path, b"\x00\x00" * 1_600)
        self.created.append(path)
        return AudioRecording(path, 0.1)

    def cancel(self):
        self.cancelled += 1


def test_benchmark_command_uses_no_llm_and_cleans_recordings(tmp_path, monkeypatch):
    paths = JarvisPaths.from_repository_root(tmp_path.resolve())
    recorder = FakeRecorder(tmp_path)
    base = FakeBenchmarkProvider("base")
    small = FakeBenchmarkProvider("small")
    output = []

    def llm_bomb(*_args, **_kwargs):
        raise AssertionError("benchmark must not construct an LLM")

    monkeypatch.setattr("jarvis.cli.create_conversation", llm_bomb)
    result = stt_benchmark_command(
        input_fn=lambda _prompt: "",
        output_fn=output.append,
        paths=paths,
        config=JarvisConfig(),
        recorder=recorder,
        providers={"base": base, "small": small},
    )

    assert result == 0
    assert recorder.started == recorder.stopped == 11
    assert len(base.calls) == len(small.calls) == 22
    assert all(not path.exists() for path in recorder.created)
    assert any("no LLM or network calls" in line for line in output)


def test_missing_benchmark_model_fails_before_microphone_capture(tmp_path):
    missing = TranscriptionFailure(
        TranscriptionErrorCode.MODEL_MISSING,
        "base model missing; run setup -Models base",
    )
    recorder = FakeRecorder(tmp_path)
    result = stt_benchmark_command(
        input_fn=lambda _prompt: "",
        output_fn=lambda _line: None,
        paths=JarvisPaths.from_repository_root(tmp_path.resolve()),
        config=JarvisConfig(),
        recorder=recorder,
        providers={
            "base": FakeBenchmarkProvider("base", readiness_error=missing),
            "small": FakeBenchmarkProvider("small"),
        },
    )

    assert result == 1
    assert recorder.started == 0


@pytest.mark.parametrize(
    ("selected", "filename"),
    [("base", "ggml-base.bin"), ("small", "ggml-small.bin")],
)
def test_selected_symbolic_model_resolves_to_allowlisted_path(tmp_path, selected, filename):
    paths = JarvisPaths.from_repository_root(tmp_path.resolve())
    provider = create_stt_provider(JarvisConfig(stt_model=selected), paths)

    assert provider.settings.model_name == selected
    assert provider.settings.model_path == paths.local_models_dir / "whisper" / filename
