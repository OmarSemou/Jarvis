import ast
import wave
from pathlib import Path

import pytest

from jarvis.audio.tts.base import SpeechSynthesisResult, SynthesizedAudio
from jarvis.audio.tts.benchmark import (
    TTS_BENCHMARK_PHRASES,
    cleanup_tts_benchmark,
    format_tts_benchmark_report,
    run_tts_benchmark,
)
from jarvis.cli import tts_benchmark_clean_command, tts_benchmark_command
from jarvis.core.config import JarvisConfig
from jarvis.core.paths import JarvisPaths


ROOT = Path(__file__).resolve().parents[1]


class Provider:
    def __init__(self, name, voices):
        self.name = name
        self.available_voices = voices
        self.calls = []

    def readiness_error(self, _voice):
        return None

    def synthesize(self, text, *, voice, speed, language):
        self.calls.append((text, voice, speed, language))
        return SpeechSynthesisResult(
            True,
            self.name,
            voice,
            0.1,
            SynthesizedAudio(b"\x00\x00" * 1000, 1000),
            0.05,
        )


def test_fixed_phrase_set_has_short_medium_and_natural_long_text():
    assert len(TTS_BENCHMARK_PHRASES) == 8
    assert TTS_BENCHMARK_PHRASES[:5] == (
        "Hey.",
        "Alright.",
        "I'm following you.",
        "There's something in the way.",
        "The emergency stop is active.",
    )
    assert 40 <= len(TTS_BENCHMARK_PHRASES[7].split()) <= 60


def test_benchmark_writes_labeled_wavs_and_aggregates_timings(tmp_path):
    providers = {
        "kokoro": Provider("kokoro", ("a", "b")),
        "piper": Provider("piper", ("c",)),
    }
    report = run_tts_benchmark(providers, tmp_path / "tts", timestamp="20260828T120000Z")

    assert report.successful
    assert len(report.samples) == 24
    assert len(report.summaries) == 3
    assert report.summaries[0].median_seconds == 0.1
    assert report.summaries[0].short_median_seconds == 0.1
    assert report.summaries[0].median_rtf == 0.1
    for sample in report.samples:
        assert sample.output_file.parent == report.output_dir
        with wave.open(str(sample.output_file), "rb") as handle:
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 1000
    rendered = format_tts_benchmark_report(report)
    assert "Voice quality and pronunciation require human listening" in rendered


def test_cleanup_only_removes_one_direct_benchmark_run(tmp_path):
    root = tmp_path / "benchmarks" / "tts"
    run = root / "20260828T120000Z"
    run.mkdir(parents=True)
    (run / "sample.wav").write_bytes(b"audio")
    cleanup_tts_benchmark(run, root)
    assert not run.exists()
    assert root.exists()

    with pytest.raises(ValueError, match="direct benchmark run"):
        cleanup_tts_benchmark(root, root)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="direct benchmark run"):
        cleanup_tts_benchmark(outside, root)


def test_benchmark_module_has_no_llm_stt_network_or_playback_imports():
    tree = ast.parse(
        (ROOT / "jarvis" / "audio" / "tts" / "benchmark.py").read_text(encoding="utf-8")
    )
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    forbidden = ("jarvis.llm", "jarvis.audio.stt", "ollama", "httpx", "requests", "sounddevice")
    assert not any(name.startswith(forbidden) for name in imports)


def test_cli_benchmark_uses_injected_tts_only_and_cleanup_is_explicit(tmp_path, monkeypatch):
    paths = JarvisPaths.from_repository_root(tmp_path.resolve())
    provider = Provider("kokoro", ("a",))
    output = []
    monkeypatch.setattr(
        "jarvis.cli.create_conversation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LLM constructed")),
    )
    monkeypatch.setattr(
        "jarvis.cli.create_voice_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("STT constructed")),
    )

    result = tts_benchmark_command(
        output_fn=output.append,
        paths=paths,
        config=JarvisConfig(),
        providers={"kokoro": provider},
    )

    assert result == 0
    assert len(provider.calls) == 8
    runs = tuple(paths.tts_benchmark_dir.iterdir())
    assert len(runs) == 1
    assert tts_benchmark_clean_command(
        str(runs[0]), output_fn=output.append, paths=paths
    ) == 0
    assert not runs[0].exists()
