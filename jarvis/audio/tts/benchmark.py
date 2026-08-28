"""Fixed-corpus, LLM-free and STT-free local TTS benchmark."""

from __future__ import annotations

import shutil
import statistics
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .base import SpeechSynthesisResult, TTSProvider


TTS_BENCHMARK_PHRASES = (
    "Hey.",
    "Alright.",
    "I'm following you.",
    "There's something in the way.",
    "The emergency stop is active.",
    "It looks like rain later this evening, so I'd probably bring an umbrella.",
    "I found the information you asked for. The newer option is faster, but the older one is considerably cheaper.",
    (
        "I checked the plan and everything important is ready for tomorrow morning. "
        "The first part should be straightforward, though traffic may add a few minutes to the journey. "
        "If anything changes overnight, we can adjust before leaving without making a production of it."
    ),
)


@dataclass(frozen=True, slots=True)
class TTSBenchmarkSample:
    provider: str
    voice: str
    phrase_number: int
    text: str
    result: SpeechSynthesisResult
    output_file: Path | None


@dataclass(frozen=True, slots=True)
class TTSBenchmarkSummary:
    provider: str
    voice: str
    successful_samples: int
    median_seconds: float | None
    mean_seconds: float | None
    median_rtf: float | None
    fastest_seconds: float | None
    slowest_seconds: float | None
    short_median_seconds: float | None


@dataclass(frozen=True, slots=True)
class TTSBenchmarkReport:
    output_dir: Path
    samples: tuple[TTSBenchmarkSample, ...]
    summaries: tuple[TTSBenchmarkSummary, ...]

    @property
    def successful(self) -> bool:
        return bool(self.samples) and all(sample.result.success for sample in self.samples)


def _write_wave(path: Path, result: SpeechSynthesisResult) -> None:
    if result.audio is None:
        raise ValueError("Cannot write a failed synthesis result")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(result.audio.channels)
        handle.setsampwidth(2)
        handle.setframerate(result.audio.sample_rate)
        handle.writeframes(result.audio.pcm16)


def _summarize(samples: list[TTSBenchmarkSample]) -> TTSBenchmarkSummary:
    successful = [sample for sample in samples if sample.result.success]
    elapsed = [sample.result.elapsed_seconds for sample in successful]
    rtfs = [
        value
        for sample in successful
        if (value := sample.result.real_time_factor) is not None
    ]
    short_elapsed = [
        sample.result.elapsed_seconds
        for sample in successful
        if sample.phrase_number <= 5
    ]
    first = samples[0]
    return TTSBenchmarkSummary(
        first.provider,
        first.voice,
        len(successful),
        statistics.median(elapsed) if elapsed else None,
        statistics.fmean(elapsed) if elapsed else None,
        statistics.median(rtfs) if rtfs else None,
        min(elapsed) if elapsed else None,
        max(elapsed) if elapsed else None,
        statistics.median(short_elapsed) if short_elapsed else None,
    )


def run_tts_benchmark(
    providers: dict[str, TTSProvider],
    benchmark_root: Path,
    *,
    phrases: tuple[str, ...] = TTS_BENCHMARK_PHRASES,
    timestamp: str | None = None,
) -> TTSBenchmarkReport:
    """Synthesize every curated voice and retain WAVs beneath benchmark_root."""

    if len(phrases) != 8:
        raise ValueError("The Phase 2C2 benchmark must contain exactly eight phrases")
    root = benchmark_root.resolve()
    run_name = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    if not run_name or any(character not in "0123456789TZ.-" for character in run_name):
        raise ValueError("Benchmark timestamp contains unsafe filename characters")
    output_dir = (root / run_name).resolve()
    if output_dir.parent != root:
        raise ValueError("Benchmark output must be a direct child of its configured root")
    output_dir.mkdir(parents=True, exist_ok=False)

    samples: list[TTSBenchmarkSample] = []
    groups: list[list[TTSBenchmarkSample]] = []
    for provider_name in sorted(providers):
        provider = providers[provider_name]
        for voice in provider.available_voices:
            group: list[TTSBenchmarkSample] = []
            for phrase_number, phrase in enumerate(phrases, start=1):
                result = provider.synthesize(phrase, voice=voice, speed=1.0, language="en")
                output_file: Path | None = None
                if result.success and result.audio is not None:
                    output_file = output_dir / (
                        f"{provider.name}__{voice}__{phrase_number:02d}.wav"
                    )
                    _write_wave(output_file, result)
                sample = TTSBenchmarkSample(
                    provider.name,
                    voice,
                    phrase_number,
                    phrase,
                    result,
                    output_file,
                )
                samples.append(sample)
                group.append(sample)
            groups.append(group)
    return TTSBenchmarkReport(
        output_dir,
        tuple(samples),
        tuple(_summarize(group) for group in groups),
    )


def cleanup_tts_benchmark(run_dir: Path, benchmark_root: Path) -> None:
    """Remove one explicit benchmark run, never the benchmark root or an ancestor."""

    root = benchmark_root.resolve()
    target = run_dir.resolve()
    if target.parent != root or target == root:
        raise ValueError("Only a direct benchmark run directory may be removed")
    if not target.is_dir():
        raise ValueError(f"Benchmark run does not exist: {target}")
    shutil.rmtree(target)


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def format_tts_benchmark_report(report: TTSBenchmarkReport) -> str:
    lines = ["TTS benchmark samples", ""]
    for sample in report.samples:
        result = sample.result
        if not result.success:
            message = result.error.message if result.error is not None else "unknown failure"
            lines.append(
                f"{sample.provider}/{sample.voice} #{sample.phrase_number}: FAILED - {message}"
            )
            continue
        duration = result.speech_duration_seconds
        rtf = result.real_time_factor
        first = _seconds(result.first_audio_seconds)
        lines.append(
            f"{sample.provider}/{sample.voice} #{sample.phrase_number}: "
            f"synthesis={result.elapsed_seconds:.3f}s speech={duration:.3f}s "
            f"RTF={rtf:.3f} rate={result.audio.sample_rate}Hz first={first} "
            f"file={sample.output_file.name}"
        )
    lines.extend(("", "Summary"))
    for summary in report.summaries:
        lines.append(
            f"{summary.provider}/{summary.voice}: {summary.successful_samples}/8; "
            f"median={_seconds(summary.median_seconds)} mean={_seconds(summary.mean_seconds)} "
            f"median RTF={'n/a' if summary.median_rtf is None else f'{summary.median_rtf:.3f}'} "
            f"fastest={_seconds(summary.fastest_seconds)} slowest={_seconds(summary.slowest_seconds)} "
            f"short median={_seconds(summary.short_median_seconds)}"
        )
    lines.extend(
        (
            "",
            f"Listen to WAV samples in: {report.output_dir}",
            "Voice quality and pronunciation require human listening; timings do not select a winner.",
            "Remove this run later with: python -m jarvis tts-benchmark-clean \"<run directory>\"",
        )
    )
    return "\n".join(lines)
