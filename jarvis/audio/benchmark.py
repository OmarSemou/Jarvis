"""Hardware-independent orchestration and reporting for local STT benchmarks."""

from __future__ import annotations

import statistics
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .recorder import AudioRecording
from .stt.whisper_cpp import WhisperBenchmarkResult, WhisperTimings


@dataclass(frozen=True, slots=True)
class BenchmarkPhrase:
    language: str
    expected: str


BENCHMARK_PHRASES = (
    BenchmarkPhrase("en", "Hello Jarvis."),
    BenchmarkPhrase("en", "Look to your right."),
    BenchmarkPhrase("en", "Point to your left."),
    BenchmarkPhrase("en", "Follow me."),
    BenchmarkPhrase("en", "Stop."),
    BenchmarkPhrase("en", "What is the weather like today?"),
    BenchmarkPhrase("da", "Hej Jarvis."),
    BenchmarkPhrase("da", "Kig til højre."),
    BenchmarkPhrase("da", "Peg til venstre."),
    BenchmarkPhrase("da", "Følg efter mig."),
    BenchmarkPhrase("da", "Stop."),
)


@dataclass(frozen=True, slots=True)
class BenchmarkRecording:
    phrase: BenchmarkPhrase
    audio: AudioRecording


class MatchKind(StrEnum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    MISS = "miss"


class BenchmarkSTT(Protocol):
    def benchmark_transcribe(self, audio_path: Path) -> WhisperBenchmarkResult: ...


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    phrase: BenchmarkPhrase
    model: str
    pass_name: str
    audio_path: Path
    transcript: str
    audio_seconds: float
    transcription_seconds: float
    match: MatchKind
    provider_success: bool
    timings: WhisperTimings | None = None
    error: str | None = None

    @property
    def real_time_factor(self) -> float | None:
        if self.audio_seconds <= 0:
            return None
        return self.transcription_seconds / self.audio_seconds


@dataclass(frozen=True, slots=True)
class ModelSummary:
    model: str
    pass_name: str
    sample_count: int
    mean_seconds: float
    median_seconds: float
    fastest_seconds: float
    slowest_seconds: float
    recognition_successes: int
    english_successes: int
    english_total: int
    danish_successes: int
    danish_total: int


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    samples: tuple[BenchmarkSample, ...]

    def samples_for(self, model: str, pass_name: str) -> tuple[BenchmarkSample, ...]:
        return tuple(
            sample
            for sample in self.samples
            if sample.model == model and sample.pass_name == pass_name
        )


PASS_NAMES = ("pass 1 (cold-ish)", "pass 2 (warm-cache)")
_TERMINAL_PUNCTUATION = ".,!?;:…"


def normalize_phrase(value: str) -> str:
    """Normalize case, spacing, Unicode form, and terminal punctuation only."""

    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = " ".join(normalized.split())
    return normalized.rstrip(_TERMINAL_PUNCTUATION).rstrip()


def compare_phrase(expected: str, actual: str) -> MatchKind:
    if expected.strip() == actual.strip():
        return MatchKind.EXACT
    if normalize_phrase(expected) == normalize_phrase(actual):
        return MatchKind.NORMALIZED
    return MatchKind.MISS


def run_benchmark(
    recordings: Sequence[BenchmarkRecording],
    providers: Mapping[str, BenchmarkSTT],
) -> BenchmarkReport:
    """Run two sequential process-per-command passes over identical WAV paths."""

    samples: list[BenchmarkSample] = []
    for pass_name in PASS_NAMES:
        for recording in recordings:
            resolved_path = recording.audio.path.resolve()
            for model, provider in providers.items():
                outcome = provider.benchmark_transcribe(resolved_path)
                result = outcome.transcription
                transcript = result.text if result.success else ""
                samples.append(
                    BenchmarkSample(
                        phrase=recording.phrase,
                        model=model,
                        pass_name=pass_name,
                        audio_path=resolved_path,
                        transcript=transcript,
                        audio_seconds=(
                            result.audio_duration_seconds
                            if result.audio_duration_seconds is not None
                            else recording.audio.duration_seconds
                        ),
                        transcription_seconds=result.elapsed_seconds,
                        match=(
                            compare_phrase(recording.phrase.expected, transcript)
                            if result.success
                            else MatchKind.MISS
                        ),
                        provider_success=result.success,
                        timings=outcome.timings,
                        error=result.error.message if result.error is not None else None,
                    )
                )
    return BenchmarkReport(tuple(samples))


def summarize_model(
    report: BenchmarkReport,
    model: str,
    *,
    pass_name: str = PASS_NAMES[1],
) -> ModelSummary:
    samples = report.samples_for(model, pass_name)
    if not samples:
        raise ValueError(f"No benchmark samples for model '{model}' and {pass_name}.")
    durations = [sample.transcription_seconds for sample in samples]
    successes = [sample for sample in samples if sample.match is not MatchKind.MISS]
    english = [sample for sample in samples if sample.phrase.language == "en"]
    danish = [sample for sample in samples if sample.phrase.language == "da"]
    return ModelSummary(
        model=model,
        pass_name=pass_name,
        sample_count=len(samples),
        mean_seconds=statistics.fmean(durations),
        median_seconds=statistics.median(durations),
        fastest_seconds=min(durations),
        slowest_seconds=max(durations),
        recognition_successes=len(successes),
        english_successes=sum(sample.match is not MatchKind.MISS for sample in english),
        english_total=len(english),
        danish_successes=sum(sample.match is not MatchKind.MISS for sample in danish),
        danish_total=len(danish),
    )


def cleanup_recordings(recordings: Sequence[BenchmarkRecording]) -> tuple[str, ...]:
    """Delete only the explicitly captured benchmark files."""

    warnings: list[str] = []
    for recording in recordings:
        try:
            recording.audio.path.unlink(missing_ok=True)
        except OSError as exc:
            warnings.append(f"Could not delete {recording.audio.path}: {exc}")
    return tuple(warnings)


def _display(value: str, width: int) -> str:
    compact = " ".join(value.split())
    if len(compact) > width:
        compact = compact[: width - 1] + "…"
    return compact


def format_benchmark_report(report: BenchmarkReport) -> str:
    """Render per-phrase comparisons, warm summaries, and process timings."""

    lines = [
        "STT benchmark results",
        "",
        "Lang | Expected                       | Model | Recognized                     | Audio | Pass 1 | Pass 2 | RTF  | Match",
        "-----+--------------------------------+-------+--------------------------------+-------+--------+--------+------+-----------",
    ]
    for model in ("base", "small"):
        first = report.samples_for(model, PASS_NAMES[0])
        warm = report.samples_for(model, PASS_NAMES[1])
        by_path = {sample.audio_path: sample for sample in first}
        for sample in warm:
            cold = by_path[sample.audio_path]
            rtf = sample.real_time_factor
            recognized = sample.transcript or f"ERROR: {sample.error or 'unknown'}"
            lines.append(
                f"{sample.phrase.language:4} | "
                f"{_display(sample.phrase.expected, 30):30} | "
                f"{model:5} | "
                f"{_display(recognized, 30):30} | "
                f"{sample.audio_seconds:5.2f} | {cold.transcription_seconds:6.2f} | "
                f"{sample.transcription_seconds:6.2f} | "
                f"{rtf:4.2f} | {sample.match.value}"
            )

    lines.extend(("", f"Warm-cache summary ({PASS_NAMES[1]})"))
    for model in ("base", "small"):
        summary = summarize_model(report, model)
        lines.append(
            f"{model:5}: median {summary.median_seconds:.2f}s, mean {summary.mean_seconds:.2f}s, "
            f"fastest {summary.fastest_seconds:.2f}s, slowest {summary.slowest_seconds:.2f}s, "
            f"matches {summary.recognition_successes}/{summary.sample_count} "
            f"(English {summary.english_successes}/{summary.english_total}, "
            f"Danish {summary.danish_successes}/{summary.danish_total})"
        )

    lines.extend(
        (
            "",
            "Process diagnostics (first phrase in each pass)",
            "The process-per-command design starts a new whisper-cli process and reloads the model. "
            "Pass 2 is an OS-file-cache repeat, not a persistent warm model.",
        )
    )
    for model in ("base", "small"):
        for pass_name in PASS_NAMES:
            sample = report.samples_for(model, pass_name)[0]
            timings = sample.timings
            if timings is None:
                lines.append(f"{model:5} {pass_name}: whisper.cpp timings unavailable")
                continue
            reported = timings.total_seconds
            overhead = (
                max(0.0, sample.transcription_seconds - reported)
                if reported is not None
                else None
            )
            lines.append(
                f"{model:5} {pass_name}: wall={sample.transcription_seconds:.2f}s, "
                f"load={_seconds(timings.load_seconds)}, encode={_seconds(timings.encode_seconds)}, "
                f"decode={_seconds(timings.decode_seconds)}, total={_seconds(reported)}, "
                f"outside-reported-total={_seconds(overhead)}"
            )
    return "\n".join(lines)


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"
