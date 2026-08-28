"""Safe external-process adapter for a configured local whisper.cpp binary."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from jarvis.audio.formats import AudioFormatError, inspect_wav

from .base import (
    TranscriptionErrorCode,
    TranscriptionFailure,
    TranscriptionResult,
)


WHISPER_SETUP_COMMAND = (
    "powershell -ExecutionPolicy Bypass -File scripts/setup_whisper_windows.ps1"
)
WHISPER_CPP_VERSION = "1.9.1"
SUPPORTED_MODELS = frozenset({"base", "small"})
SUPPORTED_LANGUAGES = frozenset({"auto", "en", "da"})
NO_SPEECH_MARKERS = frozenset(
    {
        "[ silence ]",
        "[ blank_audio ]",
        "[ no speech ]",
        "( silence )",
    }
)
ProcessRunner = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class WhisperCppSettings:
    executable_path: Path
    model_path: Path
    temp_dir: Path
    model_name: str = "small"
    language: str = "auto"
    timeout_seconds: float = 180.0
    use_gpu: bool = False

    def __post_init__(self) -> None:
        for label, path in (
            ("executable_path", self.executable_path),
            ("model_path", self.model_path),
            ("temp_dir", self.temp_dir),
        ):
            if not path.is_absolute():
                raise ValueError(f"{label} must be an absolute path")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError("language must be one of: auto, en, da")
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError("model_name must be one of: base, small")
        if not 0 < self.timeout_seconds <= 600:
            raise ValueError("timeout_seconds must be greater than 0 and at most 600")


@dataclass(frozen=True, slots=True)
class WhisperTimings:
    """Timing fields explicitly reported by whisper.cpp, in seconds."""

    load_seconds: float | None = None
    mel_seconds: float | None = None
    sample_seconds: float | None = None
    encode_seconds: float | None = None
    decode_seconds: float | None = None
    batchd_seconds: float | None = None
    prompt_seconds: float | None = None
    total_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class WhisperBenchmarkResult:
    transcription: TranscriptionResult
    timings: WhisperTimings | None = None


def parse_whisper_timings(stderr: str) -> WhisperTimings | None:
    """Parse only whisper.cpp's documented ``whisper_print_timings`` lines."""

    supported = {
        "load time": "load_seconds",
        "mel time": "mel_seconds",
        "sample time": "sample_seconds",
        "encode time": "encode_seconds",
        "decode time": "decode_seconds",
        "batchd time": "batchd_seconds",
        "prompt time": "prompt_seconds",
        "total time": "total_seconds",
    }
    values: dict[str, float] = {}
    prefix = "whisper_print_timings:"
    for raw_line in stderr.splitlines():
        line = raw_line.strip()
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix) :].strip()
        if "=" not in payload:
            continue
        label, raw_value = (part.strip() for part in payload.split("=", 1))
        field = supported.get(label)
        parts = raw_value.split()
        if field is None or len(parts) < 2 or parts[1].casefold() != "ms":
            continue
        try:
            values[field] = float(parts[0]) / 1000.0
        except ValueError:
            continue
    return WhisperTimings(**values) if values else None


class WhisperCppSTT:
    """Transcribe WAV audio through a pinned/configured local executable."""

    name = "whisper.cpp"

    def __init__(
        self,
        settings: WhisperCppSettings,
        *,
        process_runner: ProcessRunner = subprocess.run,
    ) -> None:
        self.settings = settings
        self._process_runner = process_runner

    def readiness_error(self) -> TranscriptionFailure | None:
        if not self.settings.executable_path.is_file():
            return TranscriptionFailure(
                TranscriptionErrorCode.EXECUTABLE_MISSING,
                "Whisper is not installed.\nRun:\n" + WHISPER_SETUP_COMMAND,
            )
        if not self.settings.model_path.is_file():
            return TranscriptionFailure(
                TranscriptionErrorCode.MODEL_MISSING,
                f"The multilingual Whisper {self.settings.model_name} model is missing.\nRun:\n"
                f"{WHISPER_SETUP_COMMAND} -Models {self.settings.model_name}",
            )
        return None

    def build_command(
        self,
        audio_path: Path,
        output_prefix: Path,
        *,
        include_timings: bool = False,
    ) -> tuple[str, ...]:
        command = [
            str(self.settings.executable_path),
            "--model",
            str(self.settings.model_path),
            "--file",
            str(audio_path),
            "--language",
            self.settings.language,
            "--output-txt",
            "--output-file",
            str(output_prefix),
            "--no-timestamps",
        ]
        if not include_timings:
            command.append("--no-prints")
        if not self.settings.use_gpu:
            command.append("--no-gpu")
        return tuple(command)

    def _result(
        self,
        *,
        started: float,
        duration: float | None,
        text: str = "",
        error: TranscriptionFailure | None = None,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            success=error is None,
            text=text,
            provider=self.name,
            elapsed_seconds=perf_counter() - started,
            audio_duration_seconds=duration,
            error=error,
        )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe normally, suppressing whisper.cpp diagnostic output."""

        return self._transcribe(audio_path, include_timings=False).transcription

    def benchmark_transcribe(self, audio_path: Path) -> WhisperBenchmarkResult:
        """Transcribe once and collect whisper.cpp's process timing diagnostics."""

        return self._transcribe(audio_path, include_timings=True)

    def _transcribe(
        self,
        audio_path: Path,
        *,
        include_timings: bool,
    ) -> WhisperBenchmarkResult:
        started = perf_counter()
        ready_error = self.readiness_error()
        if ready_error is not None:
            return WhisperBenchmarkResult(
                self._result(started=started, duration=None, error=ready_error)
            )

        resolved_audio = audio_path.expanduser().resolve()
        if not resolved_audio.is_file():
            return WhisperBenchmarkResult(
                self._result(
                    started=started,
                    duration=None,
                    error=TranscriptionFailure(
                        TranscriptionErrorCode.AUDIO_MISSING,
                        f"Audio recording not found: {resolved_audio}",
                    ),
                )
            )
        try:
            wav_info = inspect_wav(resolved_audio)
        except AudioFormatError as exc:
            return WhisperBenchmarkResult(
                self._result(
                    started=started,
                    duration=None,
                    error=TranscriptionFailure(TranscriptionErrorCode.INVALID_AUDIO, str(exc)),
                )
            )
        duration = wav_info.duration_seconds
        if wav_info.channels != 1 or wav_info.sample_width != 2 or wav_info.sample_rate != 16_000:
            return WhisperBenchmarkResult(
                self._result(
                    started=started,
                    duration=duration,
                    error=TranscriptionFailure(
                        TranscriptionErrorCode.INVALID_AUDIO,
                        "Whisper input must be mono PCM16 WAV at 16000 Hz.",
                    ),
                )
            )

        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="whisper-", dir=self.settings.temp_dir)).resolve()
        output_prefix = work_dir / "transcript"
        output_file = output_prefix.with_suffix(".txt")
        command: Sequence[str] = self.build_command(
            resolved_audio,
            output_prefix,
            include_timings=include_timings,
        )
        try:
            try:
                completed = self._process_runner(
                    command,
                    shell=False,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.settings.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return WhisperBenchmarkResult(
                    self._result(
                        started=started,
                        duration=duration,
                        error=TranscriptionFailure(
                            TranscriptionErrorCode.TIMEOUT,
                            f"Whisper transcription timed out after {self.settings.timeout_seconds:g} seconds.",
                        ),
                    )
                )
            except OSError as exc:
                return WhisperBenchmarkResult(
                    self._result(
                        started=started,
                        duration=duration,
                        error=TranscriptionFailure(
                            TranscriptionErrorCode.PROCESS_FAILED,
                            f"Whisper could not start: {exc}",
                        ),
                    )
                )

            if completed.returncode != 0:
                detail = (completed.stderr or "").strip()
                message = f"Whisper exited with code {completed.returncode}."
                if detail:
                    message += f" {detail[:500]}"
                return WhisperBenchmarkResult(
                    self._result(
                        started=started,
                        duration=duration,
                        error=TranscriptionFailure(TranscriptionErrorCode.PROCESS_FAILED, message),
                    )
                )
            if not output_file.is_file():
                return WhisperBenchmarkResult(
                    self._result(
                        started=started,
                        duration=duration,
                        error=TranscriptionFailure(
                            TranscriptionErrorCode.OUTPUT_MISSING,
                            "Whisper completed without creating its configured text output file.",
                        ),
                    )
                )
            try:
                transcript = output_file.read_text(encoding="utf-8-sig").strip()
            except OSError as exc:
                return WhisperBenchmarkResult(
                    self._result(
                        started=started,
                        duration=duration,
                        error=TranscriptionFailure(
                            TranscriptionErrorCode.OUTPUT_MISSING,
                            f"Could not read Whisper transcription output: {exc}",
                        ),
                    )
                )
            normalized = " ".join(transcript.casefold().split())
            if not transcript or normalized in NO_SPEECH_MARKERS:
                return WhisperBenchmarkResult(
                    self._result(
                        started=started,
                        duration=duration,
                        error=TranscriptionFailure(
                            TranscriptionErrorCode.EMPTY_TRANSCRIPT,
                            "Whisper did not detect any speech.",
                        ),
                    )
                )
            timings = parse_whisper_timings(completed.stderr or "") if include_timings else None
            return WhisperBenchmarkResult(
                self._result(started=started, duration=duration, text=transcript),
                timings,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
