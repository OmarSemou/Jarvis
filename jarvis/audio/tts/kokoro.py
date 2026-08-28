"""Lazy CPU adapter for the local kokoro-onnx package."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .base import (
    SpeechSynthesisResult,
    SynthesizedAudio,
    SynthesisErrorCode,
    SynthesisFailure,
)


KOKORO_ONNX_VERSION = "0.6.1"
KOKORO_MODEL_RELEASE = "model-files-v1.0"
KOKORO_VOICES = ("am_fenrir", "am_michael", "am_puck", "bm_george")
KOKORO_SETUP_COMMAND = (
    "powershell -ExecutionPolicy Bypass -File scripts/setup_tts_windows.ps1 "
    "-Providers kokoro"
)

EngineLoader = Callable[[Path, Path], Any]
PackageProbe = Callable[[str], bool]


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_engine(model_path: Path, voices_path: Path) -> Any:
    module = importlib.import_module("kokoro_onnx")
    return module.Kokoro(str(model_path), str(voices_path))


def _pcm16_bytes(samples: Any) -> bytes:
    """Convert a NumPy-like float array to deterministic little-endian PCM16."""

    if hasattr(samples, "clip") and hasattr(samples, "astype"):
        flattened = samples.reshape(-1) if hasattr(samples, "reshape") else samples
        return (flattened.clip(-1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    values: Iterable[float] = samples
    output = bytearray()
    for value in values:
        sample = max(-1.0, min(1.0, float(value)))
        integer = max(-32768, min(32767, round(sample * 32767.0)))
        output.extend(integer.to_bytes(2, "little", signed=True))
    return bytes(output)


@dataclass(frozen=True, slots=True)
class KokoroSettings:
    model_path: Path
    voices_path: Path

    def __post_init__(self) -> None:
        if not self.model_path.is_absolute() or not self.voices_path.is_absolute():
            raise ValueError("Kokoro model and voices paths must be absolute")


class KokoroTTS:
    name = "kokoro"
    available_voices = KOKORO_VOICES

    def __init__(
        self,
        settings: KokoroSettings,
        *,
        engine_loader: EngineLoader = _load_engine,
        package_probe: PackageProbe = _package_available,
    ) -> None:
        self.settings = settings
        self._engine_loader = engine_loader
        self._package_probe = package_probe
        self._engine: Any | None = None

    def readiness_error(self, voice: str) -> SynthesisFailure | None:
        if voice not in self.available_voices:
            return SynthesisFailure(
                SynthesisErrorCode.INVALID_VOICE,
                f"Unknown Kokoro voice '{voice}'. Allowed: {', '.join(self.available_voices)}.",
            )
        if not self._package_probe("kokoro_onnx"):
            return SynthesisFailure(
                SynthesisErrorCode.PROVIDER_UNAVAILABLE,
                f"kokoro-onnx {KOKORO_ONNX_VERSION} is not installed.\nRun:\n{KOKORO_SETUP_COMMAND}",
            )
        if not self.settings.model_path.is_file():
            return SynthesisFailure(
                SynthesisErrorCode.MODEL_MISSING,
                f"The local Kokoro model is missing.\nRun:\n{KOKORO_SETUP_COMMAND}",
            )
        if not self.settings.voices_path.is_file():
            return SynthesisFailure(
                SynthesisErrorCode.VOICE_MISSING,
                f"The local Kokoro voice bundle is missing.\nRun:\n{KOKORO_SETUP_COMMAND}",
            )
        return None

    def _result(
        self,
        started: float,
        voice: str,
        *,
        audio: SynthesizedAudio | None = None,
        error: SynthesisFailure | None = None,
        first_audio_seconds: float | None = None,
    ) -> SpeechSynthesisResult:
        elapsed = perf_counter() - started
        return SpeechSynthesisResult(
            success=error is None,
            provider=self.name,
            voice=voice,
            elapsed_seconds=elapsed,
            audio=audio,
            first_audio_seconds=first_audio_seconds,
            error=error,
        )

    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed: float = 1.0,
        language: str = "en",
    ) -> SpeechSynthesisResult:
        started = perf_counter()
        if not text.strip():
            return self._result(
                started,
                voice,
                error=SynthesisFailure(SynthesisErrorCode.INVALID_TEXT, "Speech text is empty."),
            )
        if not 0.5 <= speed <= 2.0:
            return self._result(
                started,
                voice,
                error=SynthesisFailure(
                    SynthesisErrorCode.INVALID_SPEED,
                    "Speech speed must be from 0.5 to 2.0.",
                ),
            )
        if language != "en":
            return self._result(
                started,
                voice,
                error=SynthesisFailure(
                    SynthesisErrorCode.UNSUPPORTED_LANGUAGE,
                    "Kokoro's Phase 2C2 adapter supports English only.",
                ),
            )
        readiness = self.readiness_error(voice)
        if readiness is not None:
            return self._result(started, voice, error=readiness)
        try:
            if self._engine is None:
                self._engine = self._engine_loader(
                    self.settings.model_path,
                    self.settings.voices_path,
                )
            language_code = "en-gb" if voice.startswith("b") else "en-us"
            samples, sample_rate = self._engine.create(
                text.strip(),
                voice=voice,
                speed=speed,
                lang=language_code,
            )
            pcm16 = _pcm16_bytes(samples)
            if not pcm16:
                return self._result(
                    started,
                    voice,
                    error=SynthesisFailure(
                        SynthesisErrorCode.EMPTY_AUDIO,
                        "Kokoro produced no audio.",
                    ),
                )
            audio = SynthesizedAudio(pcm16, int(sample_rate), 1)
            first_audio = perf_counter() - started
            return self._result(
                started,
                voice,
                audio=audio,
                first_audio_seconds=first_audio,
            )
        except (ImportError, OSError) as exc:
            self._engine = None
            return self._result(
                started,
                voice,
                error=SynthesisFailure(
                    SynthesisErrorCode.MODEL_LOAD_FAILED,
                    f"Kokoro could not load its local model: {exc}",
                ),
            )
        except Exception as exc:
            return self._result(
                started,
                voice,
                error=SynthesisFailure(
                    SynthesisErrorCode.SYNTHESIS_FAILED,
                    f"Kokoro synthesis failed: {exc}",
                ),
            )
