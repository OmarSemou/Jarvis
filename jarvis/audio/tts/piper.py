"""Lazy CPU adapter for the maintained Open Home Foundation Piper package."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Mapping
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


PIPER_VERSION = "1.7.0"
PIPER_VOICE_RELEASE = "v1.0.0"
PIPER_VOICES = ("en_US-joe-medium", "en_US-john-medium")
PIPER_SETUP_COMMAND = (
    "powershell -ExecutionPolicy Bypass -File scripts/setup_tts_windows.ps1 "
    "-Providers piper"
)

EngineLoader = Callable[[Path, Path], Any]
SynthesisConfigFactory = Callable[..., Any]
PackageProbe = Callable[[str], bool]


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_engine(model_path: Path, config_path: Path) -> Any:
    module = importlib.import_module("piper")
    return module.PiperVoice.load(
        model_path,
        config_path=config_path,
        use_cuda=False,
    )


def _synthesis_config(**values: Any) -> Any:
    module = importlib.import_module("piper.config")
    return module.SynthesisConfig(**values)


@dataclass(frozen=True, slots=True)
class PiperSettings:
    voice_files: Mapping[str, tuple[Path, Path]]

    def __post_init__(self) -> None:
        if set(self.voice_files) != set(PIPER_VOICES):
            raise ValueError("Piper settings must define only the curated Phase 2C2 voices")
        for model_path, config_path in self.voice_files.values():
            if not model_path.is_absolute() or not config_path.is_absolute():
                raise ValueError("Piper model and config paths must be absolute")


class PiperTTS:
    name = "piper"
    available_voices = PIPER_VOICES

    def __init__(
        self,
        settings: PiperSettings,
        *,
        engine_loader: EngineLoader = _load_engine,
        synthesis_config_factory: SynthesisConfigFactory = _synthesis_config,
        package_probe: PackageProbe = _package_available,
    ) -> None:
        self.settings = settings
        self._engine_loader = engine_loader
        self._synthesis_config_factory = synthesis_config_factory
        self._package_probe = package_probe
        self._engines: dict[str, Any] = {}

    def readiness_error(self, voice: str) -> SynthesisFailure | None:
        if voice not in self.available_voices:
            return SynthesisFailure(
                SynthesisErrorCode.INVALID_VOICE,
                f"Unknown Piper voice '{voice}'. Allowed: {', '.join(self.available_voices)}.",
            )
        if not self._package_probe("piper"):
            return SynthesisFailure(
                SynthesisErrorCode.PROVIDER_UNAVAILABLE,
                f"piper-tts {PIPER_VERSION} is not installed.\nRun:\n{PIPER_SETUP_COMMAND}",
            )
        model_path, config_path = self.settings.voice_files[voice]
        if not model_path.is_file():
            return SynthesisFailure(
                SynthesisErrorCode.MODEL_MISSING,
                f"The local Piper voice model '{voice}' is missing.\nRun:\n{PIPER_SETUP_COMMAND}",
            )
        if not config_path.is_file():
            return SynthesisFailure(
                SynthesisErrorCode.VOICE_MISSING,
                f"The local Piper voice config '{voice}' is missing.\nRun:\n{PIPER_SETUP_COMMAND}",
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
        return SpeechSynthesisResult(
            success=error is None,
            provider=self.name,
            voice=voice,
            elapsed_seconds=perf_counter() - started,
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
                    "The installed Phase 2C2 Piper voices support English only.",
                ),
            )
        readiness = self.readiness_error(voice)
        if readiness is not None:
            return self._result(started, voice, error=readiness)
        model_path, config_path = self.settings.voice_files[voice]
        try:
            engine = self._engines.get(voice)
            if engine is None:
                engine = self._engine_loader(model_path, config_path)
                self._engines[voice] = engine
            syn_config = self._synthesis_config_factory(length_scale=1.0 / speed)
            chunks = engine.synthesize(text.strip(), syn_config=syn_config)
            pcm_parts: list[bytes] = []
            sample_rate: int | None = None
            channels: int | None = None
            first_audio: float | None = None
            for chunk in chunks:
                if first_audio is None:
                    first_audio = perf_counter() - started
                chunk_rate = int(chunk.sample_rate)
                chunk_channels = int(chunk.sample_channels)
                if int(chunk.sample_width) != 2:
                    raise ValueError("Piper produced non-PCM16 audio")
                if sample_rate is None:
                    sample_rate = chunk_rate
                    channels = chunk_channels
                elif sample_rate != chunk_rate or channels != chunk_channels:
                    raise ValueError("Piper changed audio format between chunks")
                pcm_parts.append(bytes(chunk.audio_int16_bytes))
            pcm16 = b"".join(pcm_parts)
            if not pcm16 or sample_rate is None or channels is None:
                return self._result(
                    started,
                    voice,
                    error=SynthesisFailure(
                        SynthesisErrorCode.EMPTY_AUDIO,
                        "Piper produced no audio.",
                    ),
                )
            return self._result(
                started,
                voice,
                audio=SynthesizedAudio(pcm16, sample_rate, channels),
                first_audio_seconds=first_audio,
            )
        except (ImportError, OSError) as exc:
            self._engines.pop(voice, None)
            return self._result(
                started,
                voice,
                error=SynthesisFailure(
                    SynthesisErrorCode.MODEL_LOAD_FAILED,
                    f"Piper could not load voice '{voice}': {exc}",
                ),
            )
        except Exception as exc:
            return self._result(
                started,
                voice,
                error=SynthesisFailure(
                    SynthesisErrorCode.SYNTHESIS_FAILED,
                    f"Piper synthesis failed: {exc}",
                ),
            )
