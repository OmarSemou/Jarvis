"""Lazy ONNX adapter for the local OpenWakeWord ``hey Jarvis`` model."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import (
    WakeWordDetection,
    WakeWordErrorCode,
    WakeWordFailure,
)


OPENWAKEWORD_VERSION = "0.6.0"
WAKE_MODEL_NAME = "hey_jarvis_v0.1"
WAKE_PHRASE = "Hey Jarvis"
WAKE_SETUP_COMMAND = (
    "powershell -ExecutionPolicy Bypass -File scripts/setup_voice_windows.ps1"
)

PackageProbe = Callable[[str], bool]
EngineLoader = Callable[[Path, Path, Path], Any]


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_engine(model: Path, melspectrogram: Path, embedding: Path) -> Any:
    module = importlib.import_module("openwakeword.model")
    return module.Model(
        wakeword_models=[str(model)],
        inference_framework="onnx",
        melspec_model_path=str(melspectrogram),
        embedding_model_path=str(embedding),
    )


@dataclass(frozen=True, slots=True)
class OpenWakeWordSettings:
    model_path: Path
    melspectrogram_path: Path
    embedding_path: Path
    threshold: float = 0.5

    def __post_init__(self) -> None:
        for path in (
            self.model_path,
            self.melspectrogram_path,
            self.embedding_path,
        ):
            if not path.is_absolute():
                raise ValueError("wake-word model paths must be absolute")
        if not 0.05 <= self.threshold <= 0.99:
            raise ValueError("wake-word threshold must be from 0.05 to 0.99")


class OpenWakeWord:
    """Translate OpenWakeWord predictions into Jarvis-owned detections."""

    name = "openwakeword"
    phrase = WAKE_PHRASE

    def __init__(
        self,
        settings: OpenWakeWordSettings,
        *,
        engine_loader: EngineLoader = _load_engine,
        package_probe: PackageProbe = _package_available,
    ) -> None:
        self.settings = settings
        self._engine_loader = engine_loader
        self._package_probe = package_probe
        self._engine: Any | None = None

    @property
    def threshold(self) -> float:
        """Expose the configured threshold without leaking provider internals."""

        return self.settings.threshold

    def readiness_error(self) -> WakeWordFailure | None:
        if not self._package_probe("openwakeword"):
            return WakeWordFailure(
                WakeWordErrorCode.PROVIDER_UNAVAILABLE,
                f"openwakeword {OPENWAKEWORD_VERSION} is not installed. Run: {WAKE_SETUP_COMMAND}",
            )
        if not self.settings.model_path.is_file():
            return WakeWordFailure(
                WakeWordErrorCode.MODEL_MISSING,
                f"The pinned {WAKE_MODEL_NAME} classifier is missing. Run: {WAKE_SETUP_COMMAND}",
            )
        missing = [
            path.name
            for path in (
                self.settings.melspectrogram_path,
                self.settings.embedding_path,
            )
            if not path.is_file()
        ]
        if missing:
            return WakeWordFailure(
                WakeWordErrorCode.FEATURE_MODEL_MISSING,
                f"Wake-word feature model(s) missing: {', '.join(missing)}. Run: {WAKE_SETUP_COMMAND}",
            )
        return None

    def _load(self) -> WakeWordFailure | None:
        readiness = self.readiness_error()
        if readiness is not None:
            return readiness
        if self._engine is None:
            try:
                self._engine = self._engine_loader(
                    self.settings.model_path,
                    self.settings.melspectrogram_path,
                    self.settings.embedding_path,
                )
            except Exception as exc:
                self._engine = None
                return WakeWordFailure(
                    WakeWordErrorCode.MODEL_LOAD_FAILED,
                    f"Could not load the local wake-word models: {exc}",
                )
        return None

    def warmup(self) -> WakeWordFailure | None:
        failure = self._load()
        if failure is not None:
            return failure
        result = self.process(b"\x00\x00" * 1_280)
        self.reset()
        return result.error

    def process(self, pcm16: bytes) -> WakeWordDetection:
        if not pcm16 or len(pcm16) % 2:
            return WakeWordDetection(
                False,
                0.0,
                self.name,
                self.phrase,
                WakeWordFailure(
                    WakeWordErrorCode.INVALID_AUDIO,
                    "Wake-word input must be non-empty PCM16 audio.",
                ),
            )
        failure = self._load()
        if failure is not None:
            return WakeWordDetection(False, 0.0, self.name, self.phrase, failure)
        try:
            numpy = importlib.import_module("numpy")
            samples = numpy.frombuffer(pcm16, dtype="<i2")
            predictions: Mapping[str, Any] = self._engine.predict(samples)
            scores = [float(value) for value in predictions.values()]
            score = max(scores, default=0.0)
            score = max(0.0, min(1.0, score))
            return WakeWordDetection(
                score >= self.settings.threshold,
                score,
                self.name,
                self.phrase,
            )
        except Exception as exc:
            return WakeWordDetection(
                False,
                0.0,
                self.name,
                self.phrase,
                WakeWordFailure(
                    WakeWordErrorCode.INFERENCE_FAILED,
                    f"Local wake-word inference failed: {exc}",
                ),
            )

    def reset(self) -> None:
        if self._engine is not None:
            try:
                self._engine.reset()
            except Exception:
                self._engine = None
