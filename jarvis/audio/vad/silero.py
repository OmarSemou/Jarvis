"""Lazy ONNX-only Silero VAD adapter using OpenWakeWord's lightweight wrapper."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import VADErrorCode, VADFailure


SILERO_VAD_ASSET_RELEASE = "openWakeWord-v0.5.1"
VAD_SETUP_COMMAND = (
    "powershell -ExecutionPolicy Bypass -File scripts/setup_voice_windows.ps1"
)

PackageProbe = Callable[[str], bool]
EngineLoader = Callable[[Path], Any]


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_engine(model_path: Path) -> Any:
    module = importlib.import_module("openwakeword.vad")
    return module.VAD(model_path=str(model_path), n_threads=1)


@dataclass(frozen=True, slots=True)
class SileroVADSettings:
    model_path: Path

    def __post_init__(self) -> None:
        if not self.model_path.is_absolute():
            raise ValueError("Silero VAD model path must be absolute")


class SileroVAD:
    name = "silero-onnx"

    def __init__(
        self,
        settings: SileroVADSettings,
        *,
        engine_loader: EngineLoader = _load_engine,
        package_probe: PackageProbe = _package_available,
    ) -> None:
        self.settings = settings
        self._engine_loader = engine_loader
        self._package_probe = package_probe
        self._engine: Any | None = None

    def readiness_error(self) -> VADFailure | None:
        if not self._package_probe("openwakeword"):
            return VADFailure(
                VADErrorCode.PROVIDER_UNAVAILABLE,
                f"The local ONNX VAD wrapper is unavailable. Run: {VAD_SETUP_COMMAND}",
            )
        if not self.settings.model_path.is_file():
            return VADFailure(
                VADErrorCode.MODEL_MISSING,
                f"The local Silero VAD model is missing. Run: {VAD_SETUP_COMMAND}",
            )
        return None

    def _load(self) -> VADFailure | None:
        readiness = self.readiness_error()
        if readiness is not None:
            return readiness
        if self._engine is None:
            try:
                self._engine = self._engine_loader(self.settings.model_path)
            except Exception as exc:
                self._engine = None
                return VADFailure(
                    VADErrorCode.MODEL_LOAD_FAILED,
                    f"Could not load the local Silero VAD model: {exc}",
                )
        return None

    def warmup(self) -> VADFailure | None:
        failure = self._load()
        if failure is not None:
            return failure
        try:
            self.score(b"\x00\x00" * 480)
            self.reset()
            return None
        except Exception as exc:
            return VADFailure(
                VADErrorCode.INFERENCE_FAILED,
                f"Silero VAD warmup failed: {exc}",
            )

    def score(self, pcm16: bytes) -> float:
        if not pcm16 or len(pcm16) % 2:
            raise ValueError("VAD input must be non-empty PCM16 audio")
        failure = self._load()
        if failure is not None:
            raise RuntimeError(failure.message)
        try:
            numpy = importlib.import_module("numpy")
            samples = numpy.frombuffer(pcm16, dtype="<i2")
            score = float(self._engine.predict(samples, frame_size=len(samples)))
            return max(0.0, min(1.0, score))
        except Exception as exc:
            raise RuntimeError(f"Local Silero VAD inference failed: {exc}") from exc

    def reset(self) -> None:
        if self._engine is not None:
            try:
                self._engine.reset_states()
                if hasattr(self._engine, "prediction_buffer"):
                    self._engine.prediction_buffer.clear()
            except Exception:
                self._engine = None
