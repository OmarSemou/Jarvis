"""Repository and runtime path definitions.

Paths are derived from an explicit repository root.  No path in this module
depends on the process working directory, and importing it creates no files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class JarvisPaths:
    """All paths used by the Phase 1 foundation and legacy launcher."""

    repository_root: Path

    @classmethod
    def from_repository_root(cls, root: str | Path) -> "JarvisPaths":
        """Build paths from an explicit root without consulting ``cwd``."""

        root_path = Path(root).expanduser()
        if not root_path.is_absolute():
            raise ValueError("repository root must be an absolute path")
        return cls(root_path.resolve())

    @classmethod
    def discover(cls, anchor: str | Path | None = None) -> "JarvisPaths":
        """Find the repository from a code location, never from ``cwd``.

        The optional anchor exists for tests and embedded launchers.  Normal
        calls use this module's location and walk upward until both ``LICENSE``
        and ``agent.py`` are found.
        """

        if anchor is None:
            start = Path(__file__).resolve()
        else:
            anchor_path = Path(anchor).expanduser()
            if not anchor_path.is_absolute():
                raise ValueError("discovery anchor must be an absolute path")
            start = anchor_path.resolve()
        start = start.parent if start.is_file() else start
        for candidate in (start, *start.parents):
            if (candidate / "LICENSE").is_file() and (candidate / "agent.py").is_file():
                return cls.from_repository_root(candidate)
        raise RuntimeError(f"Could not locate the Jarvis repository above {start}")

    @property
    def legacy_config_file(self) -> Path:
        return self.repository_root / "config.json"

    @property
    def example_config_file(self) -> Path:
        return self.repository_root / "config.example.json"

    @property
    def data_dir(self) -> Path:
        return self.repository_root / "data"

    @property
    def user_config_file(self) -> Path:
        return self.data_dir / "config.json"

    @property
    def active_config_file(self) -> Path:
        """Prefer private runtime configuration, with legacy fallback."""

        return self.user_config_file if self.user_config_file.is_file() else self.legacy_config_file

    @property
    def memory_file(self) -> Path:
        return self.data_dir / "memory.json"

    @property
    def input_audio_file(self) -> Path:
        return self.data_dir / "input.wav"

    @property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"

    @property
    def stt_temp_dir(self) -> Path:
        return self.data_dir / "stt"

    @property
    def local_tools_dir(self) -> Path:
        return self.data_dir / "tools"

    @property
    def local_models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def tts_models_dir(self) -> Path:
        return self.local_models_dir / "tts"

    @property
    def kokoro_model(self) -> Path:
        return self.tts_models_dir / "kokoro" / "kokoro-v1.0.onnx"

    @property
    def kokoro_voices(self) -> Path:
        return self.tts_models_dir / "kokoro" / "voices-v1.0.bin"

    def piper_voice_files(self, voice: str) -> tuple[Path, Path]:
        supported = {"en_US-joe-medium", "en_US-john-medium"}
        if voice not in supported:
            raise ValueError("Piper voice must be one of: en_US-joe-medium, en_US-john-medium")
        model = self.tts_models_dir / "piper" / f"{voice}.onnx"
        return model, model.with_suffix(".onnx.json")

    @property
    def tts_benchmark_dir(self) -> Path:
        return self.data_dir / "benchmarks" / "tts"

    @property
    def wakeword_models_dir(self) -> Path:
        return self.local_models_dir / "wakeword"

    @property
    def wakeword_classifier_model(self) -> Path:
        """Return the pinned OpenWakeWord primary Hey Jarvis classifier."""

        return self.wakeword_models_dir / "hey_jarvis_v0.1.onnx"

    @property
    def wakeword_melspectrogram_model(self) -> Path:
        return self.wakeword_models_dir / "melspectrogram.onnx"

    @property
    def wakeword_embedding_model(self) -> Path:
        return self.wakeword_models_dir / "embedding_model.onnx"

    @property
    def vad_model(self) -> Path:
        return self.wakeword_models_dir / "silero_vad.onnx"

    @property
    def camera_image_file(self) -> Path:
        return self.data_dir / "current_image.jpg"

    @property
    def faces_dir(self) -> Path:
        return self.repository_root / "faces"

    @property
    def sounds_dir(self) -> Path:
        return self.repository_root / "sounds"

    @property
    def wakeword_model(self) -> Path:
        """Return the legacy upstream classifier retained for compatibility."""

        return self.repository_root / "wakeword.onnx"

    @property
    def whisper_model(self) -> Path:
        """Return the legacy compatibility launcher's small-model path."""

        return self.local_models_dir / "whisper" / "ggml-small.bin"

    def whisper_model_for(self, model: str) -> Path:
        """Resolve one allowlisted multilingual whisper.cpp model name."""

        normalized = model.strip().casefold()
        if normalized not in {"base", "small"}:
            raise ValueError("Whisper model must be one of: base, small")
        return self.local_models_dir / "whisper" / f"ggml-{normalized}.bin"

    @property
    def whisper_executable_candidates(self) -> tuple[Path, ...]:
        installed = self.local_tools_dir / "whisper.cpp" / "v1.9.1" / "Release"
        legacy = self.repository_root / "whisper.cpp" / "build" / "bin"
        return (
            installed / "whisper-cli.exe",
            installed / "whisper-cli",
            legacy / "whisper-cli.exe",
            legacy / "whisper-cli",
        )

    @property
    def piper_executable_candidates(self) -> tuple[Path, ...]:
        base = self.repository_root / "piper"
        return (base / "piper.exe", base / "piper")

    def resolve_from_root(self, value: str | Path) -> Path:
        """Resolve a configured resource path relative to the repository."""

        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.repository_root / path).resolve()

    def ensure_runtime_directories(self) -> None:
        """Create private runtime directories when the application needs them.

        This operation is deliberately explicit so importing ``jarvis.core``
        cannot write to disk.
        """

        self.data_dir.mkdir(parents=True, exist_ok=True)


def first_existing(candidates: tuple[Path, ...]) -> Path | None:
    """Return the first existing candidate without executing it."""

    return next((candidate for candidate in candidates if candidate.is_file()), None)
