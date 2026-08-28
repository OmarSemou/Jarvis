"""Validated Jarvis configuration with explicit legacy compatibility."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from .paths import JarvisPaths


class ConfigValidationError(ValueError):
    """Raised when configuration cannot be safely interpreted."""

    def __init__(self, source: Path | None, problems: list[str]) -> None:
        self.source = source
        self.problems = tuple(problems)
        location = str(source) if source is not None else "configuration"
        super().__init__(f"Invalid Jarvis configuration at {location}: " + "; ".join(problems))


class UnknownKeyPolicy(StrEnum):
    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"


LEGACY_KEY_MAP: dict[str, str] = {
    "model": "text_model",
    "vision": "vision_model",
    "voice": "voice_model",
    "memory": "chat_memory",
    "prompt": "system_prompt",
    "microphone": "input_device",
    "sample_rate": "input_sample_rate",
}

KNOWN_KEYS = frozenset(
    {
        "text_model",
        "vision_model",
        "voice_model",
        "chat_memory",
        "camera_rotation",
        "system_prompt",
        "system_prompt_extras",
        "input_device",
        "input_sample_rate",
    }
)


@dataclass(frozen=True, slots=True)
class JarvisConfig:
    """Validated values consumed by the legacy application and future core."""

    text_model: str = "gemma3:1b"
    vision_model: str = "moondream"
    voice_model: str = "piper/en_GB-semaine-medium.onnx"
    chat_memory: bool = True
    camera_rotation: int = 0
    system_prompt: str | None = None
    system_prompt_extras: str = ""
    input_device: int | str | None = None
    input_sample_rate: int | None = None

    def effective_system_prompt(self, fallback: str) -> str:
        """Return the configured prompt, then append optional extra guidance."""

        prompt = self.system_prompt.strip() if self.system_prompt and self.system_prompt.strip() else fallback.strip()
        extras = self.system_prompt_extras.strip()
        return f"{prompt}\n\n{extras}" if extras else prompt

    def as_legacy_dict(self) -> dict[str, Any]:
        """Provide the mapping shape expected by the compatibility launcher."""

        return {
            "text_model": self.text_model,
            "vision_model": self.vision_model,
            "voice_model": self.voice_model,
            "chat_memory": self.chat_memory,
            "camera_rotation": self.camera_rotation,
            "system_prompt": self.system_prompt,
            "system_prompt_extras": self.system_prompt_extras,
            "input_device": self.input_device,
            "input_sample_rate": self.input_sample_rate,
        }


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    config: JarvisConfig
    source: Path | None
    migrations: tuple[str, ...] = field(default_factory=tuple)
    unknown_keys: tuple[str, ...] = field(default_factory=tuple)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_string(values: Mapping[str, Any], key: str, problems: list[str]) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        problems.append(f"'{key}' must be a non-empty string")
        return ""
    return value.strip()


def _migrate_keys(raw: Mapping[str, Any], problems: list[str]) -> tuple[dict[str, Any], list[str]]:
    values = dict(raw)
    migrations: list[str] = []
    for legacy, current in LEGACY_KEY_MAP.items():
        if legacy not in values:
            continue
        if current in values:
            problems.append(f"legacy key '{legacy}' conflicts with '{current}'")
            values.pop(legacy)
            continue
        values[current] = values.pop(legacy)
        migrations.append(f"'{legacy}' was migrated to '{current}'")
    return values, migrations


def parse_config(
    raw: Mapping[str, Any],
    *,
    source: str | Path | None = None,
    unknown_key_policy: UnknownKeyPolicy = UnknownKeyPolicy.ERROR,
) -> ConfigLoadResult:
    """Validate an already-decoded configuration mapping."""

    source_path = Path(source).resolve() if source is not None else None
    problems: list[str] = []
    values, migrations = _migrate_keys(raw, problems)
    unknown = sorted(set(values) - KNOWN_KEYS)
    if unknown and unknown_key_policy is UnknownKeyPolicy.ERROR:
        problems.append("unknown key(s): " + ", ".join(unknown))
    elif unknown and unknown_key_policy is UnknownKeyPolicy.WARN:
        warnings.warn("Ignoring unknown Jarvis configuration key(s): " + ", ".join(unknown), stacklevel=2)
    for key in unknown:
        values.pop(key, None)

    defaults = JarvisConfig()
    merged = defaults.as_legacy_dict()
    merged.update(values)

    text_model = _validate_string(merged, "text_model", problems)
    vision_model = _validate_string(merged, "vision_model", problems)
    voice_model = _validate_string(merged, "voice_model", problems)

    chat_memory = merged["chat_memory"]
    if not isinstance(chat_memory, bool):
        problems.append("'chat_memory' must be a boolean")

    camera_rotation = merged["camera_rotation"]
    if not _is_int(camera_rotation) or camera_rotation not in {0, 90, 180, 270}:
        problems.append("'camera_rotation' must be one of 0, 90, 180, or 270")

    system_prompt = merged["system_prompt"]
    if system_prompt is not None and not isinstance(system_prompt, str):
        problems.append("'system_prompt' must be a string or null")

    system_prompt_extras = merged["system_prompt_extras"]
    if not isinstance(system_prompt_extras, str):
        problems.append("'system_prompt_extras' must be a string")

    input_device = merged["input_device"]
    if input_device is not None:
        if _is_int(input_device):
            if input_device < 0:
                problems.append("'input_device' integer must be non-negative")
        elif isinstance(input_device, str):
            if not input_device.strip():
                problems.append("'input_device' string must not be empty")
        else:
            problems.append("'input_device' must be an integer, string, or null")

    input_sample_rate = merged["input_sample_rate"]
    if input_sample_rate is not None and (
        not _is_int(input_sample_rate) or not 8_000 <= input_sample_rate <= 192_000
    ):
        problems.append("'input_sample_rate' must be an integer from 8000 to 192000, or null")

    if problems:
        raise ConfigValidationError(source_path, problems)

    config = JarvisConfig(
        text_model=text_model,
        vision_model=vision_model,
        voice_model=voice_model,
        chat_memory=chat_memory,
        camera_rotation=camera_rotation,
        system_prompt=system_prompt.strip() if isinstance(system_prompt, str) else None,
        system_prompt_extras=system_prompt_extras,
        input_device=input_device.strip() if isinstance(input_device, str) else input_device,
        input_sample_rate=input_sample_rate,
    )
    return ConfigLoadResult(config, source_path, tuple(migrations), tuple(unknown))


def load_config(
    path: str | Path | None,
    *,
    unknown_key_policy: UnknownKeyPolicy = UnknownKeyPolicy.ERROR,
) -> ConfigLoadResult:
    """Read and validate one JSON configuration file.

    A missing path returns defaults.  Invalid JSON and non-object roots produce
    a clear ``ConfigValidationError`` rather than silently changing behavior.
    """

    if path is None:
        return ConfigLoadResult(JarvisConfig(), None)
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return ConfigLoadResult(JarvisConfig(), None)
    try:
        with source.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigValidationError(source, [f"could not read valid JSON: {exc}"]) from exc
    if not isinstance(raw, dict):
        raise ConfigValidationError(source, ["top-level JSON value must be an object"])
    return parse_config(raw, source=source, unknown_key_policy=unknown_key_policy)


def load_for_paths(
    paths: JarvisPaths,
    *,
    unknown_key_policy: UnknownKeyPolicy = UnknownKeyPolicy.WARN,
) -> ConfigLoadResult:
    """Load private runtime configuration with legacy-file compatibility."""

    return load_config(paths.active_config_file, unknown_key_policy=unknown_key_policy)
