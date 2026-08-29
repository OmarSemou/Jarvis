"""Validated Jarvis configuration with explicit legacy compatibility."""

from __future__ import annotations

import json
import math
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
        "llm_model",
        "llm_thinking",
        "llm_temperature",
        "conversation_max_turns",
        "conversation_max_tool_rounds",
        "ollama_host",
        "ollama_connect_timeout_seconds",
        "ollama_read_timeout_seconds",
        "ollama_keep_alive",
        "whisper_executable_path",
        "stt_model",
        "stt_language",
        "stt_timeout_seconds",
        "stt_use_gpu",
        "retain_recordings",
        "output_device",
        "tts_enabled",
        "tts_provider",
        "tts_voice",
        "tts_profile",
        "tts_speed",
        "tts_language",
        "voice_mode_enabled",
        "wakeword_enabled",
        "wakeword_threshold",
        "vad_enabled",
        "vad_speech_threshold",
        "vad_trailing_silence_ms",
        "vad_max_utterance_seconds",
        "vad_min_speech_ms",
        "vad_listen_timeout_seconds",
        "barge_in_enabled",
        "barge_in_mode",
        "barge_in_threshold",
        "barge_in_suppression_ms",
        "barge_in_min_speech_ms",
        "barge_in_pre_roll_ms",
        "barge_in_command_start_timeout_seconds",
        "tts_preload",
        "voice_debug_latency",
        "voice_ollama_keep_alive",
        "memory_enabled",
        "memory_database_path",
        "memory_max_context_entries",
        "memory_max_context_chars",
        "memory_max_value_chars",
        "memory_max_key_chars",
        "memory_max_summary_chars",
        "memory_max_records",
    }
)

SUPPORTED_TTS_VOICES: dict[str, frozenset[str]] = {
    "kokoro": frozenset({"am_fenrir", "am_michael", "am_puck", "bm_george"}),
    "piper": frozenset({"en_US-joe-medium", "en_US-john-medium"}),
}


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
    llm_model: str = "qwen3:8b"
    llm_thinking: bool = False
    llm_temperature: float = 0.2
    conversation_max_turns: int = 12
    conversation_max_tool_rounds: int = 3
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_connect_timeout_seconds: float = 3.0
    ollama_read_timeout_seconds: float = 120.0
    ollama_keep_alive: str = "5m"
    whisper_executable_path: str = (
        "data/tools/whisper.cpp/v1.9.1/Release/whisper-cli.exe"
    )
    stt_model: str = "base"
    stt_language: str = "auto"
    stt_timeout_seconds: float = 180.0
    stt_use_gpu: bool = False
    retain_recordings: bool = False
    output_device: int | str | None = None
    tts_enabled: bool = False
    tts_provider: str = "kokoro"
    tts_voice: str = "am_fenrir"
    tts_profile: str | None = None
    tts_speed: float = 1.0
    tts_language: str = "en"
    voice_mode_enabled: bool = False
    wakeword_enabled: bool = True
    wakeword_threshold: float = 0.5
    vad_enabled: bool = True
    vad_speech_threshold: float = 0.5
    vad_trailing_silence_ms: int = 640
    vad_max_utterance_seconds: float = 18.0
    vad_min_speech_ms: int = 240
    vad_listen_timeout_seconds: float = 8.0
    barge_in_enabled: bool = True
    barge_in_mode: str = "wakeword"
    barge_in_threshold: float = 0.75
    barge_in_suppression_ms: int = 480
    barge_in_min_speech_ms: int = 240
    barge_in_pre_roll_ms: int = 320
    barge_in_command_start_timeout_seconds: float = 1.5
    tts_preload: bool = True
    voice_debug_latency: bool = False
    voice_ollama_keep_alive: str = "30m"
    memory_enabled: bool = True
    memory_database_path: str = "bmo.db"
    memory_max_context_entries: int = 8
    memory_max_context_chars: int = 3000
    memory_max_value_chars: int = 500
    memory_max_key_chars: int = 80
    memory_max_summary_chars: int = 240
    memory_max_records: int = 500

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
            "llm_model": self.llm_model,
            "llm_thinking": self.llm_thinking,
            "llm_temperature": self.llm_temperature,
            "conversation_max_turns": self.conversation_max_turns,
            "conversation_max_tool_rounds": self.conversation_max_tool_rounds,
            "ollama_host": self.ollama_host,
            "ollama_connect_timeout_seconds": self.ollama_connect_timeout_seconds,
            "ollama_read_timeout_seconds": self.ollama_read_timeout_seconds,
            "ollama_keep_alive": self.ollama_keep_alive,
            "whisper_executable_path": self.whisper_executable_path,
            "stt_model": self.stt_model,
            "stt_language": self.stt_language,
            "stt_timeout_seconds": self.stt_timeout_seconds,
            "stt_use_gpu": self.stt_use_gpu,
            "retain_recordings": self.retain_recordings,
            "output_device": self.output_device,
            "tts_enabled": self.tts_enabled,
            "tts_provider": self.tts_provider,
            "tts_voice": self.tts_voice,
            "tts_profile": self.tts_profile,
            "tts_speed": self.tts_speed,
            "tts_language": self.tts_language,
            "voice_mode_enabled": self.voice_mode_enabled,
            "wakeword_enabled": self.wakeword_enabled,
            "wakeword_threshold": self.wakeword_threshold,
            "vad_enabled": self.vad_enabled,
            "vad_speech_threshold": self.vad_speech_threshold,
            "vad_trailing_silence_ms": self.vad_trailing_silence_ms,
            "vad_max_utterance_seconds": self.vad_max_utterance_seconds,
            "vad_min_speech_ms": self.vad_min_speech_ms,
            "vad_listen_timeout_seconds": self.vad_listen_timeout_seconds,
            "barge_in_enabled": self.barge_in_enabled,
            "barge_in_mode": self.barge_in_mode,
            "barge_in_threshold": self.barge_in_threshold,
            "barge_in_suppression_ms": self.barge_in_suppression_ms,
            "barge_in_min_speech_ms": self.barge_in_min_speech_ms,
            "barge_in_pre_roll_ms": self.barge_in_pre_roll_ms,
            "barge_in_command_start_timeout_seconds": (
                self.barge_in_command_start_timeout_seconds
            ),
            "tts_preload": self.tts_preload,
            "voice_debug_latency": self.voice_debug_latency,
            "voice_ollama_keep_alive": self.voice_ollama_keep_alive,
            "memory_enabled": self.memory_enabled,
            "memory_database_path": self.memory_database_path,
            "memory_max_context_entries": self.memory_max_context_entries,
            "memory_max_context_chars": self.memory_max_context_chars,
            "memory_max_value_chars": self.memory_max_value_chars,
            "memory_max_key_chars": self.memory_max_key_chars,
            "memory_max_summary_chars": self.memory_max_summary_chars,
            "memory_max_records": self.memory_max_records,
        }


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    config: JarvisConfig
    source: Path | None
    migrations: tuple[str, ...] = field(default_factory=tuple)
    unknown_keys: tuple[str, ...] = field(default_factory=tuple)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_string(values: Mapping[str, Any], key: str, problems: list[str]) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        problems.append(f"'{key}' must be a non-empty string")
        return ""
    return value.strip()


def _migrate_keys(raw: Mapping[str, Any], problems: list[str]) -> tuple[dict[str, Any], list[str]]:
    values = dict(raw)
    migrations: list[str] = []
    # Phase 2E accepts a structured ``memory`` object while retaining the
    # upstream boolean ``memory`` compatibility key.  A mapping is therefore
    # unambiguously the new form; a bool continues to mean ``chat_memory``.
    nested_memory = values.get("memory")
    if isinstance(nested_memory, Mapping):
        values.pop("memory", None)
        nested_map = {
            "enabled": "memory_enabled",
            "database_path": "memory_database_path",
            "max_context_entries": "memory_max_context_entries",
            "max_context_chars": "memory_max_context_chars",
            "max_value_chars": "memory_max_value_chars",
            "max_key_chars": "memory_max_key_chars",
            "max_summary_chars": "memory_max_summary_chars",
            "max_records": "memory_max_records",
        }
        for nested_key, current_key in nested_map.items():
            if nested_key not in nested_memory:
                continue
            if current_key in values:
                problems.append(f"memory key '{nested_key}' conflicts with '{current_key}'")
            else:
                values[current_key] = nested_memory[nested_key]
        unknown_nested = sorted(set(nested_memory) - set(nested_map))
        if unknown_nested:
            problems.append("unknown memory key(s): " + ", ".join(unknown_nested))
        migrations.append("structured 'memory' settings were migrated to validated memory_* keys")
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

    llm_model = _validate_string(merged, "llm_model", problems)

    llm_thinking = merged["llm_thinking"]
    if not isinstance(llm_thinking, bool):
        problems.append("'llm_thinking' must be a boolean")

    llm_temperature = merged["llm_temperature"]
    if not _is_finite_number(llm_temperature) or not 0 <= llm_temperature <= 2:
        problems.append("'llm_temperature' must be a number from 0 to 2")

    conversation_max_turns = merged["conversation_max_turns"]
    if not _is_int(conversation_max_turns) or not 1 <= conversation_max_turns <= 100:
        problems.append("'conversation_max_turns' must be an integer from 1 to 100")

    conversation_max_tool_rounds = merged["conversation_max_tool_rounds"]
    if not _is_int(conversation_max_tool_rounds) or not 1 <= conversation_max_tool_rounds <= 10:
        problems.append("'conversation_max_tool_rounds' must be an integer from 1 to 10")

    ollama_host = _validate_string(merged, "ollama_host", problems)

    ollama_connect_timeout_seconds = merged["ollama_connect_timeout_seconds"]
    if not _is_finite_number(ollama_connect_timeout_seconds) or not 0 < ollama_connect_timeout_seconds <= 60:
        problems.append("'ollama_connect_timeout_seconds' must be a number greater than 0 and at most 60")

    ollama_read_timeout_seconds = merged["ollama_read_timeout_seconds"]
    if not _is_finite_number(ollama_read_timeout_seconds) or not 0 < ollama_read_timeout_seconds <= 600:
        problems.append("'ollama_read_timeout_seconds' must be a number greater than 0 and at most 600")

    ollama_keep_alive = _validate_string(merged, "ollama_keep_alive", problems)

    whisper_executable_path = _validate_string(merged, "whisper_executable_path", problems)
    stt_model = _validate_string(merged, "stt_model", problems).casefold()
    if stt_model not in {"base", "small"}:
        problems.append("'stt_model' must be one of: base, small")

    stt_language = _validate_string(merged, "stt_language", problems).casefold()
    if stt_language not in {"auto", "en", "da"}:
        problems.append("'stt_language' must be one of: auto, en, da")

    stt_timeout_seconds = merged["stt_timeout_seconds"]
    if not _is_finite_number(stt_timeout_seconds) or not 0 < stt_timeout_seconds <= 600:
        problems.append("'stt_timeout_seconds' must be a number greater than 0 and at most 600")

    stt_use_gpu = merged["stt_use_gpu"]
    if not isinstance(stt_use_gpu, bool):
        problems.append("'stt_use_gpu' must be a boolean")

    retain_recordings = merged["retain_recordings"]
    if not isinstance(retain_recordings, bool):
        problems.append("'retain_recordings' must be a boolean")

    output_device = merged["output_device"]
    if output_device is not None:
        if _is_int(output_device):
            if output_device < 0:
                problems.append("'output_device' integer must be non-negative")
        elif isinstance(output_device, str):
            if not output_device.strip():
                problems.append("'output_device' string must not be empty")
        else:
            problems.append("'output_device' must be an integer, string, or null")

    tts_enabled = merged["tts_enabled"]
    if not isinstance(tts_enabled, bool):
        problems.append("'tts_enabled' must be a boolean")

    raw_profile = merged["tts_profile"]
    if raw_profile is None:
        tts_profile = None
    elif isinstance(raw_profile, str) and raw_profile.strip():
        tts_profile = raw_profile.strip().casefold()
        if tts_profile not in {"fenrir", "bmo"}:
            problems.append("'tts_profile' must be one of: fenrir, bmo")
    else:
        tts_profile = None
        problems.append("'tts_profile' must be one of: fenrir, bmo, or null")

    tts_provider = _validate_string(merged, "tts_provider", problems).casefold()
    if tts_provider not in SUPPORTED_TTS_VOICES:
        problems.append("'tts_provider' must be one of: kokoro, piper")

    tts_voice = _validate_string(merged, "tts_voice", problems)
    if (
        tts_profile is None
        and tts_provider in SUPPORTED_TTS_VOICES
        and tts_voice not in SUPPORTED_TTS_VOICES[tts_provider]
    ):
        choices = ", ".join(sorted(SUPPORTED_TTS_VOICES[tts_provider]))
        problems.append(f"'tts_voice' must be one of the configured {tts_provider} voices: {choices}")

    tts_speed = merged["tts_speed"]
    if not _is_finite_number(tts_speed) or not 0.5 <= tts_speed <= 2.0:
        problems.append("'tts_speed' must be a number from 0.5 to 2.0")

    tts_language = _validate_string(merged, "tts_language", problems).casefold()
    if tts_language != "en":
        problems.append("'tts_language' must be 'en' in Phase 2C3")

    voice_mode_enabled = merged["voice_mode_enabled"]
    wakeword_enabled = merged["wakeword_enabled"]
    vad_enabled = merged["vad_enabled"]
    barge_in_enabled = merged["barge_in_enabled"]
    barge_in_mode = _validate_string(merged, "barge_in_mode", problems).casefold()
    if barge_in_mode not in {"wakeword", "vad_experimental"}:
        problems.append(
            "'barge_in_mode' must be one of: wakeword, vad_experimental"
        )
    tts_preload = merged["tts_preload"]
    voice_debug_latency = merged["voice_debug_latency"]
    for key, value in (
        ("voice_mode_enabled", voice_mode_enabled),
        ("wakeword_enabled", wakeword_enabled),
        ("vad_enabled", vad_enabled),
        ("barge_in_enabled", barge_in_enabled),
        ("tts_preload", tts_preload),
        ("voice_debug_latency", voice_debug_latency),
    ):
        if not isinstance(value, bool):
            problems.append(f"'{key}' must be a boolean")

    wakeword_threshold = merged["wakeword_threshold"]
    vad_speech_threshold = merged["vad_speech_threshold"]
    barge_in_threshold = merged["barge_in_threshold"]
    for key, value in (
        ("wakeword_threshold", wakeword_threshold),
        ("vad_speech_threshold", vad_speech_threshold),
        ("barge_in_threshold", barge_in_threshold),
    ):
        if not _is_finite_number(value) or not 0.05 <= value <= 0.99:
            problems.append(f"'{key}' must be a number from 0.05 to 0.99")

    vad_trailing_silence_ms = merged["vad_trailing_silence_ms"]
    vad_min_speech_ms = merged["vad_min_speech_ms"]
    barge_in_suppression_ms = merged["barge_in_suppression_ms"]
    barge_in_min_speech_ms = merged["barge_in_min_speech_ms"]
    barge_in_pre_roll_ms = merged["barge_in_pre_roll_ms"]
    for key, value, lower, upper in (
        ("vad_trailing_silence_ms", vad_trailing_silence_ms, 300, 2_000),
        ("vad_min_speech_ms", vad_min_speech_ms, 60, 2_000),
        ("barge_in_suppression_ms", barge_in_suppression_ms, 0, 3_000),
        ("barge_in_min_speech_ms", barge_in_min_speech_ms, 60, 1_000),
        ("barge_in_pre_roll_ms", barge_in_pre_roll_ms, 200, 500),
    ):
        if not _is_int(value) or not lower <= value <= upper:
            problems.append(f"'{key}' must be an integer from {lower} to {upper}")

    vad_max_utterance_seconds = merged["vad_max_utterance_seconds"]
    if not _is_finite_number(vad_max_utterance_seconds) or not 3 <= vad_max_utterance_seconds <= 60:
        problems.append("'vad_max_utterance_seconds' must be a number from 3 to 60")
    vad_listen_timeout_seconds = merged["vad_listen_timeout_seconds"]
    if not _is_finite_number(vad_listen_timeout_seconds) or not 1 <= vad_listen_timeout_seconds <= 60:
        problems.append("'vad_listen_timeout_seconds' must be a number from 1 to 60")

    barge_in_command_start_timeout_seconds = merged[
        "barge_in_command_start_timeout_seconds"
    ]
    if (
        not _is_finite_number(barge_in_command_start_timeout_seconds)
        or not 0.5 <= barge_in_command_start_timeout_seconds <= 3.0
    ):
        problems.append(
            "'barge_in_command_start_timeout_seconds' must be a number from 0.5 to 3"
        )

    voice_ollama_keep_alive = _validate_string(
        merged, "voice_ollama_keep_alive", problems
    )

    memory_enabled = merged["memory_enabled"]
    if not isinstance(memory_enabled, bool):
        problems.append("'memory_enabled' must be a boolean")
    memory_database_path = _validate_string(merged, "memory_database_path", problems)
    if memory_database_path:
        path_value = Path(memory_database_path).expanduser()
        if (
            path_value.is_absolute()
            or ".." in path_value.parts
            or not path_value.parts
            or (len(path_value.parts) == 1 and path_value.parts[0].casefold() == "data")
        ):
            problems.append("'memory_database_path' must be a relative path inside data/")
    for key in (
        "memory_max_context_entries",
        "memory_max_context_chars",
        "memory_max_value_chars",
        "memory_max_key_chars",
        "memory_max_summary_chars",
        "memory_max_records",
    ):
        value = merged[key]
        if not _is_int(value) or value <= 0:
            problems.append(f"'{key}' must be a positive integer")
    if _is_int(merged["memory_max_context_entries"]) and merged["memory_max_context_entries"] > 50:
        problems.append("'memory_max_context_entries' must be at most 50")
    if _is_int(merged["memory_max_context_chars"]) and merged["memory_max_context_chars"] > 20_000:
        problems.append("'memory_max_context_chars' must be at most 20000")
    if _is_int(merged["memory_max_records"]) and merged["memory_max_records"] > 10_000:
        problems.append("'memory_max_records' must be at most 10000")

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
        llm_model=llm_model,
        llm_thinking=llm_thinking,
        llm_temperature=float(llm_temperature),
        conversation_max_turns=conversation_max_turns,
        conversation_max_tool_rounds=conversation_max_tool_rounds,
        ollama_host=ollama_host,
        ollama_connect_timeout_seconds=float(ollama_connect_timeout_seconds),
        ollama_read_timeout_seconds=float(ollama_read_timeout_seconds),
        ollama_keep_alive=ollama_keep_alive,
        whisper_executable_path=whisper_executable_path,
        stt_model=stt_model,
        stt_language=stt_language,
        stt_timeout_seconds=float(stt_timeout_seconds),
        stt_use_gpu=stt_use_gpu,
        retain_recordings=retain_recordings,
        output_device=output_device.strip() if isinstance(output_device, str) else output_device,
        tts_enabled=tts_enabled,
        tts_provider=tts_provider,
        tts_voice=tts_voice,
        tts_profile=tts_profile,
        tts_speed=float(tts_speed),
        tts_language=tts_language,
        voice_mode_enabled=voice_mode_enabled,
        wakeword_enabled=wakeword_enabled,
        wakeword_threshold=float(wakeword_threshold),
        vad_enabled=vad_enabled,
        vad_speech_threshold=float(vad_speech_threshold),
        vad_trailing_silence_ms=vad_trailing_silence_ms,
        vad_max_utterance_seconds=float(vad_max_utterance_seconds),
        vad_min_speech_ms=vad_min_speech_ms,
        vad_listen_timeout_seconds=float(vad_listen_timeout_seconds),
        barge_in_enabled=barge_in_enabled,
        barge_in_mode=barge_in_mode,
        barge_in_threshold=float(barge_in_threshold),
        barge_in_suppression_ms=barge_in_suppression_ms,
        barge_in_min_speech_ms=barge_in_min_speech_ms,
        barge_in_pre_roll_ms=barge_in_pre_roll_ms,
        barge_in_command_start_timeout_seconds=float(
            barge_in_command_start_timeout_seconds
        ),
        tts_preload=tts_preload,
        voice_debug_latency=voice_debug_latency,
        voice_ollama_keep_alive=voice_ollama_keep_alive,
        memory_enabled=memory_enabled,
        memory_database_path=memory_database_path,
        memory_max_context_entries=merged["memory_max_context_entries"],
        memory_max_context_chars=merged["memory_max_context_chars"],
        memory_max_value_chars=merged["memory_max_value_chars"],
        memory_max_key_chars=merged["memory_max_key_chars"],
        memory_max_summary_chars=merged["memory_max_summary_chars"],
        memory_max_records=merged["memory_max_records"],
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
