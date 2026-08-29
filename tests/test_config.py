import json

import pytest

from jarvis.core.config import (
    ConfigValidationError,
    JarvisConfig,
    UnknownKeyPolicy,
    load_config,
    parse_config,
)


def test_system_prompt_and_extras_are_respected():
    result = parse_config(
        {
            "system_prompt": "You are Jarvis.",
            "system_prompt_extras": "Be concise.",
        }
    )

    assert result.config.effective_system_prompt("fallback") == "You are Jarvis.\n\nBe concise."


def test_prompt_falls_back_when_not_configured():
    config = JarvisConfig(system_prompt=None, system_prompt_extras="Local only.")

    assert config.effective_system_prompt("Default prompt.") == "Default prompt.\n\nLocal only."


def test_chat_memory_boolean_is_respected():
    result = parse_config({"chat_memory": False})

    assert result.config.chat_memory is False
    assert result.config.as_legacy_dict()["chat_memory"] is False


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"chat_memory": "yes"}, "'chat_memory' must be a boolean"),
        ({"input_sample_rate": True}, "'input_sample_rate' must be an integer"),
        ({"camera_rotation": 45}, "'camera_rotation' must be one of"),
        ({"input_device": -1}, "'input_device' integer must be non-negative"),
        ({"system_prompt": 42}, "'system_prompt' must be a string or null"),
    ],
)
def test_invalid_types_produce_clear_errors(raw, message):
    with pytest.raises(ConfigValidationError, match=message):
        parse_config(raw, source="bad-config.json")


def test_unknown_keys_default_to_error():
    with pytest.raises(ConfigValidationError, match=r"unknown key\(s\): secret_option"):
        parse_config({"secret_option": "unexpected"})


def test_unknown_keys_can_be_deliberately_warned_and_ignored():
    with pytest.warns(UserWarning, match="Ignoring unknown"):
        result = parse_config(
            {"chat_memory": False, "future_key": 1},
            unknown_key_policy=UnknownKeyPolicy.WARN,
        )

    assert result.config.chat_memory is False
    assert result.unknown_keys == ("future_key",)


def test_legacy_keys_are_migrated_with_notices():
    result = parse_config(
        {
            "model": "legacy-model",
            "memory": False,
            "prompt": "Legacy prompt",
            "sample_rate": 16_000,
        }
    )

    assert result.config.text_model == "legacy-model"
    assert result.config.chat_memory is False
    assert result.config.system_prompt == "Legacy prompt"
    assert result.config.input_sample_rate == 16_000
    assert len(result.migrations) == 4


def test_legacy_and_current_key_conflict_is_rejected():
    with pytest.raises(ConfigValidationError, match="legacy key 'model' conflicts"):
        parse_config({"model": "old", "text_model": "new"})


def test_load_config_reads_utf8_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"system_prompt": "Hej Jarvis", "chat_memory": False}), encoding="utf-8")

    result = load_config(path)

    assert result.source == path.resolve()
    assert result.config.system_prompt == "Hej Jarvis"
    assert result.config.chat_memory is False


def test_load_config_rejects_non_object_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="top-level JSON value must be an object"):
        load_config(path)


def test_missing_config_uses_safe_defaults(tmp_path):
    result = load_config(tmp_path / "missing.json")

    assert result.source is None
    assert result.config == JarvisConfig()


def test_phase2a_model_and_thinking_defaults():
    config = JarvisConfig()

    assert config.llm_model == "qwen3:8b"
    assert config.llm_thinking is False
    assert config.llm_temperature == 0.2
    assert config.ollama_host == "http://127.0.0.1:11434"
    assert config.conversation_max_turns == 12
    assert config.conversation_max_tool_rounds == 3
    assert config.stt_language == "auto"
    assert config.stt_use_gpu is False
    assert config.retain_recordings is False
    assert config.stt_model == "base"
    assert config.tts_enabled is False
    assert config.tts_provider == "kokoro"
    assert config.tts_voice == "am_fenrir"
    assert config.tts_speed == 1.0
    assert config.voice_mode_enabled is False
    assert config.wakeword_enabled is True
    assert config.vad_enabled is True
    assert config.barge_in_enabled is True
    assert config.barge_in_mode == "wakeword"
    assert config.barge_in_pre_roll_ms == 320
    assert config.barge_in_command_start_timeout_seconds == 1.5
    assert config.voice_ollama_keep_alive == "30m"


def test_phase2c3_voice_configuration_is_parsed():
    result = parse_config(
        {
            "voice_mode_enabled": True,
            "wakeword_enabled": False,
            "wakeword_threshold": 0.61,
            "vad_speech_threshold": 0.55,
            "vad_trailing_silence_ms": 700,
            "vad_max_utterance_seconds": 15,
            "vad_min_speech_ms": 300,
            "vad_listen_timeout_seconds": 6,
            "barge_in_enabled": False,
            "barge_in_mode": "VAD_EXPERIMENTAL",
            "barge_in_threshold": 0.8,
            "barge_in_suppression_ms": 600,
            "barge_in_min_speech_ms": 300,
            "barge_in_pre_roll_ms": 400,
            "barge_in_command_start_timeout_seconds": 2.0,
            "tts_preload": False,
            "voice_debug_latency": True,
            "voice_ollama_keep_alive": "45m",
        }
    )

    assert result.config.voice_mode_enabled
    assert not result.config.wakeword_enabled
    assert result.config.wakeword_threshold == 0.61
    assert result.config.vad_trailing_silence_ms == 700
    assert result.config.vad_max_utterance_seconds == 15.0
    assert not result.config.barge_in_enabled
    assert result.config.barge_in_mode == "vad_experimental"
    assert result.config.barge_in_pre_roll_ms == 400
    assert result.config.barge_in_command_start_timeout_seconds == 2.0
    assert result.config.voice_debug_latency
    assert result.config.voice_ollama_keep_alive == "45m"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"voice_mode_enabled": 1}, "must be a boolean"),
        ({"wakeword_threshold": 1.0}, "from 0.05 to 0.99"),
        ({"vad_speech_threshold": 0.0}, "from 0.05 to 0.99"),
        ({"vad_trailing_silence_ms": 100}, "integer from 300 to 2000"),
        ({"vad_max_utterance_seconds": 2}, "number from 3 to 60"),
        ({"barge_in_suppression_ms": -1}, "integer from 0 to 3000"),
        ({"barge_in_pre_roll_ms": 800}, "integer from 200 to 500"),
        (
            {"barge_in_command_start_timeout_seconds": 0.1},
            "number from 0.5 to 3",
        ),
        ({"barge_in_mode": "speech"}, "wakeword, vad_experimental"),
        ({"voice_ollama_keep_alive": ""}, "non-empty string"),
    ],
)
def test_invalid_phase2c3_configuration_is_rejected(raw, message):
    with pytest.raises(ConfigValidationError, match=message):
        parse_config(raw)


def test_phase2a_configuration_is_parsed():
    result = parse_config(
        {
            "llm_model": "qwen3:14b",
            "llm_thinking": True,
            "llm_temperature": 0.25,
            "conversation_max_turns": 4,
            "conversation_max_tool_rounds": 2,
            "ollama_host": "http://localhost:11434",
            "ollama_connect_timeout_seconds": 2,
            "ollama_read_timeout_seconds": 30.5,
            "ollama_keep_alive": "10m",
        }
    )

    assert result.config.llm_model == "qwen3:14b"
    assert result.config.llm_thinking is True
    assert result.config.llm_temperature == 0.25
    assert result.config.conversation_max_turns == 4
    assert result.config.conversation_max_tool_rounds == 2
    assert result.config.ollama_connect_timeout_seconds == 2.0
    assert result.config.ollama_read_timeout_seconds == 30.5


def test_phase2c1_configuration_is_parsed():
    result = parse_config(
        {
            "whisper_executable_path": "private/whisper-cli.exe",
            "stt_model": "BASE",
            "stt_language": "DA",
            "stt_timeout_seconds": 45,
            "stt_use_gpu": False,
            "retain_recordings": True,
        }
    )

    assert result.config.whisper_executable_path == "private/whisper-cli.exe"
    assert result.config.stt_model == "base"
    assert result.config.stt_language == "da"
    assert result.config.stt_timeout_seconds == 45.0
    assert result.config.stt_use_gpu is False
    assert result.config.retain_recordings is True


def test_phase2c2_tts_configuration_is_parsed_and_voice_can_be_off():
    result = parse_config(
        {
            "output_device": "USB Speakers",
            "tts_enabled": False,
            "tts_provider": "PIPER",
            "tts_voice": "en_US-john-medium",
            "tts_speed": 1.2,
            "tts_language": "EN",
        }
    )

    assert result.config.output_device == "USB Speakers"
    assert result.config.tts_enabled is False
    assert result.config.tts_provider == "piper"
    assert result.config.tts_voice == "en_US-john-medium"
    assert result.config.tts_speed == 1.2
    assert result.config.tts_language == "en"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"tts_enabled": 1}, "'tts_enabled' must be a boolean"),
        ({"tts_provider": "cloud"}, "must be one of: kokoro, piper"),
        ({"tts_provider": "kokoro", "tts_voice": "arbitrary"}, "configured kokoro voices"),
        ({"tts_provider": "piper", "tts_voice": "am_michael"}, "configured piper voices"),
        ({"tts_speed": 0.1}, "must be a number from 0.5 to 2.0"),
        ({"tts_language": "da"}, "must be 'en'"),
        ({"output_device": -1}, "integer must be non-negative"),
    ],
)
def test_invalid_phase2c2_configuration_is_rejected(raw, message):
    with pytest.raises(ConfigValidationError, match=message):
        parse_config(raw)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"llm_thinking": 1}, "'llm_thinking' must be a boolean"),
        ({"llm_temperature": 2.1}, "number from 0 to 2"),
        ({"conversation_max_turns": 0}, "'conversation_max_turns' must be an integer"),
        ({"conversation_max_tool_rounds": 0}, "'conversation_max_tool_rounds' must be an integer"),
        ({"ollama_connect_timeout_seconds": True}, "must be a number"),
        ({"ollama_read_timeout_seconds": float("inf")}, "must be a number"),
        ({"ollama_keep_alive": ""}, "must be a non-empty string"),
        ({"stt_language": "fr"}, "must be one of"),
        ({"stt_timeout_seconds": 0}, "must be a number"),
        ({"stt_use_gpu": "yes"}, "must be a boolean"),
        ({"retain_recordings": 1}, "must be a boolean"),
        ({"whisper_executable_path": ""}, "must be a non-empty string"),
        ({"stt_model": "large"}, "must be one of: base, small"),
        ({"whisper_model_path": "private/model.bin"}, "unknown key"),
    ],
)
def test_invalid_phase2a_configuration_is_rejected(raw, message):
    with pytest.raises(ConfigValidationError, match=message):
        parse_config(raw)
