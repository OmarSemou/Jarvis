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
