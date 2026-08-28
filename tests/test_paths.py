from pathlib import Path

import pytest

from jarvis.core.paths import JarvisPaths, first_existing


def test_paths_do_not_depend_on_current_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    paths = JarvisPaths.from_repository_root(root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert paths.repository_root == root.resolve()
    assert paths.legacy_config_file == root.resolve() / "config.json"
    assert paths.memory_file == root.resolve() / "data" / "memory.json"
    assert paths.recordings_dir == root.resolve() / "data" / "recordings"
    assert paths.stt_temp_dir == root.resolve() / "data" / "stt"
    assert paths.whisper_model == root.resolve() / "data" / "models" / "whisper" / "ggml-small.bin"
    assert paths.whisper_model_for("base") == root.resolve() / "data" / "models" / "whisper" / "ggml-base.bin"
    assert paths.whisper_model_for("SMALL") == paths.whisper_model
    assert paths.resolve_from_root("models/test.bin") == root.resolve() / "models" / "test.bin"
    assert paths.kokoro_model == root.resolve() / "data" / "models" / "tts" / "kokoro" / "kokoro-v1.0.onnx"
    assert paths.kokoro_voices.name == "voices-v1.0.bin"
    assert paths.piper_voice_files("en_US-joe-medium")[1].name == "en_US-joe-medium.onnx.json"
    assert paths.tts_benchmark_dir == root.resolve() / "data" / "benchmarks" / "tts"


def test_repository_root_must_be_explicitly_absolute():
    with pytest.raises(ValueError, match="absolute path"):
        JarvisPaths.from_repository_root("relative/repository")


def test_whisper_model_paths_are_allowlisted(tmp_path):
    paths = JarvisPaths.from_repository_root(tmp_path)

    with pytest.raises(ValueError, match="base, small"):
        paths.whisper_model_for("../../arbitrary")

    with pytest.raises(ValueError, match="Piper voice"):
        paths.piper_voice_files("../../arbitrary")


def test_constructing_paths_does_not_create_runtime_directory(tmp_path):
    paths = JarvisPaths.from_repository_root(tmp_path)

    assert not paths.data_dir.exists()
    paths.ensure_runtime_directories()
    assert paths.data_dir.is_dir()


def test_private_config_takes_precedence_over_legacy(tmp_path):
    paths = JarvisPaths.from_repository_root(tmp_path)
    paths.legacy_config_file.write_text("{}", encoding="utf-8")
    assert paths.active_config_file == paths.legacy_config_file

    paths.ensure_runtime_directories()
    paths.user_config_file.write_text("{}", encoding="utf-8")
    assert paths.active_config_file == paths.user_config_file


def test_discover_uses_anchor_not_current_directory(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    nested = root / "jarvis" / "core"
    nested.mkdir(parents=True)
    (root / "LICENSE").write_text("license", encoding="utf-8")
    (root / "agent.py").write_text("pass", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert JarvisPaths.discover(nested).repository_root == root.resolve()


def test_first_existing_preserves_candidate_order(tmp_path):
    first = tmp_path / "first.exe"
    second = tmp_path / "second"
    second.write_text("", encoding="utf-8")
    first.write_text("", encoding="utf-8")

    assert first_existing((first, second)) == first
    assert first_existing((tmp_path / "missing", second)) == second


def test_windows_executable_candidates_are_explicit(tmp_path):
    paths = JarvisPaths.from_repository_root(tmp_path)

    assert paths.whisper_executable_candidates[0].name == "whisper-cli.exe"
    assert "v1.9.1" in str(paths.whisper_executable_candidates[0])
    assert paths.piper_executable_candidates[0].name == "piper.exe"
    assert all(isinstance(path, Path) for path in paths.whisper_executable_candidates)
