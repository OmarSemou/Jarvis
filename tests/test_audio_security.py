import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_setup_script_is_pinned_verified_and_never_uses_english_only_model():
    script = (ROOT / "scripts" / "setup_whisper_windows.ps1").read_text(encoding="utf-8")

    assert "v1.9.1" in script
    assert "releases/download/v1.9.1/whisper-bin-x64.zip" in script
    assert "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539" in script
    assert "ggml-small.bin" in script
    assert "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b" in script
    assert "ggml-base.bin" in script
    assert "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe" in script
    assert 'ValidateSet("base", "small", "all")' in script
    assert "small.en" not in script
    assert "base.en" not in script
    assert "/latest/" not in script
    assert "SetEnvironmentVariable" not in script


def test_runtime_data_root_remains_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/" in ignore


def test_conversation_service_has_no_audio_or_whisper_dependency():
    tree = ast.parse((ROOT / "jarvis" / "core" / "conversation.py").read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert not any(name.startswith("jarvis.audio") for name in imports)
    assert not any("whisper" in name for name in imports)


def test_conversation_service_has_no_tts_provider_or_playback_dependency():
    source = (ROOT / "jarvis" / "core" / "conversation.py").read_text(encoding="utf-8")
    assert "kokoro" not in source.casefold()
    assert "piper" not in source.casefold()
    assert "sounddevice" not in source.casefold()
    assert "playback" not in source.casefold()


def test_tts_setup_is_pinned_hash_verified_and_avoids_disallowed_frameworks():
    script = (ROOT / "scripts" / "setup_tts_windows.ps1").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "kokoro-onnx==0.6.1" in script
    assert "piper-tts==1.7.0" in script
    assert "model-files-v1.0" in script
    assert "resolve/v1.0.0" in script
    assert script.count("Install-VerifiedAsset") >= 7
    assert "Get-FileHash -Algorithm SHA256" in script
    assert "SetEnvironmentVariable" not in script
    combined = (script + requirements).casefold()
    assert '"torch' not in combined
    for forbidden in ("transformers", "langchain", "langgraph", "opencv"):
        assert forbidden not in combined


def test_voice_setup_is_pinned_hash_verified_and_onnx_only():
    script = (ROOT / "scripts" / "setup_voice_windows.ps1").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "openwakeword==0.6.0" in script
    assert "openwakeword==0.6.0" in requirements
    assert "releases/download/v0.5.1" in script
    assert "hey_jarvis_v0.1.onnx" in script
    assert "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb" in script
    assert script.count("Install-VerifiedAsset") >= 5
    assert "wakeword.onnx" not in script
    assert script.count("Get-FileHash -Algorithm SHA256") >= 2
    combined = (script + requirements).casefold()
    assert '"torch' not in combined
    for forbidden in ("transformers", "langchain", "langgraph", "opencv", "ros"):
        assert forbidden not in combined


def test_benchmark_module_has_no_llm_network_or_download_dependency():
    tree = ast.parse((ROOT / "jarvis" / "audio" / "benchmark.py").read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    forbidden = ("jarvis.llm", "ollama", "httpx", "requests", "urllib", "socket")
    assert not any(name.startswith(forbidden) for name in imports)


def test_voice_coordinator_has_no_direct_robot_or_safety_authority():
    tree = ast.parse(
        (ROOT / "jarvis" / "audio" / "voice" / "coordinator.py").read_text(
            encoding="utf-8"
        )
    )
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert not any(name.startswith("jarvis.robot") for name in imports)
    assert not any(name.startswith("jarvis.tools") for name in imports)
    source = (ROOT / "jarvis" / "audio" / "voice" / "coordinator.py").read_text(
        encoding="utf-8"
    )
    assert "reset_emergency_stop" not in source
    assert "clear_estop" not in source


def test_local_voice_stop_integration_is_narrow_and_never_calls_simulator_directly():
    source = (
        ROOT / "jarvis" / "integrations" / "voice_stop.py"
    ).read_text(encoding="utf-8")
    normalized = source.casefold()

    assert "saferobotcontroller" in normalized
    assert "execute_intent" in normalized
    assert "robotaction.stop" in normalized
    assert "simulatedrobot" not in normalized
    assert "reset_emergency_stop" not in normalized
    assert "latch_emergency_stop" not in normalized
