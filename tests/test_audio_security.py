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
    for forbidden in ("torch", "transformers", "langchain", "langgraph", "opencv", "openwakeword"):
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
