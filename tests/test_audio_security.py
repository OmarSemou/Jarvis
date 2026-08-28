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
    assert "small.en" not in script
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
