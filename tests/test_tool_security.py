import ast
from pathlib import Path

from jarvis.tools.registry import RobotToolRegistry


ROOT = Path(__file__).resolve().parents[1]
SECURITY_FILES = (
    ROOT / "jarvis" / "tools" / "registry.py",
    ROOT / "jarvis" / "tools" / "policy.py",
    ROOT / "jarvis" / "robot" / "controller.py",
    ROOT / "jarvis" / "robot" / "simulator.py",
)


def test_tool_execution_code_contains_no_dynamic_execution_or_shell_calls():
    forbidden_names = {"eval", "exec", "compile", "__import__", "open"}
    forbidden_attributes = {"system", "popen", "run", "call", "check_call", "check_output"}

    for path in SECURITY_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_names, f"forbidden call in {path.name}"
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attributes, f"forbidden call in {path.name}"


def test_registry_never_reflects_model_tool_names_into_methods():
    source = (ROOT / "jarvis" / "tools" / "registry.py").read_text(encoding="utf-8")

    assert "getattr(" not in source
    assert "globals(" not in source
    assert "locals(" not in source
    assert "import_module" not in source


def test_ollama_adapter_never_parses_tool_calls_from_assistant_text():
    source = (ROOT / "jarvis" / "llm" / "ollama.py").read_text(encoding="utf-8")

    assert "json.loads" not in source
    assert "literal_eval" not in source
    assert "response.text" not in source
    assert '_field(message, "tool_calls"' in source


def test_llm_tool_registry_cannot_change_tts_or_speaker_configuration():
    names = {definition.name for definition in RobotToolRegistry().definitions}
    forbidden = {
        "set_voice",
        "set_tts_provider",
        "enable_tts",
        "disable_tts",
        "set_speaker",
        "play_audio",
    }
    assert names.isdisjoint(forbidden)
