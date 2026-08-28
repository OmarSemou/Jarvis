import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_LICENSE_SHA256 = "c526d38e33848313360f2007fd9377f38e47336042da55e166e214d70f7425ff"


def test_upstream_license_is_byte_for_byte_unchanged():
    digest = hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest()
    assert digest == UPSTREAM_LICENSE_SHA256


def test_agent_remains_a_compatibility_launcher():
    tree = ast.parse((ROOT / "agent.py").read_text(encoding="utf-8"))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    imported_modules = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("jarvis.core")
        for alias in node.names
    }
    has_main_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    )

    assert "BotGUI" in classes
    assert {"load_for_paths", "JarvisPaths", "BotStates"} <= imported_modules
    assert has_main_guard


def test_foundation_modules_do_not_import_side_effect_integrations():
    files = [
        ROOT / "jarvis/core/config.py",
        ROOT / "jarvis/core/paths.py",
        ROOT / "jarvis/core/state.py",
        ROOT / "jarvis/core/preflight.py",
        ROOT / "jarvis/robot/intents.py",
        ROOT / "jarvis/robot/interfaces.py",
        ROOT / "jarvis/robot/safety.py",
    ]
    forbidden = {
        "tkinter",
        "sounddevice",
        "openwakeword",
        "ollama",
        "cv2",
        "PIL",
        "requests",
        "socket",
        "subprocess",
    }

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        assert imports.isdisjoint(forbidden), f"{path.name} imports {imports & forbidden}"


def test_project_targets_python_313_only():
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.13,<3.14"' in metadata

