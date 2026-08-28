from jarvis.core.paths import JarvisPaths
from jarvis.core.config import JarvisConfig
from jarvis.audio.devices import AudioDevice, MicrophoneStatus
from jarvis.core.preflight import (
    CheckStatus,
    RequirementLevel,
    format_report,
    run_preflight,
)


def _repository(tmp_path):
    (tmp_path / "LICENSE").write_text("license", encoding="utf-8")
    (tmp_path / "agent.py").write_text("pass", encoding="utf-8")
    return JarvisPaths.from_repository_root(tmp_path)


def _by_name(report, name):
    return next(check for check in report.checks if check.name == name)


def test_preflight_passes_required_checks_without_future_components(tmp_path):
    paths = _repository(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    report = run_preflight(
        paths,
        which=lambda _name: None,
        module_available=lambda name: name == "pytest",
        version_info=(3, 13, 15),
        platform_name="win32",
    )

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert report.required_ok
    assert before == after
    assert _by_name(report, "Whisper.cpp").status is CheckStatus.MISSING
    assert _by_name(report, "Piper").level is RequirementLevel.FUTURE
    assert _by_name(report, "sounddevice").status is CheckStatus.MISSING
    assert _by_name(report, "Microphone").status is CheckStatus.NOT_CHECKED
    assert _by_name(report, "Camera support").status is CheckStatus.MISSING


def test_preflight_recognizes_windows_bundled_executables(tmp_path):
    paths = _repository(tmp_path)
    whisper = paths.whisper_executable_candidates[0]
    piper = paths.piper_executable_candidates[0]
    whisper.parent.mkdir(parents=True)
    piper.parent.mkdir(parents=True)
    whisper.write_text("", encoding="utf-8")
    piper.write_text("", encoding="utf-8")

    report = run_preflight(
        paths,
        which=lambda name: "C:/Program Files/Ollama/ollama.exe" if name == "ollama.exe" else None,
        module_available=lambda _name: False,
        version_info=(3, 13, 1),
        platform_name="win32",
    )

    assert _by_name(report, "Whisper.cpp").path == whisper
    assert _by_name(report, "Piper").path == piper
    assert _by_name(report, "Ollama").status is CheckStatus.AVAILABLE


def test_preflight_recognizes_standard_windows_ollama_location(tmp_path):
    paths = _repository(tmp_path)
    local_app_data = tmp_path / "LocalAppData"
    ollama = local_app_data / "Programs" / "Ollama" / "ollama.exe"
    ollama.parent.mkdir(parents=True)
    ollama.write_text("", encoding="utf-8")

    report = run_preflight(
        paths,
        which=lambda _name: None,
        module_available=lambda _name: False,
        version_info=(3, 13, 1),
        platform_name="win32",
        environment={"LOCALAPPDATA": str(local_app_data)},
    )

    assert _by_name(report, "Ollama").status is CheckStatus.AVAILABLE
    assert _by_name(report, "Ollama").path == ollama


def test_python_314_fails_required_version_check(tmp_path):
    report = run_preflight(
        _repository(tmp_path),
        which=lambda _name: None,
        module_available=lambda _name: False,
        version_info=(3, 14, 0),
    )

    assert not report.required_ok
    assert _by_name(report, "Python").status is CheckStatus.MISSING


def test_report_is_understandable_and_labels_future_items(tmp_path):
    report = run_preflight(
        _repository(tmp_path),
        which=lambda _name: None,
        module_available=lambda _name: False,
        version_info=(3, 13, 0),
    )

    rendered = format_report(report)
    assert "Jarvis Phase 2C1.1 preflight (read-only)" in rendered
    assert "future/optional" in rendered
    assert "Required checks: PASS" in rendered


def test_preflight_reports_configured_whisper_model_and_microphone_without_opening_stream(tmp_path):
    paths = _repository(tmp_path)
    executable = tmp_path / "private" / "whisper-cli.exe"
    model = paths.whisper_model_for("base")
    executable.parent.mkdir()
    model.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    probed = []

    def probe(configured):
        probed.append(configured)
        return MicrophoneStatus(
            True,
            AudioDevice(4, "USB Mic", 1, 48_000, True),
            configured,
            "configured input",
        )

    report = run_preflight(
        paths,
        which=lambda _name: None,
        module_available=lambda name: name == "sounddevice",
        version_info=(3, 13, 15),
        platform_name="win32",
        microphone_probe=probe,
        config=JarvisConfig(
            input_device=4,
            whisper_executable_path="private/whisper-cli.exe",
            stt_model="base",
        ),
    )

    assert probed == [4]
    assert _by_name(report, "Whisper.cpp").path == executable.resolve()
    assert _by_name(report, "Whisper model (base)").status is CheckStatus.AVAILABLE
    assert "selected" in _by_name(report, "Whisper model (base)").detail
    assert _by_name(report, "Whisper model (small)").status is CheckStatus.MISSING
    assert _by_name(report, "sounddevice").status is CheckStatus.AVAILABLE
    assert _by_name(report, "Microphone").status is CheckStatus.AVAILABLE
