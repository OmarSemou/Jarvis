from jarvis.core.paths import JarvisPaths
from jarvis.core.config import JarvisConfig
from jarvis.audio.devices import AudioDevice, MicrophoneStatus
from jarvis.audio.tts.playback import OutputDevice, SpeakerStatus
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
    assert _by_name(report, "Piper package").level is RequirementLevel.CURRENT
    assert _by_name(report, "sounddevice").status is CheckStatus.MISSING
    assert _by_name(report, "Microphone").status is CheckStatus.NOT_CHECKED
    assert _by_name(report, "Camera support").status is CheckStatus.MISSING


def test_preflight_recognizes_windows_bundled_whisper_executable(tmp_path):
    paths = _repository(tmp_path)
    whisper = paths.whisper_executable_candidates[0]
    whisper.parent.mkdir(parents=True)
    whisper.write_text("", encoding="utf-8")

    report = run_preflight(
        paths,
        which=lambda name: "C:/Program Files/Ollama/ollama.exe" if name == "ollama.exe" else None,
        module_available=lambda _name: False,
        version_info=(3, 13, 1),
        platform_name="win32",
    )

    assert _by_name(report, "Whisper.cpp").path == whisper
    assert _by_name(report, "Piper package").status is CheckStatus.MISSING
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
    assert "Jarvis Phase 2C3.1 preflight (read-only)" in rendered
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

    def speaker_probe(configured):
        return SpeakerStatus(
            True,
            OutputDevice(7, "USB Speakers", 2, 48_000, True),
            configured,
            "default output",
        )

    report = run_preflight(
        paths,
        which=lambda _name: None,
        module_available=lambda name: name == "sounddevice",
        version_info=(3, 13, 15),
        platform_name="win32",
        microphone_probe=probe,
        speaker_probe=speaker_probe,
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
    assert _by_name(report, "Speaker").status is CheckStatus.AVAILABLE
    assert "no sound played" in _by_name(report, "Speaker").detail


def test_preflight_reports_voice_assets_without_loading_models_or_opening_mic(tmp_path):
    paths = _repository(tmp_path)
    for path in (
        paths.wakeword_classifier_model,
        paths.wakeword_melspectrogram_model,
        paths.wakeword_embedding_model,
        paths.vad_model,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")
    paths.kokoro_model.parent.mkdir(parents=True, exist_ok=True)
    paths.kokoro_model.write_bytes(b"model")
    paths.kokoro_voices.write_bytes(b"voices")
    whisper = paths.whisper_executable_candidates[0]
    whisper.parent.mkdir(parents=True, exist_ok=True)
    whisper.write_bytes(b"exe")
    paths.whisper_model_for("base").parent.mkdir(parents=True, exist_ok=True)
    paths.whisper_model_for("base").write_bytes(b"model")

    def microphone_probe(configured):
        return MicrophoneStatus(
            True, AudioDevice(1, "Mic", 1, 48_000, True), configured, "default"
        )

    def speaker_probe(configured):
        return SpeakerStatus(
            True, OutputDevice(2, "Speaker", 2, 48_000, True), configured, "default"
        )

    report = run_preflight(
        paths,
        which=lambda name: "C:/Ollama/ollama.exe" if name == "ollama.exe" else None,
        module_available=lambda name: name in {"openwakeword", "sounddevice", "kokoro_onnx"},
        version_info=(3, 13, 15),
        config=JarvisConfig(voice_mode_enabled=True, tts_enabled=True),
        microphone_probe=microphone_probe,
        speaker_probe=speaker_probe,
    )

    assert _by_name(report, "OpenWakeWord").status is CheckStatus.AVAILABLE
    assert _by_name(report, "Wake-word model").status is CheckStatus.AVAILABLE
    assert _by_name(report, "VAD model").status is CheckStatus.AVAILABLE
    assert _by_name(report, "Voice mode").status is CheckStatus.AVAILABLE
    assert _by_name(report, "Microphone").status is CheckStatus.AVAILABLE


def test_preflight_reports_tts_packages_assets_and_selected_voice_without_synthesis(tmp_path):
    paths = _repository(tmp_path)
    paths.kokoro_model.parent.mkdir(parents=True)
    paths.kokoro_model.write_bytes(b"model")
    paths.kokoro_voices.write_bytes(b"voices")
    model, voice_config = paths.piper_voice_files("en_US-john-medium")
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    voice_config.write_text("{}", encoding="utf-8")

    report = run_preflight(
        paths,
        which=lambda _name: None,
        module_available=lambda name: name in {"kokoro_onnx", "piper"},
        version_info=(3, 13, 15),
        config=JarvisConfig(tts_provider="piper", tts_voice="en_US-john-medium"),
    )

    assert _by_name(report, "kokoro-onnx").status is CheckStatus.AVAILABLE
    assert _by_name(report, "Piper package").status is CheckStatus.AVAILABLE
    assert _by_name(report, "Kokoro model").status is CheckStatus.AVAILABLE
    selected = _by_name(report, "Piper en_US-john-medium model")
    assert selected.status is CheckStatus.AVAILABLE
    assert "selected" in selected.detail
    assert _by_name(report, "Speaker").status is CheckStatus.NOT_CHECKED
