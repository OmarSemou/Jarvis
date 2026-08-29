"""Read-only, platform-aware Jarvis Phase 2C3.2 diagnostics."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from jarvis.audio.devices import MicrophoneDeviceService, MicrophoneStatus
from jarvis.audio.tts.playback import AudioPlaybackService, SpeakerStatus

from .config import ConfigValidationError, JarvisConfig, load_for_paths
from .paths import JarvisPaths
from jarvis.personality.profile import ACTIVE_ROBOT_NAME


class RequirementLevel(StrEnum):
    REQUIRED = "required"
    DEVELOPMENT = "development"
    CURRENT = "current/optional"
    FUTURE = "future/optional"


class CheckStatus(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    NOT_CHECKED = "not checked"


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    level: RequirementLevel
    status: CheckStatus
    detail: str
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def required_ok(self) -> bool:
        return all(
            check.status is CheckStatus.AVAILABLE
            for check in self.checks
            if check.level is RequirementLevel.REQUIRED
        )


WhichFunction = Callable[[str], str | None]
ExistsFunction = Callable[[Path], bool]
ModuleFunction = Callable[[str], bool]
MicrophoneProbe = Callable[[int | str | None], MicrophoneStatus]
SpeakerProbe = Callable[[int | str | None], SpeakerStatus]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _microphone_probe(configured: int | str | None) -> MicrophoneStatus:
    return MicrophoneDeviceService(configured).status()


def _speaker_probe(configured: int | str | None) -> SpeakerStatus:
    return AudioPlaybackService(configured).status()


def _first_file(candidates: Iterable[Path], exists: ExistsFunction) -> Path | None:
    return next((candidate for candidate in candidates if exists(candidate)), None)


def _executable_check(
    name: str,
    level: RequirementLevel,
    command_names: Sequence[str],
    bundled_candidates: Sequence[Path],
    *,
    which: WhichFunction,
    exists: ExistsFunction,
) -> PreflightCheck:
    bundled = _first_file(bundled_candidates, exists)
    if bundled is not None:
        return PreflightCheck(name, level, CheckStatus.AVAILABLE, "local executable found", bundled)
    for command in command_names:
        located = which(command)
        if located:
            return PreflightCheck(name, level, CheckStatus.AVAILABLE, "executable found on PATH", Path(located))
    return PreflightCheck(name, level, CheckStatus.MISSING, "executable not found")


def run_preflight(
    paths: JarvisPaths | None = None,
    *,
    which: WhichFunction = shutil.which,
    exists: ExistsFunction = Path.is_file,
    module_available: ModuleFunction = _module_available,
    version_info: tuple[int, int, int] | None = None,
    platform_name: str | None = None,
    environment: Mapping[str, str] | None = None,
    microphone_probe: MicrophoneProbe = _microphone_probe,
    speaker_probe: SpeakerProbe = _speaker_probe,
    config: JarvisConfig | None = None,
) -> PreflightReport:
    """Inspect local availability without streams, subprocesses, or network I/O."""

    paths = paths or JarvisPaths.discover()
    version = version_info or (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    platform_value = platform_name or sys.platform
    environment_values = os.environ if environment is None else environment
    checks: list[PreflightCheck] = []
    if config is None:
        try:
            config = load_for_paths(paths).config
        except ConfigValidationError:
            config = JarvisConfig()

    python_ok = version[:2] == (3, 13)
    checks.append(
        PreflightCheck(
            "Python",
            RequirementLevel.REQUIRED,
            CheckStatus.AVAILABLE if python_ok else CheckStatus.MISSING,
            f"{version[0]}.{version[1]}.{version[2]} detected; {ACTIVE_ROBOT_NAME} requires >=3.13,<3.14",
            Path(sys.executable),
        )
    )

    required_files = (paths.repository_root / "LICENSE", paths.repository_root / "agent.py")
    missing_layout = [path.name for path in required_files if not exists(path)]
    checks.append(
        PreflightCheck(
            "Repository layout",
            RequirementLevel.REQUIRED,
            CheckStatus.MISSING if missing_layout else CheckStatus.AVAILABLE,
            "missing: " + ", ".join(missing_layout) if missing_layout else "LICENSE and compatibility launcher found",
            paths.repository_root,
        )
    )

    # Memory diagnostics are strictly read-only: this branch never opens or
    # initializes SQLite and therefore cannot create data during preflight.
    try:
        memory_path = paths.memory_database_path(config.memory_database_path)
        if not config.memory_enabled:
            memory_detail = "disabled by configuration; no database will be opened"
            memory_status = CheckStatus.AVAILABLE
        elif exists(memory_path):
            memory_detail = "configured SQLite database exists; schema not opened during preflight"
            memory_status = CheckStatus.AVAILABLE
        else:
            memory_detail = "configured SQLite database will be created on explicit memory initialization"
            memory_status = CheckStatus.NOT_CHECKED
    except (OSError, ValueError) as exc:
        memory_path = None
        memory_detail = f"invalid memory path: {exc}"
        memory_status = CheckStatus.MISSING
    checks.append(
        PreflightCheck(
            "Memory",
            RequirementLevel.CURRENT,
            memory_status,
            memory_detail,
            memory_path,
        )
    )

    checks.append(
        PreflightCheck(
            "pytest",
            RequirementLevel.DEVELOPMENT,
            CheckStatus.AVAILABLE if module_available("pytest") else CheckStatus.MISSING,
            "development test dependency" if module_available("pytest") else "install the development extra",
        )
    )
    ollama_candidates: tuple[Path, ...] = ()
    if platform_value.startswith("win"):
        local_app_data = environment_values.get("LOCALAPPDATA")
        program_files = environment_values.get("ProgramFiles")
        ollama_candidates = tuple(
            candidate
            for candidate in (
                Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe" if local_app_data else None,
                Path(local_app_data) / "Ollama" / "ollama.exe" if local_app_data else None,
                Path(program_files) / "Ollama" / "ollama.exe" if program_files else None,
            )
            if candidate is not None
        )
    checks.append(
        _executable_check(
            "Ollama",
            RequirementLevel.CURRENT,
            ("ollama.exe", "ollama") if platform_value.startswith("win") else ("ollama", "ollama.exe"),
            ollama_candidates,
            which=which,
            exists=exists,
        )
    )
    configured_whisper = paths.resolve_from_root(config.whisper_executable_path)
    whisper_candidates = tuple(
        dict.fromkeys((configured_whisper, *paths.whisper_executable_candidates))
    )
    checks.append(
        _executable_check(
            "Whisper.cpp",
            RequirementLevel.CURRENT,
            ("whisper-cli.exe", "whisper-cli") if platform_value.startswith("win") else ("whisper-cli", "whisper-cli.exe"),
            whisper_candidates,
            which=which,
            exists=exists,
        )
    )
    for model_name in ("base", "small"):
        model_path = paths.whisper_model_for(model_name)
        selected = model_name == config.stt_model
        found = exists(model_path)
        checks.append(
            PreflightCheck(
                f"Whisper model ({model_name})",
                RequirementLevel.CURRENT,
                CheckStatus.AVAILABLE if found else CheckStatus.MISSING,
                (
                    f"multilingual model found ({'selected' if selected else 'benchmark candidate'})"
                    if found
                    else f"multilingual model not found ({'selected' if selected else 'benchmark candidate'})"
                ),
                model_path,
            )
        )
    checks.append(
        PreflightCheck(
            "kokoro-onnx",
            RequirementLevel.CURRENT,
            CheckStatus.AVAILABLE if module_available("kokoro_onnx") else CheckStatus.MISSING,
            "local CPU TTS provider package installed"
            if module_available("kokoro_onnx")
            else "run scripts/setup_tts_windows.ps1 -Providers kokoro",
        )
    )
    checks.append(
        PreflightCheck(
            "Piper package",
            RequirementLevel.CURRENT,
            CheckStatus.AVAILABLE if module_available("piper") else CheckStatus.MISSING,
            "local OHF Piper TTS package installed"
            if module_available("piper")
            else "run scripts/setup_tts_windows.ps1 -Providers piper",
        )
    )
    for name, path in (
        ("Kokoro model", paths.kokoro_model),
        ("Kokoro voice bundle", paths.kokoro_voices),
    ):
        found = exists(path)
        checks.append(
            PreflightCheck(
                name,
                RequirementLevel.CURRENT,
                CheckStatus.AVAILABLE if found else CheckStatus.MISSING,
                "pinned local asset found" if found else "pinned local asset missing",
                path,
            )
        )
    for voice in ("en_US-joe-medium", "en_US-john-medium"):
        model_path, voice_config = paths.piper_voice_files(voice)
        for suffix, path in (("model", model_path), ("config", voice_config)):
            found = exists(path)
            selected = config.tts_provider == "piper" and config.tts_voice == voice
            checks.append(
                PreflightCheck(
                    f"Piper {voice} {suffix}",
                    RequirementLevel.CURRENT,
                    CheckStatus.AVAILABLE if found else CheckStatus.MISSING,
                    (
                        f"pinned local asset found ({'selected' if selected else 'benchmark candidate'})"
                        if found
                        else f"pinned local asset missing ({'selected' if selected else 'benchmark candidate'})"
                    ),
                    path,
                )
            )
    bmo_model, bmo_config = paths.legacy_bmo_voice_files
    for suffix, path in (("model", bmo_model), ("config", bmo_config)):
        found = exists(path)
        selected = config.tts_profile == "bmo"
        checks.append(
            PreflightCheck(
                f"Piper BMO {suffix}",
                RequirementLevel.CURRENT,
                CheckStatus.AVAILABLE if found else CheckStatus.MISSING,
                (
                    f"original legacy asset found ({'selected' if selected else 'optional'})"
                    if found
                    else f"original legacy asset missing ({'selected' if selected else 'optional'})"
                ),
                path,
            )
        )
    openwakeword_available = module_available("openwakeword")
    checks.append(
        PreflightCheck(
            "OpenWakeWord",
            RequirementLevel.CURRENT,
            CheckStatus.AVAILABLE if openwakeword_available else CheckStatus.MISSING,
            (
                "local ONNX wake/VAD package installed; no inference performed"
                if openwakeword_available
                else "run scripts/setup_voice_windows.ps1"
            ),
        )
    )
    wake_assets = (
        (
            "Wake-word model",
            paths.wakeword_classifier_model,
            "pinned official OpenWakeWord hey-Jarvis classifier found (private development only)",
        ),
        (
            "Wake-word mel model",
            paths.wakeword_melspectrogram_model,
            "pinned OpenWakeWord feature model found",
        ),
        (
            "Wake-word embedding model",
            paths.wakeword_embedding_model,
            "pinned OpenWakeWord feature model found",
        ),
        (
            "VAD model",
            paths.vad_model,
            "pinned local Silero ONNX model found",
        ),
    )
    for name, path, found_detail in wake_assets:
        found = exists(path)
        checks.append(
            PreflightCheck(
                name,
                RequirementLevel.CURRENT,
                CheckStatus.AVAILABLE if found else CheckStatus.MISSING,
                found_detail if found else "local model missing; run scripts/setup_voice_windows.ps1",
                path,
            )
        )
    voice_assets_ready = openwakeword_available and all(
        exists(path) for _, path, _ in wake_assets
    )
    voice_config_ready = all(
        (
            config.voice_mode_enabled,
            config.wakeword_enabled,
            config.vad_enabled,
            config.tts_enabled,
        )
    )
    sounddevice_available = module_available("sounddevice")
    checks.append(
        PreflightCheck(
            "sounddevice",
            RequirementLevel.CURRENT,
            CheckStatus.AVAILABLE if sounddevice_available else CheckStatus.MISSING,
            "local audio input/output dependency installed"
            if sounddevice_available
            else "install project requirements for microphone and speaker support",
        )
    )
    if sounddevice_available:
        microphone = microphone_probe(config.input_device)
        microphone_check = PreflightCheck(
            "Microphone",
            RequirementLevel.CURRENT,
            CheckStatus.AVAILABLE if microphone.available else CheckStatus.MISSING,
            (
                f"{microphone.selected.index}: {microphone.selected.name} ({microphone.detail})"
                if microphone.available and microphone.selected is not None
                else microphone.detail
            ),
        )
    else:
        microphone_check = PreflightCheck(
            "Microphone",
            RequirementLevel.CURRENT,
            CheckStatus.NOT_CHECKED,
            "not inspected because sounddevice is unavailable",
        )
    checks.append(microphone_check)
    if sounddevice_available:
        speaker = speaker_probe(config.output_device)
        speaker_check = PreflightCheck(
            "Speaker",
            RequirementLevel.CURRENT,
            CheckStatus.AVAILABLE if speaker.available else CheckStatus.MISSING,
            (
                f"{speaker.selected.index}: {speaker.selected.name} ({speaker.detail}); no sound played"
                if speaker.available and speaker.selected is not None
                else speaker.detail
            ),
        )
    else:
        speaker_check = PreflightCheck(
            "Speaker",
            RequirementLevel.CURRENT,
            CheckStatus.NOT_CHECKED,
            "not inspected because sounddevice is unavailable",
        )
    checks.append(speaker_check)
    check_status = {check.name: check.status for check in checks}
    if config.tts_profile == "bmo":
        selected_tts_checks = (
            "Piper package",
            "Piper BMO model",
            "Piper BMO config",
        )
    elif config.tts_profile == "fenrir" or config.tts_provider == "kokoro":
        selected_tts_checks = ("kokoro-onnx", "Kokoro model", "Kokoro voice bundle")
    else:
        selected_tts_checks = (
            "Piper package",
            f"Piper {config.tts_voice} model",
            f"Piper {config.tts_voice} config",
        )
    required_voice_checks = (
        "Ollama",
        "Whisper.cpp",
        f"Whisper model ({config.stt_model})",
        "sounddevice",
        "Microphone",
        "Speaker",
        *selected_tts_checks,
    )
    complete_runtime_ready = all(
        check_status.get(name) is CheckStatus.AVAILABLE
        for name in required_voice_checks
    )
    voice_ready = voice_assets_ready and voice_config_ready and complete_runtime_ready
    checks.append(
        PreflightCheck(
            "Voice mode",
            RequirementLevel.CURRENT,
            CheckStatus.AVAILABLE if voice_ready else CheckStatus.MISSING,
            (
                "configured; local LLM/STT/TTS/wake/VAD assets and audio devices are ready; no stream or inference run"
                if voice_ready
                else "not ready: inspect missing current voice dependencies, assets, devices, or private enable flags above"
            ),
        )
    )
    camera_backend = module_available("cv2") or bool(which("rpicam-still"))
    checks.append(
        PreflightCheck(
            "Camera support",
            RequirementLevel.FUTURE,
            CheckStatus.NOT_CHECKED if camera_backend else CheckStatus.MISSING,
            "camera backend found; hardware intentionally not probed" if camera_backend else "no camera backend found; hardware not probed",
        )
    )
    return PreflightReport(tuple(checks))


def format_report(report: PreflightReport) -> str:
    lines = [f"{ACTIVE_ROBOT_NAME} Phase 2C3.2 preflight (read-only)", ""]
    for check in report.checks:
        path = f" [{check.path}]" if check.path is not None else ""
        lines.append(f"{check.status.value:11} {check.level.value:15} {check.name}: {check.detail}{path}")
    lines.extend(("", "Required checks: " + ("PASS" if report.required_ok else "FAIL")))
    return "\n".join(lines)


def main() -> int:
    report = run_preflight()
    print(format_report(report))
    return 0 if report.required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
