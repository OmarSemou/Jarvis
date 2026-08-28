"""Read-only, platform-aware Jarvis Phase 2C1.1 diagnostics."""

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

from .config import ConfigValidationError, JarvisConfig, load_for_paths
from .paths import JarvisPaths


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


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _microphone_probe(configured: int | str | None) -> MicrophoneStatus:
    return MicrophoneDeviceService(configured).status()


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
            f"{version[0]}.{version[1]}.{version[2]} detected; Jarvis requires >=3.13,<3.14",
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
        _executable_check(
            "Piper",
            RequirementLevel.FUTURE,
            ("piper.exe", "piper") if platform_value.startswith("win") else ("piper", "piper.exe"),
            paths.piper_executable_candidates,
            which=which,
            exists=exists,
        )
    )
    checks.append(
        PreflightCheck(
            "Wake-word model",
            RequirementLevel.FUTURE,
            CheckStatus.AVAILABLE if exists(paths.wakeword_model) else CheckStatus.MISSING,
            "model file found" if exists(paths.wakeword_model) else "model file not found; wake word is a future feature",
            paths.wakeword_model,
        )
    )
    sounddevice_available = module_available("sounddevice")
    checks.append(
        PreflightCheck(
            "sounddevice",
            RequirementLevel.CURRENT,
            CheckStatus.AVAILABLE if sounddevice_available else CheckStatus.MISSING,
            "local microphone dependency installed"
            if sounddevice_available
            else "install project requirements for microphone capture",
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
    lines = ["Jarvis Phase 2C1.1 preflight (read-only)", ""]
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
