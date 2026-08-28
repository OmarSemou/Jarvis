import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.audio.formats import write_pcm16_mono_wav
from jarvis.audio.stt.base import TranscriptionErrorCode
from jarvis.audio.stt.whisper_cpp import WhisperCppSTT, WhisperCppSettings


def make_provider(tmp_path, runner, **overrides):
    executable = tmp_path / "tools" / "whisper-cli.exe"
    model = tmp_path / "models" / "ggml-small.bin"
    executable.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    executable.write_bytes(b"exe")
    model.write_bytes(b"model")
    values = {
        "executable_path": executable.resolve(),
        "model_path": model.resolve(),
        "temp_dir": (tmp_path / "data" / "stt").resolve(),
        "language": "auto",
        "timeout_seconds": 30,
        "use_gpu": False,
    }
    values.update(overrides)
    return WhisperCppSTT(WhisperCppSettings(**values), process_runner=runner)


def make_audio(tmp_path):
    audio = (tmp_path / "data" / "recordings" / "input.wav").resolve()
    audio.parent.mkdir(parents=True)
    write_pcm16_mono_wav(audio, b"\x00\x00" * 1_600)
    return audio


def output_prefix_from(command):
    return Path(command[command.index("--output-file") + 1])


def test_command_is_explicit_cpu_only_and_uses_no_shell(tmp_path):
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        output_prefix_from(command).with_suffix(".txt").write_text("  Look right.\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ignored last line", stderr="")

    provider = make_provider(tmp_path, runner)
    audio = make_audio(tmp_path)
    result = provider.transcribe(audio)

    assert result.success
    assert result.text == "Look right."
    command, kwargs = calls[0]
    assert command[0] == str(provider.settings.executable_path)
    assert command[command.index("--model") + 1] == str(provider.settings.model_path)
    assert command[command.index("--file") + 1] == str(audio)
    assert "--output-txt" in command
    assert "--no-gpu" in command
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 30
    assert not list(provider.settings.temp_dir.iterdir())


def test_transcript_comes_from_deterministic_output_file_not_stdout(tmp_path):
    def runner(command, **_kwargs):
        output_prefix_from(command).with_suffix(".txt").write_text("File result", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="[timestamp] Wrong stdout", stderr="")

    result = make_provider(tmp_path, runner).transcribe(make_audio(tmp_path))
    assert result.text == "File result"


def test_whisper_silence_marker_is_an_empty_transcript(tmp_path):
    def runner(command, **_kwargs):
        output_prefix_from(command).with_suffix(".txt").write_text(" [ Silence ] \n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = make_provider(tmp_path, runner).transcribe(make_audio(tmp_path))
    assert result.error.code is TranscriptionErrorCode.EMPTY_TRANSCRIPT


@pytest.mark.parametrize(
    ("missing", "code"),
    [
        ("executable", TranscriptionErrorCode.EXECUTABLE_MISSING),
        ("model", TranscriptionErrorCode.MODEL_MISSING),
    ],
)
def test_missing_setup_files_return_structured_error_and_setup_command(tmp_path, missing, code):
    provider = make_provider(tmp_path, lambda *_args, **_kwargs: None)
    target = (
        provider.settings.executable_path
        if missing == "executable"
        else provider.settings.model_path
    )
    target.unlink()

    result = provider.transcribe(make_audio(tmp_path))

    assert not result.success
    assert result.error.code is code
    assert "scripts/setup_whisper_windows.ps1" in result.error.message


def test_timeout_maps_to_structured_error_and_cleans_output(tmp_path):
    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    provider = make_provider(tmp_path, runner)
    result = provider.transcribe(make_audio(tmp_path))

    assert result.error.code is TranscriptionErrorCode.TIMEOUT
    assert not list(provider.settings.temp_dir.iterdir())


def test_nonzero_exit_maps_to_process_failure_without_stdout_dump(tmp_path):
    def runner(_command, **_kwargs):
        return SimpleNamespace(returncode=7, stdout="private transcript", stderr="model failed")

    result = make_provider(tmp_path, runner).transcribe(make_audio(tmp_path))
    assert result.error.code is TranscriptionErrorCode.PROCESS_FAILED
    assert "code 7" in result.error.message
    assert "private transcript" not in result.error.message


@pytest.mark.parametrize(
    ("write_output", "code"),
    [
        (False, TranscriptionErrorCode.OUTPUT_MISSING),
        (True, TranscriptionErrorCode.EMPTY_TRANSCRIPT),
    ],
)
def test_missing_or_empty_output_maps_cleanly(tmp_path, write_output, code):
    def runner(command, **_kwargs):
        if write_output:
            output_prefix_from(command).with_suffix(".txt").write_text(" \n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = make_provider(tmp_path, runner).transcribe(make_audio(tmp_path))
    assert result.error.code is code


def test_missing_audio_and_invalid_audio_are_rejected_before_process(tmp_path):
    calls = []
    provider = make_provider(tmp_path, lambda *args, **kwargs: calls.append((args, kwargs)))

    missing = provider.transcribe((tmp_path / "missing.wav").resolve())
    invalid_path = (tmp_path / "invalid.wav").resolve()
    invalid_path.write_text("not wav", encoding="utf-8")
    invalid = provider.transcribe(invalid_path)

    assert missing.error.code is TranscriptionErrorCode.AUDIO_MISSING
    assert invalid.error.code is TranscriptionErrorCode.INVALID_AUDIO
    assert calls == []


def test_settings_require_validated_absolute_paths_and_language(tmp_path):
    with pytest.raises(ValueError, match="absolute path"):
        WhisperCppSettings(Path("relative.exe"), tmp_path.resolve(), tmp_path.resolve())
    with pytest.raises(ValueError, match="language"):
        WhisperCppSettings(tmp_path.resolve(), tmp_path.resolve(), tmp_path.resolve(), language="fr")
