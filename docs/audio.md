# Phase 2C1 local hearing

Jarvis currently hears only after an explicit developer command. `/talk`
starts one microphone recording, Enter stops it, and local `whisper.cpp`
transcribes it. There is no wake word, VAD, always-listening loop, global
hotkey, TTS, or cloud speech service.

## Windows setup

Install Python dependencies into the project virtual environment, then run the
explicit Whisper installer:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/setup_whisper_windows.ps1
```

The installer pins:

- `whisper.cpp` v1.9.1 official Windows x64 CPU archive;
- multilingual `ggml-small.bin` at immutable upstream revision
  `c521a4b02f422512d734391fdf08bb08c0862f68`.

The official GitHub release API publishes the binary archive SHA-256. The
Hugging Face upstream file metadata publishes the model SHA-256. The script
checks both values before use, cleans partial downloads on failure, and can be
rerun. It writes only beneath ignored `data/tools/`, `data/models/`, and
`data/downloads/` paths. It does not require administrator privileges or
modify global `PATH`.

## Commands

```text
/mic list    list input device metadata without opening a stream
/mic status  show the configured/default input
/mic use N   select an input for the current chat session only
/talk        record until Enter, transcribe, then use normal conversation flow
```

Use the LLM-free integration check when diagnosing audio:

```powershell
.\.venv\Scripts\python.exe -m jarvis stt-check
.\.venv\Scripts\python.exe -m jarvis stt-check --mic 20
```

It reports audio duration, transcription duration, and real-time factor (RTF).
RTF is transcription time divided by audio duration; values below 1.0 are
faster than real time.

## Configuration

`input_device` accepts a non-negative index, exact/unique device-name fragment,
or `null` for the Windows default input. `input_sample_rate` optionally requests
a capture rate. Final Whisper input is always mono PCM16 at 16 kHz.

`stt_language` defaults to `auto` and currently accepts `auto`, `en`, or `da`.
`stt_timeout_seconds` bounds the external process. `stt_use_gpu` defaults to
false. `whisper_executable_path` and `whisper_model_path` are resolved from
repository-rooted configuration, never from model or transcript output.

`retain_recordings` defaults to false. With that default, a recording is
deleted after successful or failed transcription. If explicitly enabled, the
WAV remains only beneath ignored `data/recordings/`. Raw audio is never logged.

## Failure behavior

Missing dependencies, devices, executable/model files, empty recordings,
timeouts, non-zero Whisper exits, and empty transcripts return concise CLI
errors without entering the conversation service. Whisper temporary output is
cleaned in all normal error paths. Startup never downloads or runs Whisper.
