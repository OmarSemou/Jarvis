# Phase 2C1.1 local hearing and benchmarking

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
powershell -ExecutionPolicy Bypass -File scripts/setup_whisper_windows.ps1 -Models base
```

The installer pins:

- `whisper.cpp` v1.9.1 official Windows x64 CPU archive;
- multilingual `ggml-base.bin` and `ggml-small.bin` at immutable upstream
  revision `c521a4b02f422512d734391fdf08bb08c0862f68`.

The official GitHub release API publishes the binary archive SHA-256. The
Hugging Face upstream file metadata publishes the model SHA-256. The script
checks both values before use, cleans partial downloads on failure, and can be
rerun. It writes only beneath ignored `data/tools/`, `data/models/`, and
`data/downloads/` paths. It does not require administrator privileges or
modify global `PATH`.

The default setup installs the selected base model. `-Models small` installs or
verifies the higher-capacity alternative without deleting base;
`-Models base,small` and `-Models all` install both. Jarvis startup never invokes
this installer or downloads a model.

## Commands

```text
/mic list    list input device metadata without opening a stream
/mic status  show the configured/default input
/mic use N   select an input for the current chat session only
/talk        record until Enter, transcribe, then use normal conversation flow
/stt status  show provider version, selected model, backend, language, readiness
```

Use the LLM-free integration check when diagnosing audio:

```powershell
.\.venv\Scripts\python.exe -m jarvis stt-check
.\.venv\Scripts\python.exe -m jarvis stt-check --mic 20
```

It reports audio duration, transcription duration, and real-time factor (RTF).
RTF is transcription time divided by audio duration; values below 1.0 are
faster than real time.

## Base/small benchmark

Both candidates are multilingual. Base is being evaluated for responsive
realtime commands; small remains available as the higher-capacity alternative.
Run the LLM-free, network-free comparison only after both models are installed:

```powershell
.\.venv\Scripts\python.exe -m jarvis stt-benchmark
.\.venv\Scripts\python.exe -m jarvis stt-benchmark --mic 20
```

The command records the fixed six-English/five-Danish corpus once. It then runs
base and small sequentially over the exact same resolved WAV paths, first as a
cold-ish pass and again as an OS-file-cache repeat. Temporary recordings are
deleted in a `finally` cleanup; use `--retain-recordings` only when developer
inspection is intentional.

The table includes recognized text, audio duration, both wall-clock
transcription times, warm-pass RTF, and exact/normalized matching. Per-model
summaries include mean, median, fastest, slowest, total matches, and English and
Danish matches. The benchmark also parses whisper.cpp's documented
`whisper_print_timings` diagnostics for the first phrase in each pass.

This is still process-per-command: every phrase starts a new `whisper-cli`
process and reloads the model. The second pass benefits from OS file caching but
does not represent a persistent in-memory Whisper service. The reported load,
encode, decode, total, and outside-reported-total values help separate the
rough sources of the approximately fixed short-clip latency.

## Configuration

`input_device` accepts a non-negative index, exact/unique device-name fragment,
or `null` for the Windows default input. `input_sample_rate` optionally requests
a capture rate. Final Whisper input is always mono PCM16 at 16 kHz.

`stt_model` defaults to `base` and accepts only `base` or `small`; it maps to a repository-owned
multilingual model path beneath `data/models/whisper/`. Arbitrary model paths
are not part of the configuration or any model-facing input.

`stt_language` defaults to `auto` and currently accepts `auto`, `en`, or `da`.
`stt_timeout_seconds` bounds the external process. `stt_use_gpu` defaults to
false. `whisper_executable_path` is resolved from repository-rooted
configuration, never from model or transcript output. Phase 2C1.1 is CPU-only;
Vulkan is deliberately deferred.

`retain_recordings` defaults to false. With that default, a recording is
deleted after successful or failed transcription. If explicitly enabled, the
WAV remains only beneath ignored `data/recordings/`. Raw audio is never logged.

## Failure behavior

Missing dependencies, devices, executable/model files, empty recordings,
timeouts, non-zero Whisper exits, and empty transcripts return concise CLI
errors without entering the conversation service. Whisper temporary output is
cleaned in all normal error paths. Startup never downloads or runs Whisper.
