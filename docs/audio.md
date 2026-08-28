# Phase 2C2 local hearing, speech, and benchmarking

Jarvis currently hears only after an explicit developer command. `/talk`
starts one microphone recording, Enter stops it, and local `whisper.cpp`
transcribes it. There is no wake word, VAD, always-listening loop, global
hotkey or cloud speech service. When explicitly enabled, Jarvis speaks each
completed response through a fully local provider and the selected speaker.

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

## Local TTS architecture

```text
ConversationService response text
  -> CLI/application coordinator
  -> TTSService
  -> allowlisted TTSProvider
  -> provider-neutral PCM16 SynthesizedAudio
  -> AudioPlaybackService
  -> selected sounddevice RawOutputStream
```

`ConversationService` does not import TTS, Kokoro, Piper, or sounddevice. The
CLI prints the complete response before synthesis begins, then performs one
synchronous synthesize/play operation. There is no streaming, sentence
chunking, playback thread, interruption, or barge-in in Phase 2C2. The playback
interface nevertheless has an idempotent `stop`/`cancel` boundary for later use.

Normal response audio never touches disk. Both adapters return interleaved
signed little-endian PCM16 plus sample rate and channel count. Kokoro float
samples are clipped and converted deterministically; Piper's PCM16 chunks are
validated for a consistent format before concatenation. Playback accepts mono
or stereo and closes the stream on success or failure.

## TTS setup and versions

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_tts_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts/setup_tts_windows.ps1 -Providers kokoro
powershell -ExecutionPolicy Bypass -File scripts/setup_tts_windows.ps1 -Providers piper
```

The script pins `kokoro-onnx==0.6.1`, its stable `model-files-v1.0` float32 CPU
model and voice bundle, and `piper-tts==1.7.0` from the maintained Open Home
Foundation project. Piper voices are pinned to the immutable official voice
repository tag `v1.0.0`. Every one of the six model/config assets has a fixed
SHA-256 in the script. Existing valid files are reused; mismatches stop setup
unless the developer explicitly supplies `-Force`. Files live beneath ignored
`data/models/tts/`; startup has no download or install path.

The direct packages do not require Torch or Transformers. Kokoro's wrapper
depends on ONNX Runtime, NumPy, phonemizer, and its eSpeak-NG loader. Piper's
runtime depends on ONNX Runtime and pathvalidate. Phonemizer declares GPL-3.0,
so it is documented separately from the MIT Kokoro wrapper in `NOTICE.md`.
Phase 2C2 uses CPU inference.

## Voice and speaker controls

```text
/voice status
/voice on | /voice off
/voice provider kokoro | /voice provider piper
/voice use <allowlisted voice>
/speaker list | /speaker status
/speaker use <index or unique name>
```

The curated Kokoro candidates are `am_fenrir`, `am_michael`, `am_puck`, and
`bm_george`. Piper candidates are `en_US-joe-medium` and
`en_US-john-medium`. Provider switching selects a conservative provider-local
candidate, after which `/voice use` can refine it. These settings are
session-only unless supplied in validated configuration. An unavailable
package, model, voice, output, or synthesis/playback failure produces a concise
message while keeping the assistant text visible.

Phase 2C2 accepts English only. The official Piper catalog has Danish voices,
so the language boundary can support Danish later; the official Kokoro voice
set currently has no Danish voice. No Danish TTS model is installed here.

## TTS benchmark and retention

```powershell
.\.venv\Scripts\python.exe -m jarvis tts-benchmark
.\.venv\Scripts\python.exe -m jarvis tts-benchmark-clean "<run-directory>"
```

The benchmark constructs only the TTS adapters. It makes no LLM, STT, network,
microphone, or playback calls. It generates eight fixed phrases for all six
voices and writes labeled WAVs beneath one timestamped ignored directory. Each
row reports synthesis wall time, speech duration, real-time factor, sample
rate, first usable audio when the provider API exposes it, and output file.
Summaries report median/mean synthesis time, median RTF, fastest/slowest, and
short-utterance median.

For Kokoro's non-streaming API, “first usable audio” is the completed array and
therefore approximately equals total synthesis time. Piper yields chunks, so
the adapter records the first chunk. Neither provider is streamed to speakers
in this phase. Timings cannot decide voice quality; a human must listen for
tone, clarity, and pronunciation. Samples are retained deliberately until the
explicit guarded cleanup command removes one direct benchmark-run directory.
