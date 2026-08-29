# Jarvis

Jarvis is becoming a fully local, API-free personal companion robot. The
repository now includes the **Phase 1 architecture and safety foundation**,
**Phase 2A local text conversation**, **Phase 2B structured robot tools with a
deterministic simulator**, **Phase 2C3.2 responsive local voice interaction**,
and **Phase 2D animated face prototype**
for Windows 11. Continuous listening is an explicitly enabled local wake-word
mode; Jarvis does not control a physical robot.

The project is derived from
[Be More Agent](https://github.com/brenpoly/be-more-agent), an MIT-licensed
offline-first Raspberry Pi assistant. The upstream license and history are
preserved; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

## What works now

- Validated JSON configuration with explicit legacy-key migration.
- Repository-rooted paths that do not depend on the current working directory.
- Ignored `data/` paths for private configuration, memory, recordings, and
  camera captures.
- GUI-independent application state definitions.
- A read-only Windows-aware preflight diagnostic.
- Typed, meaning-level robot intent contracts with no low-level motor fields.
- A deterministic, fail-closed safety supervisor and renewable movement-lease
  contract.
- Unit and contract tests that require no external programs or hardware.
- Upstream attribution and documentation of unresolved third-party licensing.
- A provider-independent, in-memory conversation service.
- Local text conversation with `qwen3:8b` through a loopback-only Ollama adapter.
- Thinking control that defaults to off for lower latency.
- Separate immutable policy, personality, and bounded customization sections.
- A developer text CLI and an explicit local integration check.
- Native structured Ollama tool calling translated into Jarvis-owned types.
- An explicit 16-tool allowlist containing only semantic robot actions.
- A bounded tool loop with deterministic validation and stop precedence.
- A safety-gated controller and hardware-free simulated robot state machine.
- Trusted developer-only simulator status and emergency-stop commands.
- Lazy, Windows-compatible microphone discovery and deterministic device
  selection through `sounddevice`.
- Explicit developer push-to-talk recording to mono PCM16 16 kHz WAV.
- Provider-neutral local speech-to-text through a configured `whisper.cpp`
  process using an allowlisted multilingual Whisper `base` or `small` model.
- Non-STOP voice transcripts routed into the same conversation, structured-
  tool, safety, and simulator path as typed text; explicit voice STOP uses the
  narrower deterministic controller path.
- Private recording deletion by default plus LLM-free `stt-check` and bilingual
  `stt-benchmark` commands.
- Provider-neutral, fully local English TTS through pinned CPU builds of
  `kokoro-onnx` or Open Home Foundation Piper.
- In-memory PCM16 playback through a session-selectable Windows speaker,
  including a cancellable voice-mode handle and text-only failure fallback.
- An LLM-free, STT-free, playback-free `tts-benchmark` command covering four
  Kokoro and two Piper voice candidates.
- A deterministic continuous voice coordinator with explicit wake, listening,
  processing, speaking, interruption, error, and shutdown states.
- Local OpenWakeWord activation, ONNX-only Silero VAD endpointing, controlled
  Kokoro `am_fenrir` preload, cancellable playback, and structured latency data.
- Wake-word-gated playback interruption by default, with the former generic
  VAD interruption available only as `vad_experimental`.
- Deterministic blank-audio rejection and a one-command local STOP router that
  bypasses Qwen while retaining the controller and safety boundary.
- TTS-only Markdown normalization: terminal output and conversation history
  retain formatting while Kokoro/Piper receive plain speakable text.
- Semantic voice profiles keep Fenrir (`kokoro/am_fenrir`) as the default and
  optionally restore the original upstream BMO Piper model when its historical
  `voices/bmo-custom.onnx` assets are supplied.
- Deterministic sentence-level speech chunks, Kokoro 0.6.1 audio streaming, and
  a two-chunk bounded producer/consumer queue that begins playback with the
  first usable PCM rather than waiting for complete response synthesis.
- Monotonic speech-generation identities and wake-barge cancellation that abort
  current playback, clear queued PCM, and discard late provider output.
- Lossless bounded pre-roll for both normal wake activation and wake-gated
  playback interruption, with score and frame-continuity diagnostics.
- A conservative, configurable Ollama conversation temperature of `0.2`.
- An observation-only, resizable Tk face using the unchanged BMO prototype
  PNGs, with lifecycle animation and generation-safe playback state.

The existing `agent.py` remains a compatibility launcher. Phase 2C3.2 does not
reuse its legacy last-stdout-line Whisper parsing or GUI audio thread. Wake
word, camera, memory, and GUI implementations there have not been
modularized. The new chat/hearing path does not use the legacy `BotGUI` class.

## Development environment

The active development target is Windows 11 with Python 3.13. Project metadata
requires `>=3.13,<3.14`, preventing accidental use of Python 3.14.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Install and start Ollama separately, then install the model manually if needed:

```powershell
ollama pull qwen3:8b
```

Jarvis never pulls or downloads a model automatically. Start local text chat
with:

```powershell
.\.venv\Scripts\python.exe -m jarvis chat
```

The CLI supports `/status`, `/reset`, `/think on`, `/think off`,
`/robot status`, `/robot estop`, `/robot estop-reset`, `/talk`, `/stt status`, `/mic list`,
`/mic status`, `/mic use <device>`, `/voice status`, `/voice on`, `/voice off`,
`/voice provider <kokoro|piper>`, `/voice use <voice>`, `/speaker list`,
`/speaker status`, `/speaker use <device>`, and `/quit`.
Robot actions print concise developer events such as `[ROBOT] gesture=wave`.
The `/robot estop-reset` command is trusted local CLI control and is not in the
LLM tool registry.

After running the explicit voice setup and enabling voice mode in private
configuration, start local hands-free interaction with:

```powershell
.\.venv\Scripts\python.exe -m jarvis voice
.\.venv\Scripts\python.exe -m jarvis voice --debug-latency
.\.venv\Scripts\python.exe -m jarvis face
.\.venv\Scripts\python.exe -m jarvis face-demo --gallery
.\.venv\Scripts\python.exe -m jarvis voice --face
```

The active classifier is the pinned official OpenWakeWord **Hey Jarvis** model
downloaded into ignored local runtime storage. Saying only “Jarvis” can have a
higher miss rate. After each response Jarvis returns to requiring the wake
phrase; it does not leave the room microphone open to Whisper. With
`--debug-latency`, a one-second rolling wake-score peak is also printed while
idle so microphone/model problems are visible before a trigger.
The explicit integration check performs one small local inference:

```powershell
.\.venv\Scripts\python.exe -m jarvis llm-check
```

### Local hearing setup (Windows)

Run the explicit installer yourself; Jarvis startup never downloads Whisper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_whisper_windows.ps1
```

The default command installs the official `whisper.cpp` v1.9.1 Windows x64 CPU
build and the selected multilingual `ggml-base.bin` beneath ignored `data/`
storage. Install or verify base explicitly without removing small with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_whisper_windows.ps1 -Models base
```

Use `-Models small` for the higher-capacity alternative, or `-Models base,small`
(or `-Models all`) when setting up both. The binary and
both model sources are pinned and SHA-256 verified. Valid files are not
redownloaded. The script requires no administrator access, does not change
global `PATH`, and is safe to rerun.

List or inspect microphone inputs with `/mic list` and `/mic status`. A
session-only selection can be made with `/mic use <index or unique name>`. Use
`/talk`, speak, and press Enter to stop. The transcript then enters the exact
same `ConversationService` used by typed input. This is a developer-mode
push-to-talk workflow and remains available independently of continuous voice
mode.

Install the minimal continuous-voice dependency and hash-verified ONNX feature
models explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_voice_windows.ps1
```

The script installs pinned `openwakeword==0.6.0` in `.venv` and downloads the
pinned official Hey Jarvis classifier plus the mel, embedding, and Silero VAD
ONNX assets beneath ignored `data/`. It does not install PyTorch. Normal
startup does not install or download anything. The tracked legacy
`wakeword.onnx` remains untouched and is not used by continuous voice mode.

Test microphone capture and transcription without Ollama or an LLM:

```powershell
.\.venv\Scripts\python.exe -m jarvis stt-check
```

For a one-off check with a non-default input, add `--mic 20` (or a unique
device-name fragment). This does not persist configuration.

Compare base and small on the fixed six-English/five-Danish phrase set with:

```powershell
.\.venv\Scripts\python.exe -m jarvis stt-benchmark
```

Each phrase is recorded once and the same WAV is used for both models over a
cold-ish first pass and OS-file-cache repeat. Recordings are deleted after the
run; `--retain-recordings` is an explicit developer opt-in. This command never
constructs an LLM or performs network/download work. Each transcription still
starts `whisper-cli` and reloads its model, so reported latency includes that
process-per-command design.

Run the read-only diagnostic with:

```powershell
.\.venv\Scripts\python.exe -m jarvis.core.preflight
```

The diagnostic remains read-only: it performs no downloads, installations,
network requests, subprocess execution, model inference, or recording-stream
activation. When `sounddevice` is installed it enumerates input-device metadata
and output-device metadata to report the configured/default microphone and
speaker, but it does not record or play sound.

### Local speech setup and benchmark (Windows)

Install or verify both pinned CPU providers and their curated English voices:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_tts_windows.ps1
```

Use `-Providers kokoro` or `-Providers piper` to set up one provider. The script
installs only into `.venv`, verifies every downloaded model/config with a pinned
SHA-256, refuses to replace a hash-mismatched local asset unless `-Force` is
explicit, and never modifies global `PATH`. Normal Jarvis startup does not
install or download anything.

Generate labeled listening samples and timing summaries without Ollama,
Whisper, microphone access, speaker playback, or network access:

```powershell
.\.venv\Scripts\python.exe -m jarvis tts-benchmark
```

The 48 curated WAV files remain under ignored `data/benchmarks/tts/<timestamp>/`
so a human can compare voice quality; an installed legacy BMO model adds eight
more labeled samples. Remove exactly one run later with:

```powershell
.\.venv\Scripts\python.exe -m jarvis tts-benchmark-clean "<run-directory>"
```

Speech remains disabled in tracked example configuration. The semantic voice
profiles are Fenrir (the default Kokoro `am_fenrir`) and the original legacy
BMO Piper model. In chat, use `/voice on` to enable output, `/voice fenrir` or
`/voice bmo` to switch profiles, and `/voice status` to inspect the mapping.
The continuous command also accepts `--voice fenrir` or `--voice bmo`. BMO is
available only when the original `voices/bmo-custom.onnx` and JSON config have
been supplied; the app never downloads an imitation. Both typed and `/talk`
responses are spoken only after the complete text response is visible.
Continuous mode uses cancellable background playback and requires the local
**Hey Jarvis** wake detector—not generic VAD alone—to authorize barge-in by
default. Synthesized audio remains in memory and is not retained.
See [docs/audio.md](docs/audio.md) for provider and latency details.

## Configuration

`config.example.json` documents the current schema and contains no secrets.
The compatibility launcher reads private `data/config.json` when present and
otherwise falls back to the tracked legacy `config.json`.

Supported fields are:

- `text_model`, `vision_model`, `voice_model`
- `chat_memory`
- `camera_rotation`
- `system_prompt`, `system_prompt_extras`
- `input_device`, `input_sample_rate`
- `llm_model`, `llm_thinking`, `llm_temperature` (default `0.2`)
- `conversation_max_turns`
- `conversation_max_tool_rounds` (default `3`)
- `ollama_host`, `ollama_connect_timeout_seconds`
- `ollama_read_timeout_seconds`, `ollama_keep_alive`
- `whisper_executable_path`, `stt_model` (`base` or `small` only)
- `stt_language` (`auto`, `en`, or `da`; default `auto`)
- `stt_timeout_seconds`, `stt_use_gpu` (default `false` in Phase 2C1.1)
- `retain_recordings` (default `false`)
- `output_device` (default `null`, meaning the system default output)
- `tts_enabled` (tracked default `false`; continuous mode requires it enabled)
- `tts_profile` (`fenrir` or `bmo`; omitted preserves legacy provider/voice settings)
- `tts_provider` (`kokoro` or `piper`, retained compatibility setting)
- `tts_voice` (Fenrir maps to Kokoro `am_fenrir`; BMO maps to the original
  custom Piper model)
- `tts_speed` (`0.5` through `2.0`), `tts_language` (`en` in Phase 2C3)
- `voice_mode_enabled`, `wakeword_enabled`, `wakeword_threshold`
- `vad_enabled`, `vad_speech_threshold`, `vad_trailing_silence_ms`
- `vad_max_utterance_seconds`, `vad_min_speech_ms`, `vad_listen_timeout_seconds`
- `barge_in_enabled`, `barge_in_mode` (`wakeword` by default or explicitly
  experimental `vad_experimental`), `barge_in_threshold`, `barge_in_suppression_ms`
- `barge_in_min_speech_ms`, `barge_in_pre_roll_ms` (default `320`)
- `barge_in_command_start_timeout_seconds` (default `1.5`), `tts_preload`,
  `voice_debug_latency`
- `voice_ollama_keep_alive` (default `30m` for an active voice session)

Unknown keys and legacy aliases are handled deliberately by
`jarvis.core.config`. Runtime data is written beneath ignored `data/`, not into
tracked source locations. The new Ollama endpoint defaults to
`http://127.0.0.1:11434`; only HTTP loopback addresses are accepted. The adapter
disables environment proxy discovery and does not silently honor `OLLAMA_HOST`.

`stt_model` defaults to `base`. On the Phase 2C1.1 Windows benchmark it retained
the tested English command accuracy while reducing warm median transcription
latency from about 3.56 seconds (`small`) to 1.12 seconds. Small remains an
explicit higher-capacity alternative. Both candidates require further Danish
testing because automatic language detection performed poorly on this short
corpus.

`system_prompt` and `system_prompt_extras` participate in Phase 2B as an
explicitly bounded customization section. They cannot override immutable
policy or grant capabilities. The tracked legacy `config.json` is retained for
compatibility; put new private settings in ignored `data/config.json`.

## Safety architecture

The intended authority path is:

```text
LLM
  -> typed high-level tool request
  -> policy and validation
  -> deterministic SafetySupervisor
  -> RobotController
  -> transport
  -> ESP32
```

An LLM is never a safety authority. A future physical emergency stop will
disable motor enable/power independently of software, networking, and the
desktop computer. See [docs/architecture.md](docs/architecture.md) and
[docs/safety.md](docs/safety.md).

Phase 2B uses explicit synthetic clear/fresh/ready sensor inputs for desktop
simulation. Every robot tool becomes a high-level `RobotIntent`, passes the
existing `SafetySupervisor`, and reaches the simulator only as an
`ApprovedRobotIntent`. An e-stop blocks base and expressive simulated motion;
`stop` and the screen-only `set_expression` action remain available.

The simulator demonstrates software architecture and test behavior only. It is
not hardware safety validation and says nothing about real braking distance,
electrical faults, motor drivers, sensor coverage, watchdogs, or emergency-stop
circuits.

TTS receives only final response or fixed local acknowledgement text in the CLI
coordinator. It
cannot call tools, mutate the simulator, refresh safety heartbeats, clear an
e-stop, or bypass robot policy. Wake-word barge-in only cancels TTS playback.
After STT, an anchored local grammar recognizes only explicit STOP utterances;
it sends the existing semantic STOP intent through `SafeRobotController` and
`SafetySupervisor` without asking Qwen. Every other valid transcript uses the
normal `ConversationService` and structured-tool path. No SafetySupervisor
behavior changed in Phase 2C3.2. Sentence synthesis starts only after
`ConversationService.respond()` has completed its bounded native tool loop and
returned the final user-visible response, so speech cannot race or contradict a
later safety decision. Ollama token streaming is deliberately not used for the
initial tool-enabled request in this phase.

Voice stop is not the physical emergency stop. A future robot still requires
an ESP32 watchdog, local motor-command timeout, and a physical e-stop/power
disable independent of the desktop, Wi-Fi, speech stack, and LLM.

## Privacy and local execution

Microphone recordings use unique names beneath ignored `data/recordings/`.
They are deleted after successful or failed transcription unless
`retain_recordings` is explicitly set to `true`. Temporary Whisper output is
also deleted. Jarvis does not log raw audio or invoke a cloud speech service.
Normal synthesized response audio stays in memory and is discarded after
playback. Idle wake/VAD frames and wake buffers are never written. Only a VAD-
accepted utterance becomes a unique temporary WAV for process-per-command
Whisper, and the existing `retain_recordings=false` cleanup remains the default.
Only an explicit `tts-benchmark` retains generated WAVs,
under ignored runtime storage, until an explicit cleanup command. The Phase
2C3 speech stack is CPU/ONNX-only; `stt_use_gpu` remains false. Vulkan and
other acceleration backends remain deferred until a later,
explicit acceleration decision.

## Not implemented yet

Full acoustic echo cancellation, streaming STT, LLM token streaming, camera/
vision, web search, persistent memory, ESP32 communication,
motors, servos, and physical movement are not part of Phase 2D. Raspberry Pi
deployment comes after desktop and simulator validation.

## Legacy files

`setup.sh`, `start_agent.sh`, and `be-more-agent.desktop` are retained unchanged
for upstream history and later Linux migration. They are Raspberry-Pi/Linux
specific and are not the Windows Phase 1 setup path.

Existing faces, sounds, and `wakeword.onnx` are retained unchanged. Phase 2D
uses the original BMO face/image assets as a private, read-only prototype face;
they are not the final Jarvis design and must not be redistributed until their
individual provenance/licensing is resolved. See [docs/face.md](docs/face.md)
and [NOTICE.md](NOTICE.md).
