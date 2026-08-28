# Jarvis

Jarvis is becoming a fully local, API-free personal companion robot. The
repository now includes the **Phase 1 architecture and safety foundation** and
**Phase 2A local text conversation**, and **Phase 2B structured robot tools with
a deterministic simulator** for Windows 11. **Phase 2C1.1 adds configurable
multilingual push-to-talk hearing and a local whisper.cpp benchmark.** Jarvis does not yet speak, listen
continuously, or control a physical robot.

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
- Voice transcripts routed into the same conversation, structured-tool, safety,
  and simulator path as typed text.
- Private recording deletion by default plus LLM-free `stt-check` and bilingual
  `stt-benchmark` commands.

The existing `agent.py` remains a compatibility launcher. Phase 2C1.1 does not
reuse its legacy last-stdout-line Whisper parsing or GUI audio thread. Wake
word, Piper, camera, memory, and GUI implementations there have not been
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
`/mic status`, `/mic use <device>`, and `/quit`.
Robot actions print concise developer events such as `[ROBOT] gesture=wave`.
The `/robot estop-reset` command is trusted local CLI control and is not in the
LLM tool registry.
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
push-to-talk workflow—not a wake word, VAD, global hotkey, or always-listening
mode.

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
to report the configured/default microphone, but it does not record.

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
- `llm_model`, `llm_thinking`, `conversation_max_turns`
- `conversation_max_tool_rounds` (default `3`)
- `ollama_host`, `ollama_connect_timeout_seconds`
- `ollama_read_timeout_seconds`, `ollama_keep_alive`
- `whisper_executable_path`, `stt_model` (`base` or `small` only)
- `stt_language` (`auto`, `en`, or `da`; default `auto`)
- `stt_timeout_seconds`, `stt_use_gpu` (default `false` in Phase 2C1.1)
- `retain_recordings` (default `false`)

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

## Privacy and local execution

Microphone recordings use unique names beneath ignored `data/recordings/`.
They are deleted after successful or failed transcription unless
`retain_recordings` is explicitly set to `true`. Temporary Whisper output is
also deleted. Jarvis does not log raw audio or invoke a cloud speech service.
The Phase 2C1.1 build is CPU-only; `stt_use_gpu` remains false. Vulkan and
other acceleration backends remain deferred until a later,
explicit acceleration decision.

## Not implemented yet

TTS/speech playback, wake word, VAD, barge-in/interruption, face GUI,
camera/vision, web search, persistent memory, ESP32 communication, motors,
servos, and physical movement are not part of Phase 2C1.1. Raspberry Pi
deployment comes after desktop and simulator validation.

## Legacy files

`setup.sh`, `start_agent.sh`, and `be-more-agent.desktop` are retained unchanged
for upstream history and later Linux migration. They are Raspberry-Pi/Linux
specific and are not the Windows Phase 1 setup path.

Existing faces, sounds, and `wakeword.onnx` are also retained unchanged. Their
licensing/provenance is not fully resolved by the repository MIT license; see
[NOTICE.md](NOTICE.md) before redistributing them.
