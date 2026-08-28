# Jarvis

Jarvis is becoming a fully local, API-free personal companion robot. The
repository now includes the **Phase 1 architecture and safety foundation** and
**Phase 2A local text conversation** for Windows 11. It is not yet a voice
assistant or physical robot.

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

The existing `agent.py` remains a compatibility launcher. Its audio, wake-word,
Whisper, Piper, Ollama, camera, memory, and GUI implementations have not been
modularized. The new text path does not use the legacy `BotGUI` class.

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

The CLI supports `/status`, `/reset`, `/think on`, `/think off`, and `/quit`.
The explicit integration check performs one small local inference:

```powershell
.\.venv\Scripts\python.exe -m jarvis llm-check
```

Run the read-only diagnostic with:

```powershell
.\.venv\Scripts\python.exe -m jarvis.core.preflight
```

The diagnostic remains read-only: it performs no downloads, installations,
network requests, subprocess execution, model inference, or hardware probing.

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
- `ollama_host`, `ollama_connect_timeout_seconds`
- `ollama_read_timeout_seconds`, `ollama_keep_alive`

Unknown keys and legacy aliases are handled deliberately by
`jarvis.core.config`. Runtime data is written beneath ignored `data/`, not into
tracked source locations. The new Ollama endpoint defaults to
`http://127.0.0.1:11434`; only HTTP loopback addresses are accepted. The adapter
disables environment proxy discovery and does not silently honor `OLLAMA_HOST`.

`system_prompt` and `system_prompt_extras` participate in Phase 2A as an
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

## Not implemented yet

Voice/audio, wake word, camera/vision, web search, persistent memory, robot tool
calling, and physical movement are not part of Phase 2A. Later phases will add
those capabilities incrementally, beginning with simulator-safe high-level
robot tools before physical controllers. Raspberry Pi deployment comes after
desktop and simulator validation.

## Legacy files

`setup.sh`, `start_agent.sh`, and `be-more-agent.desktop` are retained unchanged
for upstream history and later Linux migration. They are Raspberry-Pi/Linux
specific and are not the Windows Phase 1 setup path.

Existing faces, sounds, and `wakeword.onnx` are also retained unchanged. Their
licensing/provenance is not fully resolved by the repository MIT license; see
[NOTICE.md](NOTICE.md) before redistributing them.
