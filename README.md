# Jarvis

Jarvis is becoming a fully local, API-free personal companion robot. This
repository is currently at **Phase 1: platform-aware architecture and safety
contracts**. It is not yet a working Windows voice assistant or physical robot.

The project is derived from
[Be More Agent](https://github.com/brenpoly/be-more-agent), an MIT-licensed
offline-first Raspberry Pi assistant. The upstream license and history are
preserved; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

## What Phase 1 provides

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

The existing `agent.py` remains a compatibility launcher. Its audio, wake-word,
Whisper, Piper, Ollama, camera, memory, and GUI implementations have not been
modularized in Phase 1 and their external dependencies are intentionally not
installed by the Phase 1 requirements.

## Development environment

The active development target is Windows 11 with Python 3.13. Project metadata
requires `>=3.13,<3.14`, preventing accidental use of Python 3.14.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Run the read-only diagnostic with:

```powershell
.\.venv\Scripts\python.exe -m jarvis.core.preflight
```

The diagnostic performs no downloads, installations, network requests,
subprocess execution, or hardware probing. Future components may legitimately
appear as missing during Phase 1.

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

Unknown keys and legacy aliases are handled deliberately by
`jarvis.core.config`. Runtime data is written beneath ignored `data/`, not into
tracked source locations.

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

## Planned work—not current capability

Later phases will add a Windows local-conversation slice, coordinated audio,
Whisper.cpp speech recognition, Piper and/or Kokoro speech synthesis,
OpenWakeWord, SQLite memory, optional explicit web lookup, camera/vision
providers, a simulated robot, and finally ESP32-backed physical subsystems.
Raspberry Pi deployment comes after desktop and simulator validation.

Ollama and the locally available `qwen3:8b` model are deliberately not
integrated during Phase 1.

## Legacy files

`setup.sh`, `start_agent.sh`, and `be-more-agent.desktop` are retained unchanged
for upstream history and later Linux migration. They are Raspberry-Pi/Linux
specific and are not the Windows Phase 1 setup path.

Existing faces, sounds, and `wakeword.onnx` are also retained unchanged. Their
licensing/provenance is not fully resolved by the repository MIT license; see
[NOTICE.md](NOTICE.md) before redistributing them.
