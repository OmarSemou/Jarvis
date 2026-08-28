# Jarvis architecture

## Phase 1 boundary

Phase 1 introduces side-effect-free configuration, path, application-state,
preflight, robot-intent, and safety contracts. The upstream `agent.py` remains
the compatibility launcher. Audio, wake word, speech recognition, speech
synthesis, Ollama conversation, camera capture, memory storage, and Tkinter
orchestration remain legacy code until later phases.

Importing `jarvis.core` or `jarvis.robot` must not start a GUI, open audio or
camera devices, invoke subprocesses, perform network requests, or write files.

## Long-term authority path

```text
LLM
  -> typed high-level tool call
  -> policy and validation
  -> deterministic SafetySupervisor
  -> RobotController
  -> transport
  -> ESP32
```

Safety inputs follow a separate deterministic path:

```text
sensors / watchdog / emergency-stop input
  -> SafetySupervisor
```

The LLM will only be able to request meaning-level actions such as `wave`,
`look_at_user`, `move_forward`, or `stop`. Robot intents contain no PWM, motor
voltage, duty cycle, current, raw wheel speed, arbitrary servo angle, GPIO,
serial, or transport fields.

The safety supervisor produces an explicit approval or denial. Approved
physical movement receives a short renewable lease. A future controller and
ESP32 must stop when that lease or the communication heartbeat expires.

## Modules

- `jarvis.core.config`: validated configuration and documented legacy keys.
- `jarvis.core.paths`: repository-rooted source and ignored runtime paths.
- `jarvis.core.state`: GUI-independent application/connection states.
- `jarvis.core.preflight`: read-only local availability diagnostics.
- `jarvis.robot.intents`: high-level semantic actions only.
- `jarvis.robot.safety`: deterministic fail-closed decisions and e-stop latch.
- `jarvis.robot.interfaces`: post-safety controller and movement-lease contracts.

Future modules will isolate audio, LLM, personality, memory, tools, face,
vision, integrations, simulated robot components, and ESP32 transport. None of
those later integrations are part of Phase 1.

