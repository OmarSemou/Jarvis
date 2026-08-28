# Jarvis architecture

## Current boundary

Phase 1 introduced side-effect-free configuration, path, application-state,
preflight, robot-intent, and safety contracts. Phase 2A adds only local text
conversation through Ollama. The upstream `agent.py` remains the compatibility
launcher. Audio, wake word, speech recognition, speech synthesis, camera
capture, memory storage, tools, and Tkinter orchestration remain outside the new
path until later phases.

Importing any `jarvis` module must not start a GUI, open devices, invoke
subprocesses, perform network requests, or write files. Ollama transport is
created only by an explicit CLI command, and requests are sent only by chat or
`llm-check` actions.

## Phase 2A text path

```text
python -m jarvis chat
  -> CLI
  -> ConversationService
  -> LLMProvider protocol
  -> OllamaLLM
  -> http://127.0.0.1:11434
  -> configured local model (qwen3:8b by default)
```

`ConversationService` owns complete in-memory user/assistant turns. It always
preserves the single system message and retains only the configured number of
recent complete turns. Failed or interrupted requests are not appended.

The system message is assembled predictably from three explicit sections:

1. immutable Phase 2A system/capability policy;
2. the structured Jarvis personality profile;
3. optional configured customization, marked as untrusted preference input.

The Ollama adapter owns transport details. It passes an explicit validated
loopback host, disables environment proxy discovery, applies bounded connect
and read timeouts, passes `think` and `keep_alive` as top-level request fields,
and maps expected transport failures into provider-neutral errors. It contains
no pull/download operation and streaming is deliberately deferred.

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
- `jarvis.core.conversation`: provider-independent in-memory turn orchestration.
- `jarvis.llm.base`: provider-neutral messages, responses, cancellation, and errors.
- `jarvis.llm.ollama`: explicit loopback-only Ollama transport adapter.
- `jarvis.personality.profile`: immutable structured personality data.
- `jarvis.personality.prompt`: policy/personality/customization prompt boundaries.
- `jarvis.cli`: developer text UI and explicit local integration check.
- `jarvis.robot.intents`: high-level semantic actions only.
- `jarvis.robot.safety`: deterministic fail-closed decisions and e-stop latch.
- `jarvis.robot.interfaces`: post-safety controller and movement-lease contracts.

Future modules will isolate audio, memory, tools, face, vision, integrations,
simulated robot components, and ESP32 transport. None of those integrations is
part of Phase 2A.
