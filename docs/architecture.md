# Jarvis architecture

## Current boundary

Phase 1 introduced side-effect-free configuration, path, application-state,
preflight, robot-intent, and safety contracts. Phase 2A added local text
conversation through Ollama. Phase 2B adds native structured tool calling and
a deterministic safe robot simulator. The upstream `agent.py` remains the compatibility
launcher. Audio, wake word, speech recognition, speech synthesis, camera
capture, memory storage, physical hardware, and Tkinter orchestration remain outside the new
path until later phases.

Importing any `jarvis` module must not start a GUI, open devices, invoke
subprocesses, perform network requests, or write files. Ollama transport is
created only by an explicit CLI command, and requests are sent only by chat or
`llm-check` actions.

## Phase 2B text and tool path

```text
python -m jarvis chat
  -> CLI
  -> ConversationService
  -> LLMProvider protocol
  -> OllamaLLM
  -> http://127.0.0.1:11434
  -> configured local model (qwen3:8b by default)
```

For native tool calls the return path is:

```text
Ollama native tool call
  -> OllamaLLM translates to Jarvis ToolCall
  -> RobotToolRegistry validates the exact name and arguments
  -> RobotToolPolicy applies batch and stop-precedence rules
  -> SafeRobotController
  -> deterministic SafetySupervisor
  -> ApprovedRobotIntent with movement lease
  -> SimulatedRobot
  -> Jarvis ToolResult
  -> OllamaLLM translates the provider-neutral result
  -> final natural-language response
```

`ConversationService` owns complete in-memory turns, including assistant tool
calls and structured tool-result messages. It preserves the system message and
retains only the configured number of recent user turns. A failure before any
action leaves history unchanged; if an action completed before a later model
failure, the truthful tool transcript is retained with a controlled marker.

The loop executes at most `conversation_max_tool_rounds` successful tool
batches per user message (default three). A denial immediately closes tool
execution for that message, so the model receives the denial but cannot retry.
The final provider request has no tool schemas. Tool calls returned despite a
closed loop never execute.

All calls in a batch are validated before non-stop execution. Valid non-
conflicting calls execute sequentially. A valid `stop` runs first and suppresses
every other physical-motion call in that batch. If any call is malformed or
unknown, no non-stop call in that batch executes.
Each batch is also capped at eight calls; an oversized batch executes nothing.

The system message is assembled predictably from three explicit sections:

1. immutable Phase 2B system/capability policy;
2. the structured Jarvis personality profile;
3. optional configured customization, marked as untrusted preference input.

The Ollama adapter owns transport details. It passes an explicit validated
loopback host, disables environment proxy discovery, applies bounded connect
and read timeouts, passes `think` and `keep_alive` as top-level request fields,
and maps expected transport failures into provider-neutral errors. It also owns
all translation between native Ollama tool objects and Jarvis types. It
contains no pull/download operation and streaming is deliberately deferred.

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
- `jarvis.tools.types`: provider-neutral tool definitions, calls, and results.
- `jarvis.tools.registry`: static semantic robot-tool allowlist and validation.
- `jarvis.tools.policy`: sequential batch policy and stop precedence.
- `jarvis.personality.profile`: immutable structured personality data.
- `jarvis.personality.prompt`: policy/personality/customization prompt boundaries.
- `jarvis.cli`: developer text UI and explicit local integration check.
- `jarvis.robot.intents`: high-level semantic actions only.
- `jarvis.robot.safety`: deterministic fail-closed decisions and e-stop latch.
- `jarvis.robot.interfaces`: post-safety controller and movement-lease contracts.
- `jarvis.robot.controller`: safety-gated semantic simulator controller.
- `jarvis.robot.simulator`: deterministic in-memory robot state and event log.

Future modules will isolate audio, memory, face, vision, integrations, physical
robot components, and ESP32 transport. None of those integrations is part of
Phase 2B. The simulator is not hardware safety validation.
