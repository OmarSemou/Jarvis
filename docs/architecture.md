# Jarvis architecture

## Current boundary

Phase 1 introduced side-effect-free configuration, path, application-state,
preflight, robot-intent, and safety contracts. Phase 2A added local text
conversation through Ollama. Phase 2B adds native structured tool calling and
a deterministic safe robot simulator. Phase 2C1.1 adds configurable microphone
capture, local speech recognition, and comparative STT benchmarking. Phase
2C2 adds provider-neutral local speech synthesis, speaker playback, and a
retained listening benchmark. The upstream `agent.py` remains the compatibility
launcher. Wake word, streaming/interruption, camera capture, memory
storage, physical hardware, and Tkinter orchestration remain outside the new
path until later phases.

Importing any `jarvis` module must not start a GUI, open devices, invoke
subprocesses, perform network requests, or write files. Ollama transport is
created only by an explicit CLI command, and requests are sent only by chat or
`llm-check` actions. `sounddevice` is loaded only when an explicit microphone
command runs, and `whisper.cpp` is invoked only after a completed recording, an
explicit `stt-check`, or the explicit local `stt-benchmark`.

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

## Phase 2C1.1 local hearing path

```text
/talk (or explicit stt-check)
  -> MicrophoneDeviceService (read-only selection)
  -> PushToTalkRecorder (explicit start/stop)
  -> mono PCM16 WAV normalized to 16000 Hz
  -> STTProvider
  -> WhisperCppSTT
  -> configured local whisper-cli executable
  -> allowlisted multilingual base or small model path
  -> provider-neutral TranscriptionResult
```

For chat, only the final transcription crosses into the existing AI path:

```text
TranscriptionResult.text
  -> ConversationService.respond(text)
  -> existing LLM/tool/policy/safety/simulator flow
```

`ConversationService` has no audio or whisper.cpp dependency. Typed input and
voice input therefore cannot diverge in tool authority, safety policy, or
history behavior.

The recorder requests mono PCM16 directly. It first tries the configured input
rate, then 16 kHz, then the selected device's default rate. When direct 16 kHz
capture is unavailable, a deterministic audio-layer resampler produces the
required 16 kHz WAV; there is no hidden ffmpeg dependency.

The Whisper adapter uses an explicit argument vector with `shell=False`, a
bounded timeout, captured stdout/stderr, return-code checks, and the documented
`--output-txt`/`--output-file` mechanism. Transcripts are read from the exact
generated file—not inferred from console lines. The executable comes from
validated local configuration. The symbolic
`stt_model` value is restricted to `base` or `small` and maps to a fixed local
path; arbitrary model paths are not accepted. Phase 2C1.1 passes `--no-gpu` and
uses the official CPU release. A future backend executable can be selected in
configuration without changing `ConversationService`.

The benchmark path is separate from conversation:

```text
stt-benchmark
  -> record fixed bilingual corpus once
  -> retain WAV paths for this command only
  -> base then small, sequentially, on each identical WAV
  -> repeat as an OS-file-cache pass
  -> aggregate wall time, RTF, normalized matches, and documented Whisper timings
  -> delete captured WAV files unless explicitly retained
```

It creates neither `ConversationService` nor an LLM/network client. Normal
`/talk` continues to pass `--no-prints`; only the benchmark omits that flag and
parses known `whisper_print_timings` fields. Every call starts a subprocess and
reloads its model. No persistent STT server was introduced.

Recordings are uniquely named beneath ignored `data/recordings/` and deleted
after success or failure by default. Whisper output uses a unique ignored
`data/stt/` working directory and is always cleaned up. `retain_recordings=true`
is the only supported retention opt-in.

## Phase 2C2 local speech-output path

```text
final ConversationService response text
  -> CLI coordinator prints text
  -> TTSService selects one allowlisted local provider/voice
  -> KokoroTTS or PiperTTS (CPU ONNX)
  -> provider-neutral SynthesizedAudio (PCM16/rate/channels)
  -> AudioPlaybackService selects a local output
  -> sounddevice RawOutputStream
```

This dependency flows in one direction. `ConversationService` remains unaware
of speech synthesis and playback. The LLM and robot tool layers cannot select
a provider, voice, model path, speaker, or generated audio. TTS consumes only
the already-final assistant text and holds no robot or SafetySupervisor
reference. Speech therefore has no route to tools, motion, heartbeat, or
emergency-stop authority.

Both providers load packages and models lazily on the first explicit synthesis.
Playback loads sounddevice and opens a stream only for explicit device commands
or actual speech. Importing modules has no device, inference, network,
subprocess, or filesystem side effects. Normal response audio remains in
memory. The explicit benchmark is separate from conversation and playback:

```text
tts-benchmark
  -> fixed eight-phrase English corpus
  -> four Kokoro plus two Piper candidates, sequentially
  -> labeled retained WAV samples in ignored runtime storage
  -> per-sample and aggregate timing report
```

The benchmark makes no LLM or STT calls. A guarded explicit cleanup removes
one direct benchmark run, never the benchmark root or an arbitrary path.

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
- `jarvis.audio.devices`: lazy input-device enumeration and validated selection.
- `jarvis.audio.formats`: mono PCM16 resampling and WAV validation/writing.
- `jarvis.audio.recorder`: explicit push-to-talk recording lifecycle.
- `jarvis.audio.service`: STT coordination and private recording cleanup.
- `jarvis.audio.benchmark`: fixed-corpus comparison, scoring, aggregation, and cleanup.
- `jarvis.audio.stt.base`: provider-neutral transcription result/error contract.
- `jarvis.audio.stt.whisper_cpp`: bounded local subprocess adapter.
- `jarvis.audio.tts.base`: provider-neutral PCM16 and structured result/error contracts.
- `jarvis.audio.tts.kokoro`: lazy local `kokoro-onnx` CPU adapter.
- `jarvis.audio.tts.piper`: lazy local OHF Piper CPU adapter.
- `jarvis.audio.tts.playback`: lazy output discovery and synchronous PCM16 playback.
- `jarvis.audio.tts.service`: session selection and synthesize-then-play coordination.
- `jarvis.audio.tts.benchmark`: fixed-corpus local synthesis and guarded sample cleanup.
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

Future modules will add wake word, VAD, streaming/early TTS, interruption, memory, face,
vision, integrations, physical robot components, and ESP32 transport. None of
those integrations is part of Phase 2C2. Microphone/STT/TTS testing and the robot
simulator are not hardware safety validation.
