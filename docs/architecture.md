# Jarvis architecture

## Current boundary

Phase 1 introduced side-effect-free configuration, path, application-state,
preflight, robot-intent, and safety contracts. Phase 2A added local text
conversation through Ollama. Phase 2B adds native structured tool calling and
a deterministic safe robot simulator. Phase 2C1.1 adds configurable microphone
capture, local speech recognition, and comparative STT benchmarking. Phase
2C2 adds provider-neutral local speech synthesis, speaker playback, and a
retained listening benchmark. Phase 2C3 adds local wake/VAD inference,
deterministic voice states, bounded utterances, warmup, latency metrics, and
cancellable playback. Phase 2C3.1 makes barge-in wake-word-gated by default,
rejects blank STT turns, adds deterministic safety-routed local STOP, and
restores general conversational scope. Phase 2C3.2 adds provider-neutral
sentence chunking, bounded PCM lookahead, and early ordered playback after the
final tool-safe response is committed. The upstream `agent.py` remains the
compatibility launcher. Streaming STT/LLM text, full echo cancellation, camera
capture, memory storage, physical hardware, and Tkinter orchestration remain
outside the new path until later phases.

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
ordinary voice input therefore cannot diverge in tool authority, safety policy,
or history behavior. Phase 2C3.1 adds one deliberate exception before the
service: exact local STOP bypasses conversation history and the LLM but still
uses `SafeRobotController` and the same `SafetySupervisor`.

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

## Phase 2C3.1a lossless wake-barge handoff

```text
SoundDeviceRealtimeInput (fixed 30 ms, 16 kHz PCM16 frames)
  -> OpenWakeWord adapter (Hey Jarvis, local ONNX)
  -> VoiceStateMachine: IDLE -> WAKE_DETECTED -> LISTENING
  -> SileroVAD adapter returns probability only
  -> VADSegmenter applies deterministic start/end/max/no-speech policy
  -> VoiceInputService creates one private temporary WAV
  -> existing WhisperCppSTT process-per-command adapter
  -> exact blank/no-speech and wake-only transcript filter
  -> LocalVoiceCommandRouter
       -> STOP -> injected integration -> SafeRobotController/SafetySupervisor/simulator
       -> otherwise -> ConversationService.respond(transcript)
                    -> existing LLM/tool/policy/SafetySupervisor/simulator flow
  -> TTSService normalizes display Markdown and synthesizes Kokoro am_fenrir
  -> AudioPlaybackService background handle
  -> VoiceStateMachine: PROCESSING -> SPEAKING -> IDLE
```

Wake, VAD, realtime input, and state contracts are Jarvis-owned. OpenWakeWord,
Silero, sounddevice, and Ollama objects cannot cross their respective adapters.
Imports remain side-effect free; models and devices load only when the explicit
`voice` command starts. Preflight checks packages, paths, configuration, and
device metadata but opens no stream and runs no model inference.

The state set is `IDLE`, `WAKE_DETECTED`, `LISTENING`, `PROCESSING`, `SPEAKING`,
`INTERRUPTED`, `ERROR`, and `SHUTDOWN`. Transition edges are allowlisted and
tested. An error transitions through `ERROR` to safe `IDLE`; command shutdown
then reaches `SHUTDOWN`. After each successful response, continuation behavior
A returns to `IDLE` and requires the wake phrase again.

While `SPEAKING`, the default `wakeword` barge mode invokes the same local wake
provider against a strictly bounded 320 ms rolling buffer. Generic speech
probability cannot cancel playback. A confirmed wake phrase signals only the
TTS playback handle and hands the still-open microphone to VAD without a
drain, close, reopen, or playback-cancellation wait:

```text
SPEAKING -> wake event + playback cancel signal -> INTERRUPTED -> LISTENING
  -> buffered tail + same live stream -> VAD endpointing
  -> STT -> blank/wake-only filter -> local STOP or ConversationService
```

The newest buffered tail is deliberately shorter than the VAD speech-start
minimum, so it can preserve an immediate command onset but cannot authorize a
capture without at least one newly consumed live frame. The remaining frames
are passive pre-roll. A separate 1.5-second command-start timeout supports a
short natural pause after the wake phrase and returns silent attempts to
`IDLE`. Realtime frames carry monotonic capture timestamps and sequence values;
debug mode reports the first handoff gap without logging audio.

The coordinator still has no import or reference to robot tools, controllers,
robot state, SafetySupervisor, or e-stop reset authority. It receives a narrow
`LocalVoiceCommandExecutor` protocol through composition. The integration
adapter maps only `LocalVoiceCommand.STOP` to the existing high-level
`RobotIntent(STOP)` and `SafeRobotController`; all safety/controller contracts
remain authoritative. The router exposes no movement, e-stop reset, sensor,
heartbeat, or low-level controls. Every non-STOP transcript follows the normal
LLM/tool path.

The old higher-threshold VAD speech-start interruption remains only behind
explicit `vad_experimental` configuration. Normal playback drains queued
self-speech and resets wake/VAD state before `IDLE`. The wake-gated default is
self-barge-in mitigation, not acoustic echo cancellation. “Hey Jarvis, stop”
is the supported interruption; plain “Stop” over loud playback is not claimed
reliable.

Wake and rejected VAD audio exist only in bounded memory queues. The handoff
buffer is cleared after its snapshot and cannot become the next command. Only an
accepted utterance is written as a temporary WAV because whisper.cpp remains a
process-per-command file adapter; the file is removed by default. Warmup runs
local wake/VAD silence and optionally one discarded, unplayed TTS synthesis.
Ollama uses a voice-session keep-alive but is never automatically started and
no model is pulled.

Normal idle activation also retains the segmenter's bounded pre-roll and sends
its newest sub-threshold tail through VAD before newly read frames. This makes
`IDLE -> WAKE_DETECTED -> LISTENING` continuous in the same way as the speaking
handoff, without retaining room audio beyond the current wake attempt.

Display and speech text intentionally diverge only after conversation is
complete:

```text
LLMResponse.text (original Markdown)
  -> terminal + ConversationService history unchanged
  -> TTSService.prepare_text_for_speech
  -> SpeechChunker (ordered semantic sentences, bounded long-sentence fallback)
  -> SpeechPipeline producer (response generation ID, two-slot PCM queue)
  -> Kokoro create_stream or one-shot Piper sentence fallback
  -> AudioPlaybackService consumer (one continuous ordered speaker stream)
```

The formatter is deterministic text transformation only. It has no rendering,
network, URL, filesystem, code-execution, model, or tool authority.

## Phase 2C3.2 early sentence speech

`ConversationService.respond()` still completes the entire provider-neutral
tool loop before TTS sees any text. A structured tool call, tool result, safety
denial, and post-tool final response therefore settle before the first sentence
can be spoken. This phase deliberately does not use Ollama `stream=True` on the
initial tool-enabled request and does not infer tool intent with keywords or
regular expressions. The latency gain is entirely after final response commit.

The speech producer is a daemon thread and the playback consumer is the existing
cancellable background speaker thread. Each response receives a monotonic
generation ID. The producer may hold at most two queued provider-neutral PCM16
chunks ahead of the current playback chunk; blocking `Queue` operations provide
backpressure without unbounded memory. Kokoro 0.6.1's async `create_stream()`
remains contained inside `KokoroTTS` and yields Jarvis-owned `SpeechAudioChunk`
values. Piper retains full local synthesis for each Jarvis semantic sentence and
yields that sentence as one audio chunk.

Playback opens one `RawOutputStream` for the lazy sequence, verifies a stable
sample rate/channel format, and writes all chunks strictly in order without
overlap or disk files. `PROCESSING -> SPEAKING` occurs only after that stream
actually starts. Later sentence inference continues while earlier audio plays.
On wake-barge cancellation, the current stream is aborted, the bounded queue is
cleared, and the generation token is latched. An inference call already inside
a provider may finish, but its result is discarded and no subsequent sentence
is synthesized. Provider inference is serialized so a late cancelled call and
the next response never use the same ONNX engine concurrently.

`VoiceLatencyTracker` records monotonic event timestamps and derives
wake-to-speech start, utterance duration, endpoint delay, STT, LLM/tools, TTS,
playback-start, speech-end-to-audio-start, wake-to-playback-cancel, and
STT-to-local-stop values. Phase 2C3.2 additionally records
assistant-text-ready, first-chunk-ready, first-audio-started, TTS-first-chunk,
full generation time, and queued/played audio-chunk counts. Normal mode is
quiet; `--debug-latency` prints actual per-interaction metrics and labeled
wake-barge/local-stop/no-speech events.

`LLMRequest` carries provider-neutral temperature with a validated default of
`0.2`; the Ollama adapter maps it to native options. Thinking remains off by
default. Immutable prompt policy explicitly preserves ordinary general-
knowledge conversation and directs uncertainty acknowledgment. Robot tools are
additional action mechanisms, never a restriction on conversational subjects.

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
- `jarvis.audio.tts.chunks`: deterministic provider-neutral semantic speech splitting.
- `jarvis.audio.tts.kokoro`: lazy local `kokoro-onnx` CPU adapter.
- `jarvis.audio.tts.piper`: lazy local OHF Piper CPU adapter.
- `jarvis.audio.tts.pipeline`: bounded generation-aware synthesis/playback queue.
- `jarvis.audio.tts.playback`: lazy output discovery and continuous queued PCM16 playback.
- `jarvis.audio.tts.text`: deterministic Markdown-to-speech text normalization.
- `jarvis.audio.tts.service`: provider selection and pipelined speech-session coordination.
- `jarvis.audio.tts.benchmark`: fixed-corpus local synthesis and guarded sample cleanup.
- `jarvis.audio.realtime`: bounded continuous local PCM frames with no persistence.
- `jarvis.audio.wake`: provider-neutral wake contract and lazy OpenWakeWord adapter.
- `jarvis.audio.vad`: provider-neutral Silero scoring and deterministic endpoint policy.
- `jarvis.audio.voice.state`: explicit voice state transition graph.
- `jarvis.audio.voice.latency`: structured per-turn timing and aggregation.
- `jarvis.audio.voice.commands`: exact no-speech filtering and STOP-only grammar.
- `jarvis.audio.voice.coordinator`: authority-free wake-to-response orchestration.
- `jarvis.integrations.voice_stop`: narrow STOP-to-safe-controller composition adapter.
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

Future modules may add streaming/early STT, safe LLM text streaming, real AEC,
memory, a separate Jarvis face, vision, external integrations, physical robot
components, and ESP32 transport. None of those features is part of Phase
2C3.2. The original BMO face/image assets remain untouched as historical and
design reference; they must not be removed, overwritten, renamed, replaced, or
made the primary Jarvis face without explicit user approval. Microphone/STT/TTS
testing and the robot simulator are not hardware safety validation.
