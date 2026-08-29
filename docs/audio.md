# Phase 2C3.2 responsive local voice and early speech

BMO supports both the preserved `/talk` developer path and an explicitly
enabled continuous local voice mode. Continuous mode listens for a local
**Hey Jarvis** wake phrase while idle and while monitoring spoken playback for
an explicitly gated interruption. It uses deterministic VAD to bound one
utterance, invokes local `whisper.cpp`, uses either deterministic local STOP or
the existing conversation/tool path, and speaks through the selected local
voice profile (Fenrir/Kokoro by default).
There is no cloud speech service.

The current classifier still recognizes **Hey Jarvis** because that is the
available pinned model; this wake phrase is a compatibility limitation, not the
robot's active identity. BMO identity is established by the immutable
personality/profile layer.

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
`-Models base,small` and `-Models all` install both. BMO startup never invokes
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
CLI prints the complete response before synthesis begins. Typed chat and
`/talk` retain the Phase 2C2 synchronous synthesize/play behavior. Continuous
voice mode uses the same provider-neutral audio contract with a background
playback handle so the coordinator can cancel speech during barge-in. Phase
2C3.2 splits final text into deterministic sentence chunks and uses a bounded
lookahead queue. Kokoro can yield provider chunks; Piper, including the legacy
BMO model, uses the provider-neutral complete-sentence fallback.

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
/voice fenrir | /voice bmo
/voice provider kokoro | /voice provider piper
/voice use <allowlisted voice>
/speaker list | /speaker status
/speaker use <index or unique name>
```

The semantic voice profiles are:

- **Fenrir** (the default): Kokoro `am_fenrir`, using the current low-latency
  local path.
- **BMO**: the original upstream Be More Agent custom Piper model, exposed as
  Piper voice `bmo` only when both `voices/bmo-custom.onnx` and
  `voices/bmo-custom.onnx.json` are present. The repository does not download
  or synthesize a replacement model.

Select one for a continuous session with `python -m jarvis voice --voice fenrir`
or `--voice bmo`, or switch a running developer chat with `/voice fenrir` or
`/voice bmo`. `/voice status` reports the semantic profile and its concrete
provider mapping. The face subsystem is independent, so the BMO prototype face
can be used with either voice. Provider switching remains available for
diagnostics and selects a provider-local curated voice; `/voice use` can refine
that selection. These settings are session-only unless supplied in validated
configuration. An unavailable package, model, voice, output, or
synthesis/playback failure produces a concise message while keeping assistant
text visible.

Both profiles receive `prepare_text_for_speech()` output, never raw Markdown.
Profile changes stop the active generation, clear playback, and invalidate
warmup state before selecting the next provider. The existing generation-aware
pipeline therefore discards late Piper/BMO audio just as it discards late
Fenrir audio. BMO synthesis is not true provider streaming; it is still
sentence-level and cancellable at the Jarvis pipeline boundary.

### Restored BMO asset and provenance

The historical Linux `setup.sh` downloaded the original model from the
upstream Be More Agent release as `voices/bmo-custom.onnx` plus its JSON config.
The legacy `agent.py` launched a Piper executable with `--output-raw` and played
the returned mono 22050-Hz PCM. This was generated Piper TTS, not a library of
pre-recorded greeting samples. The modern adapter keeps that model/config
identity but loads it through the already-installed `piper-tts==1.7.0`
provider-neutral adapter; no legacy executable or subprocess is reintroduced.

The current private checkout contains the expected ignored artifacts. Their
local fingerprints are:

| file | size | SHA-256 |
| --- | ---: | --- |
| `voices/bmo-custom.onnx` | 63,511,038 bytes | `0b5a2f9e035f7798977320167f7b1bc5a5eeab4b15470d975b80fc56ae3bd8e0` |
| `voices/bmo-custom.onnx.json` | 7,105 bytes | `32e87407fd1a33b1282d6ddc80cc2af58eeec86d5c004062100732a8e996ca05` |

The JSON reports 22,050 Hz, `en-us`, eSpeak phonemes, one speaker, an empty
speaker map, dataset `ko_voice_dojo`, and Piper version `1.0.0`. These are
observed local values, not a claim that the model or character is MIT-licensed.

The model and any source voice dataset are not tracked in this repository. Their
complete license and redistribution provenance is unresolved, and the
repository MIT license does not grant rights to them. Obtain the original files
from a trusted source for private prototype use only and review their terms
before redistribution. `python -m jarvis.core.preflight` reports the two
historical BMO files as optional unless `tts_profile` is set to `bmo`, in which
case they become selected readiness checks. See `NOTICE.md` for the separate
dependency and asset notices. Historical source: https://github.com/brenpoly/be-more-agent

Phase 2C2 accepts English only. The official Piper catalog has Danish voices,
so the language boundary can support Danish later; the official Kokoro voice
set currently has no Danish voice. No Danish TTS model is installed here.

## TTS benchmark and retention

```powershell
.\.venv\Scripts\python.exe -m jarvis tts-benchmark
.\.venv\Scripts\python.exe -m jarvis tts-benchmark-clean "<run-directory>"
```

The benchmark constructs only the TTS adapters. It makes no LLM, STT, network,
microphone, or playback calls. It generates eight fixed phrases for the six
curated voices, plus the optional BMO voice when its original files are
installed, and writes labeled WAVs beneath one timestamped ignored directory.
Each row reports synthesis wall time, speech duration, real-time factor, sample
rate, first usable audio when the provider API exposes it, and output file.
Summaries report median/mean synthesis time, median RTF, fastest/slowest, and
short-utterance median.

For Kokoro's non-streaming API, “first usable audio” is the completed array and
therefore approximately equals total synthesis time. Piper yields chunks, so
the adapter records the first chunk. Neither provider is streamed to speakers
in this phase. Timings cannot decide voice quality; a human must listen for
tone, clarity, and pronunciation. Samples are retained deliberately until the
explicit guarded cleanup command removes one direct benchmark-run directory.

## Continuous voice setup and flow

Run the setup explicitly; application startup never installs packages or
downloads models:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_voice_windows.ps1
.\.venv\Scripts\python.exe -m jarvis voice
```

`scripts/setup_voice_windows.ps1` installs `openwakeword==0.6.0` in the project
virtual environment. OpenWakeWord's required SciPy/scikit-learn/ONNX Runtime
dependencies are ordinary local Python wheels; Phase 2C3.1 does not install
PyTorch or TorchAudio. The script downloads and SHA-256 verifies the official
OpenWakeWord v0.5.1 `hey_jarvis_v0.1.onnx`, `melspectrogram.onnx`,
`embedding_model.onnx`, and `silero_vad.onnx` assets into ignored runtime
storage.

The exact flow is:

```text
IDLE (local wake inference; no persistence or LLM)
  -> WAKE_DETECTED
  -> LISTENING (Silero score + deterministic endpoint policy)
  -> PROCESSING
  -> temporary 16 kHz WAV -> whisper.cpp -> delete by default
  -> exact no-speech marker filter
  -> anchored LocalVoiceCommandRouter
       -> STOP only -> SafeRobotController -> SafetySupervisor -> simulator
       -> otherwise -> ConversationService -> structured tools/safety/simulator
  -> selected Fenrir/Kokoro or BMO/Piper synthesis
  -> SPEAKING (cancellable local playback)
  -> IDLE
```

Phase 2C3.1 keeps continuation behavior A: every completed response returns to
the wake-word requirement. A follow-up listening window and indefinite open
conversation are deliberately not implemented. `/talk`, `stt-check`, typed
chat, and all existing CLI device commands remain available.

## Wake-word model and licensing

Continuous voice mode uses the official OpenWakeWord v0.5.1
`hey_jarvis_v0.1.onnx` classifier downloaded to ignored runtime storage. Its
SHA-256 is
`94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb`.
The repository's tracked legacy `wakeword.onnx` has SHA-256
`2b359120c5facbc7cfe58e87812cb7c303b697f3360cb99d3cd6f9a5e1dd64b9`.
It came from the upstream repository history, whose Linux setup labeled and
downloaded it as `hey_jarvis_v0.1.onnx`. Its graph input shape is compatible
with the OpenWakeWord feature pipeline, but its bytes no longer correspond to
a file at that historical moving URL. That is provenance evidence, not a
complete redistribution chain. It is retained untouched for legacy
compatibility but is not the continuous-voice classifier. The runtime and CLI
identify the phrase honestly as **Hey Jarvis**; bare “Jarvis” may have a higher
false-reject rate.

OpenWakeWord code is Apache-2.0. Its project documents bundled pretrained wake
models under CC BY-NC-SA 4.0. The official classifier is downloaded rather
than redistributed by this repository and remains suitable only for this
private prototype unless those terms are acceptable. Do not assume Jarvis's
MIT license grants commercial model rights. The legacy classifier's provenance
is still unresolved.

## VAD and endpoint policy

Phase 2C3.1 uses the ONNX Silero VAD wrapper already supplied by OpenWakeWord.
This keeps one lightweight ONNX inference stack and avoids a separate VAD
framework or PyTorch. The VAD provider returns only a speech probability. A
Jarvis-owned deterministic segmenter—not the model and never the LLM—requires
sustained speech, retains a short in-memory pre-roll, ends after trailing
silence, rejects no-speech/noise-only windows, and enforces a maximum duration.

Initial prototype defaults are threshold `0.50`, minimum sustained speech
`240 ms`, trailing silence `640 ms`, no-speech timeout `8 s`, and maximum
utterance `18 s`. These are explicit starting points within the requested
500–800 ms and 15–20 s ranges, not universal measured optima. They must be
measured with the actual microphone and room before further tuning.

## Barge-in and echo limitation

The default `barge_in_mode` is `wakeword`. During `SPEAKING`, generic Silero
speech probability cannot cancel playback. The same local Hey Jarvis detector
runs against a strictly bounded 320 ms in-memory rolling buffer. A confirmed
wake detection signals the chunk-cancellable playback handle and immediately
hands the same continuously open microphone stream to command segmentation:

```text
SPEAKING -> wake detected -> playback cancel signal
  -> INTERRUPTED -> LISTENING (same stream, no drain/restart)
  -> Silero-controlled command capture -> Whisper
```

The oldest buffer frames are passive pre-roll. Only the newest tail, kept
shorter than the configured VAD speech minimum, is re-scored with subsequent
live frames. This preserves a command phoneme that began in the wake-detection
frame while requiring at least one new live frame before VAD can confirm
speech; old Fenrir leakage cannot confirm a command by itself. The command
start window is separately bounded to 1.5 seconds after wake detection, so a
natural pause of about half a second remains supported without inheriting the
normal eight-second idle listening window. Both values are configurable as
`barge_in_pre_roll_ms` (200--500) and
`barge_in_command_start_timeout_seconds` (0.5--3.0).

Normal playback completion drains stale microphone frames and resets both wake
and VAD state before returning to `IDLE`. The handoff path never drains or
restarts the microphone. The former generic VAD-only interruption can be
selected as `vad_experimental`, but is not the default because Fenrir speaker
leakage caused repeatable self-barge-ins.

The same lossless handoff principle applies to normal idle activation. While
`IDLE`, wake detection keeps only the segmenter's bounded 240 ms pre-roll. On
detection, its newest sub-threshold tail and subsequent frames enter VAD in
sequence, preserving a first command word that begins immediately after “Hey
Jarvis.” The microphone is neither drained nor restarted between
`WAKE_DETECTED` and `LISTENING`.

Detector reset ownership is centralized at idle-wait entry, playback start,
playback completion, interruption handoff, and no-speech discard.
Speaking-state debug records include the actual wake score and
`source=speaking`, along with cancellation and frame-continuity evidence. This
distinguishes a strong human activation followed by no command from low-score
Fenrir leakage, but is not acoustic echo cancellation.

This is not acoustic echo cancellation. Plain speech-over-speaker interruption
such as saying only “Stop” is not considered reliable. The supported Phase
2C3.1 interruption phrase is **“Hey Jarvis, stop.”** Speaker leakage could
still trigger if playback itself contains the wake phrase, and far-field
performance remains environment-dependent.

## Blank audio and deterministic local stop

After VAD and STT, empty text and exact known Whisper no-speech markers such as
`[BLANK_AUDIO]` and `[ Silence ]` are discarded before conversation or TTS. A
wake-only transcript (`Jarvis`, `Hey Jarvis`, `BMO`, or `Hey BMO`) from handoff
pre-roll is also
discarded rather than sent to Qwen.
Recordings shorter than the configured VAD-confirmed speech minimum are also
discarded. Debug mode reports `[VOICE] no_speech_discarded`; normal mode stays
quiet and returns to `IDLE`.

An anchored, punctuation-normalizing allowlist recognizes only explicit STOP
forms such as `stop`, `please stop`, `hey bmo stop`, and legacy `hey jarvis stop`.
Negations,
questions, stop-sign discussion, and every other robot action do not match.
Matched STOP bypasses Qwen and executes this existing semantic path:

```text
Whisper transcript
  -> LocalVoiceCommandRouter (STOP only)
  -> SafeLocalVoiceCommandExecutor integration
  -> SafeRobotController.execute_intent(RobotIntent(STOP))
  -> SafetySupervisor (STOP always allowed)
  -> SimulatedRobot
  -> fixed local “Stopped.” acknowledgement through existing TTS
```

The router cannot clear e-stop, publish safety state, or request any movement
other than STOP. This remains a convenience/safety path, not the physical
emergency-stop implementation. A future physical robot requires an ESP32
watchdog, local motor timeout, and physical power-disable/e-stop circuit.

## Warmup, latency, and privacy

Voice-mode startup lazily loads and warms wake/VAD inference. When
`tts_preload=true`, Kokoro synthesizes one tiny phrase once; the returned audio
is discarded and never played or written. Whisper remains process-per-command.
Ollama requests in this mode use `voice_ollama_keep_alive` (`30m` by default),
but BMO never starts Ollama or pulls a model automatically.

Use `python -m jarvis voice --debug-latency` or set
`voice_debug_latency=true` to print real monotonic timings for wake-to-speech,
utterance duration, end detection, STT, LLM/tools, TTS, playback start, and
speech-end-to-audio-start, wake-to-playback-cancel, and STT-to-local-stop. It
also distinguishes assistant-text-ready, first speech chunk, first actual
audio, and eventual full TTS generation, with queued and played chunk counts.
While waiting for the wake phrase, debug mode prints
only a one-second rolling score peak and the configured threshold; it never
logs room audio. Normal voice mode does not print these diagnostics.
The prototype target is under `2.0 s` and the acceptable threshold is
`2.5 s`; hardware measurements are reported separately and are never
fabricated by the code.

Wake-barge debug output additionally reports the bounded pre-roll duration,
the timestamp gap and optional frame-sequence gap at the first newly consumed
frame, and the command speech-start offset. Audio content is never logged.

## Display Markdown and speech text

Assistant display text and `ConversationService` history retain the model's
original Markdown. Immediately before either Kokoro or Piper synthesis,
`TTSService` applies a provider-independent, deterministic
`prepare_text_for_speech` transformation. It removes emphasis markers,
heading/blockquote/list markers, code fences, and Markdown link destinations;
keeps visible link labels, image alt text, inline-code contents, useful numbered
list boundaries, and short fenced-code contents; and normalizes whitespace and
punctuation spacing. It never renders Markdown, fetches a link, opens a URL,
executes code, or touches the filesystem.

All normal assistant speech and the fixed local “Stopped.” acknowledgement pass
through this boundary. `SpeechChunker` then finds natural sentence, question,
exclamation, paragraph, and list boundaries while protecting common decimals,
abbreviations, and initials. A 220-character fallback prefers clause punctuation
and then whitespace, and never intentionally drops or duplicates speech text.

## Phase 2C3.2 sentence synthesis and queued playback

After the complete final assistant text is committed, `SpeechPipeline` assigns
a monotonic generation ID and starts a bounded producer/consumer path:

```text
final assistant Markdown
  -> prepare_text_for_speech
  -> ordered semantic SpeechChunk values
  -> Kokoro create_stream (or one-shot Piper sentence synthesis)
  -> two-slot provider-neutral PCM16 queue
  -> one continuous, strictly ordered RawOutputStream
```

The first usable PCM chunk starts playback immediately; the producer prepares
later chunks while earlier audio plays. A two-item queue is the default and
hard-limited to one through three, so fast synthesis blocks under backpressure
instead of building an answer-sized PCM buffer. Kokoro's pinned 0.6.1 async
stream API is fully contained in its adapter. Piper remains functional by
returning one PCM chunk for each semantic sentence. Normal response audio is
ephemeral and never written to disk.

Wake-barge cancellation latches the generation token, aborts current playback,
empties queued PCM, and stops future enqueue/sentence work. If an ONNX inference
already running cannot stop immediately, playback cancellation does not wait for
it; the late result is discarded. A subsequent response uses a newer ID, so old
audio cannot cross turn boundaries. Kokoro and Piper inference calls are also
serialized per provider to protect their shared loaded engines.

No Ollama token streaming is enabled in this phase. The initial request remains
tool-capable and completes its native structured tool/safety loop before speech.
This prevents a speculative sentence from contradicting a tool result or
`SafetySupervisor` denial while still removing the full-response TTS wait.

On the Phase 2C3.2 Windows workstation acceptance run, the real local
Qwen-to-Kokoro-to-Jabra text path measured first-chunk versus eventual TTS
generation as 1.35 s versus 23.04 s for the combustion-engine answer, 2.48 s
versus 72.16 s for the Roman Empire answer, and 1.13 s versus 67.29 s for the
Xbox Series X answer. Their generated speech durations were 42.67 s, 99.50 s,
and 99.69 s respectively. These are machine-specific provider/pipeline values,
not fabricated full voice-turn measurements: endpoint, microphone STT, and
speech-end-to-audio timing still require a person to perform the documented
`voice --debug-latency` acceptance run.

Normal conversation uses validated Ollama temperature `0.2` by default, with
thinking still off. This is a conservative factual-assistant setting; it does
not eliminate hallucinations. The immutable prompt requires normal general-
knowledge answers, treats robot tools as optional action mechanisms, and tells
BMO to acknowledge uncertainty instead of inventing facts.

Continuous microphone frames, wake buffers, VAD scores, and rejected room
audio remain in memory and are not logged. Only an accepted utterance becomes
a uniquely named temporary WAV required by process-per-command whisper.cpp.
It is deleted after success or failure unless the existing explicit
`retain_recordings=true` developer setting is used. There is no network or
cloud fallback.
