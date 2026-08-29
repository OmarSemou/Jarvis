"""Developer CLI for local text/voice input and the safe robot simulator."""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from jarvis.audio.benchmark import (
    BENCHMARK_PHRASES,
    BenchmarkRecording,
    BenchmarkSTT,
    cleanup_recordings,
    format_benchmark_report,
    run_benchmark,
)
from jarvis.audio.devices import (
    MicrophoneDeviceService,
    MicrophoneError,
    format_device_list,
    format_microphone_status,
)
from jarvis.audio.recorder import PushToTalkRecorder, RecordingError
from jarvis.audio.realtime import SoundDeviceRealtimeInput
from jarvis.audio.service import VoiceInputError, VoiceInputService
from jarvis.audio.stt.base import TranscriptionResult
from jarvis.audio.stt.whisper_cpp import (
    WHISPER_CPP_VERSION,
    WhisperCppSTT,
    WhisperCppSettings,
)
from jarvis.audio.tts.benchmark import (
    cleanup_tts_benchmark,
    format_tts_benchmark_report,
    run_tts_benchmark,
)
from jarvis.audio.tts.base import TTSProvider
from jarvis.audio.tts.kokoro import KokoroSettings, KokoroTTS
from jarvis.audio.tts.piper import PiperSettings, PiperTTS
from jarvis.audio.tts.profiles import resolve_voice_profile
from jarvis.audio.tts.playback import (
    AudioPlaybackService,
    SpeakerError,
    format_speaker_list,
    format_speaker_status,
)
from jarvis.audio.tts.service import TTSService
from jarvis.audio.vad.segmenter import VADSegmenter, VADSegmenterSettings
from jarvis.audio.vad.silero import SileroVAD, SileroVADSettings
from jarvis.audio.voice.coordinator import (
    BargeInMode,
    VoiceModeCoordinator,
    VoiceModeSettings,
)
from jarvis.audio.voice.state import VoiceStateMachine
from jarvis.audio.wake.openwakeword import OpenWakeWord, OpenWakeWordSettings, WAKE_PHRASE
from jarvis.core.config import ConfigValidationError, JarvisConfig, load_for_paths
from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.core.paths import JarvisPaths
from jarvis.memory import MemoryPolicy, MemoryService, SQLiteMemoryStore
from jarvis.memory.models import MemoryEntry
from jarvis.memory.tools import MemoryToolExecutor
from jarvis.face.assets import default_bmo_asset_set
from jarvis.face.controller import FaceController
from jarvis.face.demo import FaceDemoSettings, run_face_demo
from jarvis.face.tkinter_view import TkinterFaceView
from jarvis.llm.base import (
    ChatMessage,
    LLMError,
    LLMInterruptedError,
    LLMRequest,
    MessageRole,
)
from jarvis.llm.ollama import OllamaLLM, OllamaSettings
from jarvis.personality.profile import ACTIVE_ROBOT_NAME
from jarvis.personality.prompt import build_system_prompt
from jarvis.robot.controller import SafeRobotController, create_simulated_controller
from jarvis.robot.safety import SafetyAuthority
from jarvis.integrations.voice_stop import SafeLocalVoiceCommandExecutor
from jarvis.tools.policy import RobotToolPolicy
from jarvis.tools.registry import RobotToolRegistry
from jarvis.tools.types import ToolExecutor
from jarvis.tools.composite import CompositeToolExecutor


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class RobotRuntime:
    controller: SafeRobotController
    tools: RobotToolPolicy


def create_memory_runtime(
    config: JarvisConfig,
    paths: JarvisPaths,
    *,
    logger: OutputFunction | None = None,
    debug: bool = False,
) -> MemoryService:
    """Create the explicitly initialized local memory service."""

    database = paths.memory_database_path(config.memory_database_path)
    policy = MemoryPolicy(
        max_key_chars=config.memory_max_key_chars,
        max_value_chars=config.memory_max_value_chars,
        max_summary_chars=config.memory_max_summary_chars,
        max_context_entries=config.memory_max_context_entries,
        max_context_chars=config.memory_max_context_chars,
        max_records=config.memory_max_records,
    )
    return MemoryService(
        SQLiteMemoryStore(database),
        policy,
        enabled=config.memory_enabled,
        logger=logger,
        debug=debug,
    )


def create_voice_runtime(
    config: JarvisConfig,
    paths: JarvisPaths | None = None,
) -> VoiceInputService:
    paths = paths or JarvisPaths.discover()
    devices = MicrophoneDeviceService(config.input_device)
    recorder = PushToTalkRecorder(
        devices,
        paths.recordings_dir,
        preferred_sample_rate=config.input_sample_rate,
    )
    stt = create_stt_provider(config, paths)
    return VoiceInputService(
        devices,
        recorder,
        stt,
        retain_recordings=config.retain_recordings,
    )


def create_tts_providers(config: JarvisConfig, paths: JarvisPaths) -> dict[str, TTSProvider]:
    del config
    piper_files = {
        voice: paths.piper_voice_files(voice)
        for voice in ("en_US-joe-medium", "en_US-john-medium")
    }
    bmo_files = paths.legacy_bmo_voice_files
    if all(path.is_file() for path in bmo_files):
        piper_files["bmo"] = bmo_files
    return {
        "kokoro": KokoroTTS(
            KokoroSettings(paths.kokoro_model.resolve(), paths.kokoro_voices.resolve())
        ),
        "piper": PiperTTS(PiperSettings(piper_files)),
    }


def create_tts_runtime(
    config: JarvisConfig,
    paths: JarvisPaths | None = None,
    *,
    profile: str | None = None,
) -> TTSService:
    paths = paths or JarvisPaths.discover()
    selected_profile = profile if profile is not None else getattr(config, "tts_profile", None)
    if selected_profile is not None:
        selected = resolve_voice_profile(selected_profile)
        provider = selected.provider
        voice = selected.provider_voice
    else:
        provider = config.tts_provider
        voice = config.tts_voice
    return TTSService(
        create_tts_providers(config, paths),
        AudioPlaybackService(config.output_device),
        enabled=config.tts_enabled,
        provider=provider,
        voice=voice,
        speed=config.tts_speed,
        language=config.tts_language,
        profile=selected_profile,
    )


def create_stt_provider(
    config: JarvisConfig,
    paths: JarvisPaths,
    *,
    model: str | None = None,
) -> WhisperCppSTT:
    selected_model = (model or config.stt_model).casefold()
    return WhisperCppSTT(
        WhisperCppSettings(
            executable_path=paths.resolve_from_root(config.whisper_executable_path),
            model_path=paths.whisper_model_for(selected_model).resolve(),
            temp_dir=paths.stt_temp_dir.resolve(),
            model_name=selected_model,
            language=config.stt_language,
            timeout_seconds=config.stt_timeout_seconds,
            use_gpu=config.stt_use_gpu,
        )
    )


def create_robot_runtime(
    *,
    output_fn: OutputFunction | None = None,
    state_sink: Callable[[object], None] | None = None,
) -> RobotRuntime:
    controller = create_simulated_controller(event_sink=output_fn, state_sink=state_sink)
    return RobotRuntime(controller, RobotToolPolicy(RobotToolRegistry(), controller))


def create_ollama_provider(
    config: JarvisConfig,
    *,
    keep_alive: str | None = None,
) -> OllamaLLM:
    return OllamaLLM(
        OllamaSettings(
            host=config.ollama_host,
            connect_timeout_seconds=config.ollama_connect_timeout_seconds,
            read_timeout_seconds=config.ollama_read_timeout_seconds,
            keep_alive=keep_alive or config.ollama_keep_alive,
        )
    )


def create_conversation(
    config: JarvisConfig,
    *,
    tool_executor: ToolExecutor | None = None,
    memory_service: MemoryService | None = None,
    keep_alive: str | None = None,
) -> ConversationService:
    if memory_service is not None and memory_service.enabled:
        memory_tools = MemoryToolExecutor(memory_service)
        if tool_executor is None:
            tool_executor = memory_tools
        else:
            names = {
                definition.name
                for definition in getattr(tool_executor, "definitions", ())
            }
            if not {"remember_memory", "forget_memory"}.issubset(names):
                tool_executor = CompositeToolExecutor(tool_executor, memory_tools)
    return ConversationService(
        create_ollama_provider(config, keep_alive=keep_alive),
        ConversationSettings(
            model=config.llm_model,
            max_turns=config.conversation_max_turns,
            thinking=config.llm_thinking,
            max_tool_rounds=config.conversation_max_tool_rounds,
            temperature=config.llm_temperature,
        ),
        system_prompt=build_system_prompt(
            configured_prompt=config.system_prompt,
            configured_extras=config.system_prompt_extras,
        ),
        tool_executor=tool_executor,
        memory_service=memory_service,
    )


def create_voice_mode_coordinator(
    config: JarvisConfig,
    paths: JarvisPaths,
    conversation: ConversationService,
    voice_input: VoiceInputService,
    tts: TTSService,
    *,
    robot_controller: SafeRobotController | None = None,
    output_fn: OutputFunction = print,
    debug_latency: bool = False,
    state: VoiceStateMachine | None = None,
    speech_activity_sink: Callable[[str, int], None] | None = None,
) -> VoiceModeCoordinator:
    source = SoundDeviceRealtimeInput(
        voice_input.devices,
        preferred_sample_rate=config.input_sample_rate,
    )
    wakeword = OpenWakeWord(
        OpenWakeWordSettings(
            paths.wakeword_classifier_model.resolve(),
            paths.wakeword_melspectrogram_model.resolve(),
            paths.wakeword_embedding_model.resolve(),
            config.wakeword_threshold,
        )
    )
    vad = SileroVAD(SileroVADSettings(paths.vad_model.resolve()))
    segmenter = VADSegmenter(
        vad,
        VADSegmenterSettings(
            threshold=config.vad_speech_threshold,
            trailing_silence_ms=config.vad_trailing_silence_ms,
            max_utterance_seconds=config.vad_max_utterance_seconds,
            min_speech_ms=config.vad_min_speech_ms,
            listen_timeout_seconds=config.vad_listen_timeout_seconds,
        ),
    )
    return VoiceModeCoordinator(
        source,
        wakeword,
        vad,
        segmenter,
        voice_input,
        conversation,
        tts,
        local_command_executor=(
            SafeLocalVoiceCommandExecutor(robot_controller)
            if robot_controller is not None
            else None
        ),
        settings=VoiceModeSettings(
            preload_tts=config.tts_preload,
            barge_in_enabled=config.barge_in_enabled,
            barge_in_mode=BargeInMode(config.barge_in_mode),
            barge_in_threshold=config.barge_in_threshold,
            barge_in_suppression_ms=config.barge_in_suppression_ms,
            barge_in_min_speech_ms=config.barge_in_min_speech_ms,
            barge_in_pre_roll_ms=config.barge_in_pre_roll_ms,
            barge_in_command_start_timeout_seconds=(
                config.barge_in_command_start_timeout_seconds
            ),
            debug_latency=debug_latency or config.voice_debug_latency,
        ),
        state=state,
        speech_activity_sink=speech_activity_sink,
        output_fn=output_fn,
    )


def _format_status(service: ConversationService) -> str:
    status = service.status()
    return (
        f"Provider: {status.provider}\n"
        f"Endpoint: {status.endpoint}\n"
        f"Model: {status.model}\n"
        f"Thinking: {'on' if status.thinking else 'off'}\n"
        f"Temperature: {status.temperature:g}\n"
        f"History: {status.history_turns}/{status.max_turns} turns\n"
        f"Robot tools: {'enabled' if status.tools_enabled else 'disabled'}\n"
        f"Memory retrieval: {'enabled' if status.memory_enabled else 'disabled'}\n"
        f"Tool round limit: {status.max_tool_rounds}"
    )


def _format_robot_status(controller: SafeRobotController) -> str:
    state = controller.state
    gesture = state.last_gesture.value if state.last_gesture is not None else "none"
    return (
        "Robot simulation\n"
        f"Motion: {state.motion.value}\n"
        f"Following: {'yes' if state.following else 'no'}\n"
        f"Head: {state.head.value}\n"
        f"Expression: {state.expression.value}\n"
        f"Last gesture: {gesture}\n"
        f"E-stop: {'latched' if state.emergency_stop_latched else 'clear'}"
    )


def _format_memory_entry(entry: MemoryEntry) -> str:
    return f"{entry.id}: [{entry.category.value}] {entry.key} = {entry.value}"


def _format_memory_status(memory: MemoryService) -> str:
    status = memory.status()
    if not status.get("enabled", False):
        return "Memory: disabled"
    if not status.get("available", False):
        return f"Memory: unavailable\nPath: {status.get('path', '')}\nReason: {status.get('reason', 'unavailable')}"
    return (
        "Memory: enabled\n"
        f"Path: {status['path']}\n"
        f"Schema: {status['schema_version']}\n"
        f"Active: {status['active']}\n"
        f"Total records: {status['total']}"
    )


def _format_voice_selection(tts_runtime: TTSService) -> str:
    # Keep compatibility with the light-weight fake runtimes used by the
    # developer CLI tests and by embedders that predate semantic profiles.
    if not hasattr(tts_runtime, "profile"):
        provider = getattr(tts_runtime, "provider", "unknown")
        voice = getattr(tts_runtime, "voice", "unknown")
        return f"{provider} / {voice}"
    profile = getattr(tts_runtime, "profile", None)
    provider = getattr(tts_runtime, "provider", "unknown")
    voice = getattr(tts_runtime, "voice", "unknown")
    if profile is None:
        return f"legacy selection / {provider} {voice}"
    return f"{profile.display_name} / {profile.provider.title()} {profile.provider_voice}"


def _format_stt_status(voice: VoiceInputService) -> str:
    provider = voice.stt
    if not isinstance(provider, WhisperCppSTT):
        return f"STT provider: {provider.name}"
    settings = provider.settings
    readiness = provider.readiness_error()
    return (
        f"STT provider: {provider.name}\n"
        f"Version: {WHISPER_CPP_VERSION}\n"
        f"Model: {settings.model_name}\n"
        f"Backend: {'GPU' if settings.use_gpu else 'CPU'}\n"
        f"Language: {settings.language}\n"
        f"Ready: {'yes' if readiness is None else 'no'}"
    )


def _print_transcription_metrics(
    result: TranscriptionResult,
    output_fn: OutputFunction,
) -> None:
    duration = result.audio_duration_seconds
    factor = result.real_time_factor
    if duration is None:
        output_fn(f"[STT] transcription={result.elapsed_seconds:.2f}s")
        return
    factor_text = f" RTF={factor:.2f}" if factor is not None else ""
    output_fn(
        f"[STT] audio={duration:.2f}s transcription={result.elapsed_seconds:.2f}s{factor_text}"
    )


def _parse_microphone_selector(value: str) -> int | str:
    stripped = value.strip()
    return int(stripped) if stripped.isdecimal() else stripped


def _capture_voice(
    voice: VoiceInputService,
    *,
    input_fn: InputFunction,
    output_fn: OutputFunction,
    wait_to_start: bool = False,
) -> TranscriptionResult | None:
    try:
        if wait_to_start:
            input_fn("Press Enter to begin recording...")
        session = voice.start()
        output_fn(
            "Recording... press Enter to stop. "
            f"[{session.device.name}, {session.capture_sample_rate} Hz capture]"
        )
        input_fn("")
    except (EOFError, KeyboardInterrupt):
        voice.cancel()
        output_fn("Recording cancelled.")
        return None
    except (VoiceInputError, RecordingError) as exc:
        voice.cancel()
        output_fn(f"Voice input error: {exc}")
        return None

    output_fn("Transcribing locally...")
    try:
        outcome = voice.finish()
    except RecordingError as exc:
        output_fn(f"Voice input error: {exc}")
        return None
    except Exception as exc:
        output_fn(f"Voice input error: transcription failed safely: {exc}")
        return None
    if outcome.cleanup_warning:
        output_fn(f"Privacy warning: {outcome.cleanup_warning}")
    if outcome.retained_recording is not None:
        output_fn(f"Recording retained: {outcome.retained_recording}")
    result = outcome.transcription
    _print_transcription_metrics(result, output_fn)
    if not result.success:
        message = result.error.message if result.error is not None else "Unknown transcription failure."
        output_fn(f"Voice input error: {message}")
        return None
    return result


def _respond(
    service: ConversationService,
    user_text: str,
    output_fn: OutputFunction,
    tts_runtime: TTSService | None = None,
) -> int | None:
    try:
        response = service.respond(user_text)
    except LLMInterruptedError as exc:
        output_fn(f"Request interrupted: {exc}")
        return 130
    except LLMError as exc:
        output_fn(f"Error: {exc}")
        return None
    output_fn(f"{ACTIVE_ROBOT_NAME} > {response.text}")
    if tts_runtime is not None and tts_runtime.enabled:
        start_speech = getattr(tts_runtime, "start_speech", None)
        if callable(start_speech) and callable(
            getattr(tts_runtime.playback, "start_sequence", None)
        ):
            handle = start_speech(response.text)
            speech = handle.wait()
        else:
            speech = tts_runtime.speak(response.text)
        if speech is None:
            output_fn("Voice output error: local speech did not complete. Text response remains available.")
            return None
        if speech.success:
            metrics = getattr(speech, "metrics", None)
            if metrics is not None:
                first = metrics.tts_first_chunk or 0.0
                total = metrics.tts_total_generation or 0.0
                audio_start = (
                    0.0
                    if metrics.first_audio_started is None
                    else max(
                        0.0,
                        metrics.first_audio_started - metrics.assistant_text_ready,
                    )
                )
                output_fn(
                    f"[TTS] {speech.provider}/{speech.voice} "
                    f"first={first:.2f}s audio_start={audio_start:.2f}s "
                    f"total={total:.2f}s "
                    f"speech={metrics.speech_duration_seconds:.2f}s "
                    f"sentences={metrics.semantic_chunks} "
                    f"audio_chunks={metrics.played_chunks}"
                )
            else:
                synthesis = speech.synthesis
                duration = synthesis.speech_duration_seconds
                rtf = synthesis.real_time_factor
                output_fn(
                    f"[TTS] {synthesis.provider}/{synthesis.voice} "
                    f"synthesis={synthesis.elapsed_seconds:.2f}s "
                    f"speech={duration:.2f}s RTF={rtf:.2f}"
                )
        else:
            output_fn(
                "Voice output error: "
                f"{speech.error_message or 'unknown local speech failure'} "
                "Text response remains available."
            )
    return None


def run_chat(
    service: ConversationService,
    *,
    robot_controller: SafeRobotController | None = None,
    voice_runtime: VoiceInputService | None = None,
    tts_runtime: TTSService | None = None,
    memory_service: MemoryService | None = None,
    debug_tools: bool = False,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    status = service.status()
    output_fn(f"{ACTIVE_ROBOT_NAME} Local")
    output_fn(f"Model: {status.model}")
    output_fn(f"Thinking: {'on' if status.thinking else 'off'}")
    if debug_tools:
        names = tuple(getattr(service, "tool_names", ()))
        output_fn("[TOOLS] available=" + ",".join(names or ("none",)))
    if voice_runtime is not None:
        stt_provider = getattr(voice_runtime, "stt", None)
        stt_settings = getattr(stt_provider, "settings", None)
        stt_model = getattr(stt_settings, "model_name", "configured")
        stt_name = getattr(stt_provider, "name", "local")
        output_fn(f"STT: {stt_name} / {stt_model}")
    if tts_runtime is None or not tts_runtime.enabled:
        output_fn("Voice: off")
    else:
        output_fn(f"Voice: {_format_voice_selection(tts_runtime)}")
    output_fn(
        "Commands: /status, /reset, /think on, /think off, "
        "/robot status, /robot estop, /robot estop-reset, "
        "/memory status|list|show <id>|search <text>|forget <id>|forget-all, "
        "/talk, /stt status, /mic list, /mic status, /mic use <device>, "
        "/voice status|on|off|<fenrir|bmo>, /voice provider <name>, /voice use <voice>, "
        "/speaker list|status|use <device>, /quit"
    )

    while True:
        try:
            user_text = input_fn("\nYou > ").strip()
        except EOFError:
            output_fn("\nGoodbye.")
            return 0
        except KeyboardInterrupt:
            output_fn("\nInterrupted. Goodbye.")
            return 130

        if not user_text:
            continue
        command = user_text.lower()
        if command in {"/quit", "/exit"}:
            output_fn("Goodbye.")
            return 0
        if command == "/reset":
            service.reset()
            output_fn("Conversation reset. Robot and safety state unchanged.")
            continue
        if command == "/status":
            output_fn(_format_status(service))
            continue
        if command in {"/think on", "/think off"}:
            service.set_thinking(command.endswith("on"))
            output_fn(f"Thinking {'on' if service.thinking else 'off'}.")
            continue
        if command == "/mic list":
            if voice_runtime is None:
                output_fn("Voice input unavailable.")
            else:
                try:
                    output_fn(format_device_list(voice_runtime.devices.list_inputs()))
                except MicrophoneError as exc:
                    output_fn(f"Microphone error: {exc}")
            continue
        if command == "/stt status":
            if voice_runtime is None:
                output_fn("Voice input unavailable.")
            else:
                output_fn(_format_stt_status(voice_runtime))
            continue
        if command == "/mic status":
            if voice_runtime is None:
                output_fn("Voice input unavailable.")
            else:
                output_fn(format_microphone_status(voice_runtime.devices.status()))
            continue
        if command == "/mic use" or command.startswith("/mic use "):
            if voice_runtime is None:
                output_fn("Voice input unavailable.")
                continue
            requested = user_text[len("/mic use") :].strip()
            if not requested:
                output_fn("Usage: /mic use <device index or unique name>")
                continue
            previous = voice_runtime.devices.configured_device
            voice_runtime.devices.configured_device = _parse_microphone_selector(requested)
            selected_status = voice_runtime.devices.status()
            if not selected_status.available:
                voice_runtime.devices.configured_device = previous
                output_fn(f"Microphone error: {selected_status.detail}")
            else:
                selected = selected_status.selected
                output_fn(
                    f"Microphone selected for this session: {selected.index}: {selected.name}"
                )
            continue
        if command == "/talk":
            if voice_runtime is None:
                output_fn("Voice input unavailable.")
                continue
            transcription = _capture_voice(
                voice_runtime,
                input_fn=input_fn,
                output_fn=output_fn,
            )
            if transcription is None:
                continue
            output_fn(f"You (voice) > {transcription.text}")
            interrupted = _respond(service, transcription.text, output_fn, tts_runtime)
            if interrupted is not None:
                return interrupted
            continue
        if command == "/voice status":
            if tts_runtime is None:
                output_fn("Voice output unavailable.")
            else:
                voice_status = tts_runtime.status()
                output_fn(
                    f"Voice output: {'on' if voice_status.enabled else 'off'}\n"
                    f"Profile: {_format_voice_selection(tts_runtime)}\n"
                    f"Provider: {voice_status.provider}\n"
                    f"Voice: {voice_status.voice}\n"
                    f"Speed: {voice_status.speed:g}\n"
                    f"Language: {voice_status.language}\n"
                    f"Ready: {'yes' if voice_status.ready else 'no'}\n"
                    f"Detail: {voice_status.detail}"
                )
            continue
        if command in {"/voice on", "/voice off"}:
            if tts_runtime is None:
                output_fn("Voice output unavailable.")
            else:
                tts_runtime.set_enabled(command.endswith("on"))
                output_fn(f"Voice output {'on' if tts_runtime.enabled else 'off'}.")
            continue
        if command == "/voice" or command.startswith("/voice "):
            requested_profile = user_text[len("/voice") :].strip()
            if requested_profile.casefold() in {"fenrir", "bmo"}:
                if tts_runtime is None:
                    output_fn("Voice output unavailable.")
                    continue
                switch_profile = getattr(tts_runtime, "set_profile", None)
                if not callable(switch_profile):
                    output_fn("Voice profiles unavailable in this runtime.")
                    continue
                try:
                    switch_profile(requested_profile)
                except ValueError as exc:
                    output_fn(f"Voice configuration error: {exc}")
                else:
                    output_fn(
                        f"Voice profile selected for this session: "
                        f"{_format_voice_selection(tts_runtime)}"
                    )
                continue
        if command == "/voice provider" or command.startswith("/voice provider "):
            if tts_runtime is None:
                output_fn("Voice output unavailable.")
                continue
            requested = user_text[len("/voice provider") :].strip()
            if not requested:
                output_fn("Usage: /voice provider <kokoro or piper>")
                continue
            try:
                tts_runtime.set_provider(requested)
            except ValueError as exc:
                output_fn(f"Voice configuration error: {exc}")
            else:
                output_fn(
                    f"Voice provider selected for this session: "
                    f"{tts_runtime.provider} / {tts_runtime.voice}"
                )
            continue
        if command == "/voice use" or command.startswith("/voice use "):
            if tts_runtime is None:
                output_fn("Voice output unavailable.")
                continue
            requested = user_text[len("/voice use") :].strip()
            if not requested:
                output_fn("Usage: /voice use <allowlisted voice>")
                continue
            try:
                tts_runtime.set_voice(requested)
            except ValueError as exc:
                output_fn(f"Voice configuration error: {exc}")
            else:
                output_fn(f"Voice selected for this session: {tts_runtime.voice}")
            continue
        if command == "/speaker list":
            if tts_runtime is None:
                output_fn("Voice output unavailable.")
            else:
                try:
                    output_fn(format_speaker_list(tts_runtime.playback.list_outputs()))
                except SpeakerError as exc:
                    output_fn(f"Speaker error: {exc}")
            continue
        if command == "/speaker status":
            if tts_runtime is None:
                output_fn("Voice output unavailable.")
            else:
                output_fn(format_speaker_status(tts_runtime.playback.status()))
            continue
        if command == "/speaker use" or command.startswith("/speaker use "):
            if tts_runtime is None:
                output_fn("Voice output unavailable.")
                continue
            requested = user_text[len("/speaker use") :].strip()
            if not requested:
                output_fn("Usage: /speaker use <device index or unique name>")
                continue
            previous = tts_runtime.playback.configured_device
            tts_runtime.playback.configured_device = _parse_microphone_selector(requested)
            selected_status = tts_runtime.playback.status()
            if not selected_status.available:
                tts_runtime.playback.configured_device = previous
                output_fn(f"Speaker error: {selected_status.detail}")
            else:
                selected = selected_status.selected
                output_fn(
                    f"Speaker selected for this session: {selected.index}: {selected.name}"
                )
            continue
        if command == "/robot status":
            if robot_controller is None:
                output_fn("Robot simulation unavailable.")
            else:
                output_fn(_format_robot_status(robot_controller))
            continue
        if command == "/robot estop":
            if robot_controller is None:
                output_fn("Robot simulation unavailable.")
            else:
                transition = robot_controller.latch_emergency_stop(
                    authority=SafetyAuthority.LOCAL_OPERATOR
                )
                output_fn(transition.message.capitalize() + ".")
            continue
        if command == "/robot estop-reset":
            if robot_controller is None:
                output_fn("Robot simulation unavailable.")
            else:
                transition = robot_controller.reset_emergency_stop(
                    authority=SafetyAuthority.LOCAL_OPERATOR
                )
                prefix = "Reset accepted" if transition.accepted else "Reset denied"
                output_fn(f"{prefix}: {transition.message}.")
            continue
        if command == "/memory status":
            if memory_service is None:
                output_fn("Memory unavailable.")
            else:
                output_fn(_format_memory_status(memory_service))
            continue
        if command == "/memory list":
            if memory_service is None:
                output_fn("Memory unavailable.")
            else:
                entries = memory_service.list()
                output_fn("No active memories." if not entries else "\n".join(_format_memory_entry(entry) for entry in entries))
            continue
        if command == "/memory" or command.startswith("/memory "):
            if memory_service is None:
                output_fn("Memory unavailable.")
                continue
            argument = user_text[len("/memory") :].strip()
            parts = argument.split(maxsplit=1)
            action = parts[0].casefold() if parts else ""
            value = parts[1].strip() if len(parts) > 1 else ""
            if action == "show" and value.isdecimal():
                entry = memory_service.show(int(value))
                output_fn("Memory not found." if entry is None else _format_memory_entry(entry))
            elif action == "search" and value:
                entries = memory_service.search(value)
                output_fn("No matching memories." if not entries else "\n".join(_format_memory_entry(entry) for entry in entries))
            elif action == "forget" and value.isdecimal():
                output_fn(memory_service.forget(int(value)).message)
            elif action == "forget-all":
                try:
                    confirmation = input_fn("Type FORGET ALL to confirm: ").strip()
                except (EOFError, KeyboardInterrupt):
                    output_fn("Forget-all cancelled.")
                    continue
                if confirmation != "FORGET ALL":
                    output_fn("Forget-all cancelled.")
                else:
                    output_fn(memory_service.forget_all().message)
            else:
                output_fn("Usage: /memory status|list|show <id>|search <text>|forget <id>|forget-all")
            continue
        if command.startswith("/"):
            output_fn("Unknown command. Use /status for the command list.")
            continue

        interrupted = _respond(service, user_text, output_fn, tts_runtime)
        if interrupted is not None:
            return interrupted


def _load_runtime_config() -> JarvisConfig:
    return load_for_paths(JarvisPaths.discover()).config


def chat_command(output_fn: OutputFunction = print) -> int:
    memory: MemoryService | None = None
    try:
        paths = JarvisPaths.discover()
        config = load_for_paths(paths).config
        runtime = create_robot_runtime(output_fn=output_fn)
        memory = create_memory_runtime(config, paths, logger=output_fn)
        tools: ToolExecutor | None = runtime.tools
        if memory.enabled and hasattr(runtime.tools, "definitions"):
            tools = CompositeToolExecutor(runtime.tools, MemoryToolExecutor(memory))
        voice = create_voice_runtime(config, paths)
        tts = create_tts_runtime(config, paths)
        try:
            service = create_conversation(config, tool_executor=tools, memory_service=memory)
        except TypeError as exc:
            if "memory_service" not in str(exc):
                raise
            service = create_conversation(config, tool_executor=tools)
    except (ConfigValidationError, ImportError, OSError, RuntimeError, ValueError) as exc:
        output_fn(f"Configuration error: {exc}")
        return 2
    try:
        return run_chat(
            service,
            robot_controller=runtime.controller,
            voice_runtime=voice,
            tts_runtime=tts,
            memory_service=memory,
            output_fn=output_fn,
            debug_tools=False,
        )
    finally:
        service.close()
        if memory is not None:
            memory.close()


def face_command(
    *,
    fullscreen: bool = False,
    output_fn: OutputFunction = print,
) -> int:
    """Show the current read-only BMO prototype face."""

    try:
        assets = default_bmo_asset_set()
        controller = FaceController(assets)
        view = TkinterFaceView(controller, assets, fullscreen=fullscreen)
        output_fn("Face: BMO prototype")
        view.run()
        return 0
    except Exception as exc:
        output_fn(f"Face unavailable: {exc}")
        return 2


def face_demo_command(
    *,
    fullscreen: bool = False,
    gallery: bool = False,
    output_fn: OutputFunction = print,
) -> int:
    try:
        return run_face_demo(
            settings=FaceDemoSettings(fullscreen=fullscreen, gallery=gallery),
            output_fn=output_fn,
        )
    except Exception as exc:
        output_fn(f"Face demo unavailable: {exc}")
        return 2


def _run_voice_with_face(
    coordinator: VoiceModeCoordinator,
    view: TkinterFaceView,
) -> int:
    """Keep Tk on the main thread while voice inference runs in a worker."""

    result = [1]

    def worker() -> None:
        try:
            result[0] = coordinator.run()
        finally:
            view.request_close()

    thread = threading.Thread(target=worker, name="jarvis-voice", daemon=True)
    thread.start()
    try:
        view.run()
    except KeyboardInterrupt:
        coordinator.request_shutdown()
    finally:
        coordinator.request_shutdown()
        thread.join(timeout=5.0)
    return result[0] if not thread.is_alive() else 1


def voice_command(
    *,
    debug_latency: bool = False,
    face: bool = False,
    fullscreen: bool = False,
    voice_profile: str | None = None,
    output_fn: OutputFunction = print,
) -> int:
    """Run local continuous wake/VAD/conversation/speech mode."""

    service: ConversationService | None = None
    memory: MemoryService | None = None
    try:
        paths = JarvisPaths.discover()
        config = load_for_paths(paths).config
        if not config.voice_mode_enabled:
            output_fn(
                "Continuous voice mode is disabled. Set voice_mode_enabled to true "
                "in private data/config.json; /talk remains available."
            )
            return 2
        if not config.wakeword_enabled:
            output_fn("Voice mode unavailable: wake-word detection is disabled.")
            return 2
        if not config.vad_enabled:
            output_fn("Voice mode unavailable: VAD is disabled.")
            return 2
        if not config.tts_enabled:
            output_fn("Voice mode unavailable: local TTS is disabled.")
            return 2

        face_controller: FaceController | None = None
        face_view: TkinterFaceView | None = None
        face_state: VoiceStateMachine | None = None
        if face:
            assets = default_bmo_asset_set()
            face_controller = FaceController(assets)
            face_state = VoiceStateMachine(on_transition=face_controller.observe_voice_state)
            face_view = TkinterFaceView(
                face_controller,
                assets,
                fullscreen=fullscreen,
                on_close=lambda: None,
            )
        robot = create_robot_runtime(
            output_fn=output_fn,
            state_sink=(face_controller.observe_robot_state if face_controller else None),
        )
        memory = create_memory_runtime(
            config,
            paths,
            logger=output_fn,
            debug=debug_latency or config.voice_debug_latency,
        )
        voice_tools: ToolExecutor = robot.tools
        if memory.enabled and hasattr(robot.tools, "definitions"):
            voice_tools = CompositeToolExecutor(robot.tools, MemoryToolExecutor(memory))
        voice_input = create_voice_runtime(config, paths)
        tts = (
            create_tts_runtime(config, paths)
            if voice_profile is None
            else create_tts_runtime(config, paths, profile=voice_profile)
        )
        try:
            service = create_conversation(
                config,
                tool_executor=voice_tools,
                memory_service=memory,
                keep_alive=config.voice_ollama_keep_alive,
            )
        except TypeError as exc:
            # Preserve compatibility with embedders that still expose the
            # pre-memory factory signature.
            if "memory_service" not in str(exc):
                raise
            service = create_conversation(
                config,
                tool_executor=robot.tools,
                keep_alive=config.voice_ollama_keep_alive,
            )
        coordinator = create_voice_mode_coordinator(
            config,
            paths,
            service,
            voice_input,
            tts,
            robot_controller=robot.controller,
            output_fn=output_fn,
            debug_latency=debug_latency,
            state=face_state,
            speech_activity_sink=(
                face_controller.observe_playback_event if face_controller else None
            ),
        )
        output_fn(f"{ACTIVE_ROBOT_NAME} Voice")
        output_fn(
            f"Wake word: {WAKE_PHRASE} "
            "(the model is trained for the full phrase)"
        )
        output_fn(f"STT: whisper.cpp / {config.stt_model}")
        output_fn(f"LLM: {config.llm_model}")
        output_fn(f"Voice: {_format_voice_selection(tts)}")
        if debug_latency or config.voice_debug_latency:
            names = tuple(getattr(service, "tool_names", ()))
            output_fn("[TOOLS] available=" + ",".join(names or ("none",)))
        barge_status = config.barge_in_mode if config.barge_in_enabled else "disabled"
        output_fn(f"Barge-in: {barge_status}")
        if face:
            output_fn("Face: BMO prototype")
        output_fn("Preparing local voice models...")
        if face_view is not None:
            face_view.on_close = coordinator.request_shutdown
            return _run_voice_with_face(coordinator, face_view)
        return coordinator.run()
    except (ConfigValidationError, ImportError, OSError, RuntimeError, ValueError) as exc:
        output_fn(f"Configuration error: {exc}")
        return 2
    except Exception as exc:
        output_fn(f"Voice mode unavailable: {exc}")
        return 2
    finally:
        if service is not None:
            service.close()
        if memory is not None:
            memory.close()


def llm_check_command(output_fn: OutputFunction = print) -> int:
    """Perform an explicit local model check; never download or pull anything."""

    try:
        config = _load_runtime_config()
        provider = create_ollama_provider(config)
    except (ConfigValidationError, ImportError, ValueError) as exc:
        output_fn(f"Configuration error: {exc}")
        return 2

    try:
        output_fn(f"Checking {provider.endpoint} ...")
        provider.ensure_model_available(config.llm_model)
        output_fn(f"Model available: {config.llm_model}")
        request = LLMRequest(
            model=config.llm_model,
            messages=(
                ChatMessage(MessageRole.SYSTEM, "Return only a short final answer. Do not use tools."),
                ChatMessage(
                    MessageRole.USER,
                    f"Reply with exactly: {ACTIVE_ROBOT_NAME} local check OK",
                ),
            ),
            thinking=False,
            temperature=config.llm_temperature,
        )
        started = perf_counter()
        response = provider.generate(request)
        elapsed = perf_counter() - started
        output_fn(f"Response: {response.text}")
        output_fn(f"End-to-end response time: {elapsed:.2f}s")
        if response.load_duration_ns is not None:
            output_fn(f"Model load time reported by Ollama: {response.load_duration_ns / 1_000_000_000:.2f}s")
        if response.eval_count is not None:
            output_fn(f"Generated tokens reported by Ollama: {response.eval_count}")
        if response.eval_count and response.eval_duration_ns:
            rate = response.eval_count / (response.eval_duration_ns / 1_000_000_000)
            output_fn(f"Generation rate reported by Ollama: {rate:.1f} tokens/s")
        output_fn("Local Ollama integration: PASS")
        return 0
    except LLMInterruptedError as exc:
        output_fn(f"Integration check interrupted: {exc}")
        return 130
    except LLMError as exc:
        output_fn(f"Integration check failed: {exc}")
        return 1
    finally:
        provider.close()


def stt_check_command(
    *,
    microphone: str | None = None,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    """Record and transcribe once without constructing or calling an LLM."""

    try:
        paths = JarvisPaths.discover()
        config = load_for_paths(paths).config
        voice = create_voice_runtime(config, paths)
        if microphone is not None:
            voice.devices.configured_device = _parse_microphone_selector(microphone)
    except (ConfigValidationError, ValueError) as exc:
        output_fn(f"Configuration error: {exc}")
        return 2
    output_fn(f"{ACTIVE_ROBOT_NAME} local STT check (no LLM call)")
    result = _capture_voice(
        voice,
        input_fn=input_fn,
        output_fn=output_fn,
        wait_to_start=True,
    )
    if result is None:
        return 1
    output_fn(f"Transcription: {result.text}")
    output_fn("Local whisper.cpp integration: PASS")
    return 0


def stt_benchmark_command(
    *,
    microphone: str | None = None,
    retain_recordings: bool = False,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
    paths: JarvisPaths | None = None,
    config: JarvisConfig | None = None,
    recorder: PushToTalkRecorder | None = None,
    providers: dict[str, BenchmarkSTT] | None = None,
) -> int:
    """Capture the fixed corpus once and compare both local models without an LLM."""

    recordings: list[BenchmarkRecording] = []
    try:
        paths = paths or JarvisPaths.discover()
        config = config or load_for_paths(paths).config
        if recorder is None:
            devices = MicrophoneDeviceService(config.input_device)
            if microphone is not None:
                devices.configured_device = _parse_microphone_selector(microphone)
            recorder = PushToTalkRecorder(
                devices,
                paths.recordings_dir,
                preferred_sample_rate=config.input_sample_rate,
            )
        elif microphone is not None:
            recorder.devices.configured_device = _parse_microphone_selector(microphone)
        providers = providers or {
            model: create_stt_provider(config, paths, model=model)
            for model in ("base", "small")
        }
    except (ConfigValidationError, ValueError) as exc:
        output_fn(f"Configuration error: {exc}")
        return 2

    for model in ("base", "small"):
        provider = providers.get(model)
        if provider is None:
            output_fn(f"STT benchmark unavailable: no provider configured for {model}.")
            return 1
        readiness_error = provider.readiness_error()
        if readiness_error is not None:
            output_fn(f"STT benchmark unavailable for {model}: {readiness_error.message}")
            return 1

    output_fn(f"{ACTIVE_ROBOT_NAME} local STT benchmark (no LLM or network calls)")
    output_fn(
        "Record each phrase once. The exact same temporary WAV will be used by base and small."
    )
    try:
        for index, phrase in enumerate(BENCHMARK_PHRASES, start=1):
            output_fn(f'Phrase {index}/{len(BENCHMARK_PHRASES)} [{phrase.language}]: "{phrase.expected}"')
            input_fn("Press Enter to begin recording...")
            session = recorder.start()
            output_fn(
                "Recording... press Enter to stop. "
                f"[{session.device.name}, {session.capture_sample_rate} Hz capture]"
            )
            input_fn("")
            recordings.append(BenchmarkRecording(phrase, recorder.stop()))

        output_fn("Running two sequential process-per-command passes...")
        report = run_benchmark(recordings, providers)
        output_fn(format_benchmark_report(report))
        return 0
    except (EOFError, KeyboardInterrupt):
        recorder.cancel()
        output_fn("Benchmark cancelled.")
        return 130
    except RecordingError as exc:
        recorder.cancel()
        output_fn(f"Benchmark recording error: {exc}")
        return 1
    except Exception as exc:
        recorder.cancel()
        output_fn(f"STT benchmark failed safely: {exc}")
        return 1
    finally:
        if retain_recordings:
            for recording in recordings:
                output_fn(f"Benchmark recording retained: {recording.audio.path}")
        else:
            for warning in cleanup_recordings(recordings):
                output_fn(f"Privacy warning: {warning}")


def tts_benchmark_command(
    *,
    output_fn: OutputFunction = print,
    paths: JarvisPaths | None = None,
    config: JarvisConfig | None = None,
    providers: dict[str, TTSProvider] | None = None,
) -> int:
    """Synthesize the fixed English corpus without constructing LLM or STT services."""

    try:
        paths = paths or JarvisPaths.discover()
        config = config or load_for_paths(paths).config
        providers = providers or create_tts_providers(config, paths)
        for provider in providers.values():
            for voice in provider.available_voices:
                failure = provider.readiness_error(voice)
                if failure is not None:
                    output_fn(
                        f"TTS benchmark unavailable for {provider.name}/{voice}: "
                        f"{failure.message}"
                    )
                    return 1
        output_fn(
            f"{ACTIVE_ROBOT_NAME} local TTS benchmark "
            "(no LLM, STT, playback, or network calls)"
        )
        report = run_tts_benchmark(providers, paths.tts_benchmark_dir)
        output_fn(format_tts_benchmark_report(report))
        return 0 if report.successful else 1
    except (ConfigValidationError, ValueError) as exc:
        output_fn(f"Configuration error: {exc}")
        return 2
    except Exception as exc:
        output_fn(f"TTS benchmark failed safely: {exc}")
        return 1


def tts_benchmark_clean_command(
    run_directory: str,
    *,
    output_fn: OutputFunction = print,
    paths: JarvisPaths | None = None,
) -> int:
    try:
        paths = paths or JarvisPaths.discover()
        target = Path(run_directory)
        if not target.is_absolute():
            target = paths.tts_benchmark_dir / target
        cleanup_tts_benchmark(target, paths.tts_benchmark_dir)
        output_fn(f"Removed TTS benchmark samples: {target.resolve()}")
        return 0
    except (OSError, ValueError) as exc:
        output_fn(f"TTS benchmark cleanup refused: {exc}")
        return 2


def memory_command(
    action: str = "status",
    value: str | None = None,
    *,
    confirm: bool = False,
    output_fn: OutputFunction = print,
) -> int:
    """Inspect or explicitly edit local memory without invoking the LLM."""

    memory: MemoryService | None = None
    try:
        paths = JarvisPaths.discover()
        config = load_for_paths(paths).config
        memory = create_memory_runtime(config, paths, logger=output_fn)
        normalized = action.casefold()
        if normalized == "status":
            output_fn(_format_memory_status(memory))
        elif normalized == "list":
            entries = memory.list()
            output_fn("No active memories." if not entries else "\n".join(_format_memory_entry(entry) for entry in entries))
        elif normalized == "show" and value and value.isdecimal():
            entry = memory.show(int(value))
            output_fn("Memory not found." if entry is None else _format_memory_entry(entry))
        elif normalized == "search" and value:
            entries = memory.search(value)
            output_fn("No matching memories." if not entries else "\n".join(_format_memory_entry(entry) for entry in entries))
        elif normalized == "forget" and value and value.isdecimal():
            output_fn(memory.forget(int(value)).message)
        elif normalized == "forget-all":
            if not confirm:
                output_fn("Refusing forget-all without --confirm.")
                return 2
            output_fn(memory.forget_all().message)
        else:
            output_fn("Usage: python -m jarvis memory status|list|show <id>|search <text>|forget <id>|forget-all --confirm")
            return 2
        return 0
    except (ConfigValidationError, OSError, RuntimeError, ValueError) as exc:
        output_fn(f"Memory configuration error: {exc}")
        return 2
    finally:
        if memory is not None:
            memory.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m jarvis", description="Jarvis local developer commands")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="start the local text conversation CLI")
    voice_parser = subparsers.add_parser(
        "voice", help="start continuous local wake/VAD conversational voice mode"
    )
    voice_parser.add_argument(
        "--debug-latency",
        action="store_true",
        help="print structured timing for each completed voice interaction",
    )
    voice_parser.add_argument(
        "--face",
        action="store_true",
        help="show the animated prototype face while voice mode runs",
    )
    voice_parser.add_argument(
        "--voice",
        dest="voice_profile",
        choices=("fenrir", "bmo"),
        help="select semantic voice profile for this session",
    )
    voice_parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="use fullscreen for the optional face view",
    )
    face_parser = subparsers.add_parser(
        "face", help="show the read-only BMO prototype face"
    )
    face_parser.add_argument("--fullscreen", action="store_true")
    demo_parser = subparsers.add_parser(
        "face-demo", help="cycle the animated prototype face"
    )
    demo_parser.add_argument("--fullscreen", action="store_true")
    demo_parser.add_argument(
        "--gallery", action="store_true", help="show the asset gallery label"
    )
    subparsers.add_parser("llm-check", help="explicitly test the configured local Ollama model")
    stt_parser = subparsers.add_parser(
        "stt-check", help="record and transcribe once without an LLM call"
    )
    stt_parser.add_argument(
        "--mic",
        metavar="DEVICE",
        help="session-only microphone index or unique name",
    )
    benchmark_parser = subparsers.add_parser(
        "stt-benchmark",
        help="record the fixed bilingual corpus once and compare base with small",
    )
    benchmark_parser.add_argument(
        "--mic",
        metavar="DEVICE",
        help="session-only microphone index or unique name",
    )
    subparsers.add_parser(
        "tts-benchmark",
        help="synthesize the fixed local corpus for all curated voices without playback",
    )
    cleanup_parser = subparsers.add_parser(
        "tts-benchmark-clean",
        help="remove one explicit retained TTS benchmark run",
    )
    cleanup_parser.add_argument("run_directory", help="run directory name or full path")
    benchmark_parser.add_argument(
        "--retain-recordings",
        action="store_true",
        help="keep the temporary benchmark WAV files for developer inspection",
    )
    memory_parser = subparsers.add_parser("memory", help="inspect or edit local persistent memory")
    memory_parser.add_argument(
        "action",
        choices=("status", "list", "show", "search", "forget", "forget-all"),
    )
    memory_parser.add_argument("value", nargs="?", help="memory id or search text")
    memory_parser.add_argument("--confirm", action="store_true", help="confirm destructive forget-all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "chat":
        return chat_command()
    if args.command == "voice":
        return voice_command(
            debug_latency=args.debug_latency,
            face=args.face,
            fullscreen=args.fullscreen,
            voice_profile=args.voice_profile,
        )
    if args.command == "face":
        return face_command(fullscreen=args.fullscreen)
    if args.command == "face-demo":
        return face_demo_command(fullscreen=args.fullscreen, gallery=args.gallery)
    if args.command == "llm-check":
        return llm_check_command()
    if args.command == "stt-check":
        return stt_check_command(microphone=args.mic)
    if args.command == "stt-benchmark":
        return stt_benchmark_command(
            microphone=args.mic,
            retain_recordings=args.retain_recordings,
        )
    if args.command == "tts-benchmark":
        return tts_benchmark_command()
    if args.command == "tts-benchmark-clean":
        return tts_benchmark_clean_command(args.run_directory)
    if args.command == "memory":
        return memory_command(args.action, args.value, confirm=args.confirm)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
