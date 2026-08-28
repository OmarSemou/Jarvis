"""Developer CLI for local text/voice input and the safe robot simulator."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
from jarvis.audio.service import VoiceInputError, VoiceInputService
from jarvis.audio.stt.base import TranscriptionResult
from jarvis.audio.stt.whisper_cpp import (
    WHISPER_CPP_VERSION,
    WhisperCppSTT,
    WhisperCppSettings,
)
from jarvis.core.config import ConfigValidationError, JarvisConfig, load_for_paths
from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.core.paths import JarvisPaths
from jarvis.llm.base import (
    ChatMessage,
    LLMError,
    LLMInterruptedError,
    LLMRequest,
    MessageRole,
)
from jarvis.llm.ollama import OllamaLLM, OllamaSettings
from jarvis.personality.prompt import build_system_prompt
from jarvis.robot.controller import SafeRobotController, create_simulated_controller
from jarvis.robot.safety import SafetyAuthority
from jarvis.tools.policy import RobotToolPolicy
from jarvis.tools.registry import RobotToolRegistry
from jarvis.tools.types import ToolExecutor


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class RobotRuntime:
    controller: SafeRobotController
    tools: RobotToolPolicy


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


def create_robot_runtime(*, output_fn: OutputFunction | None = None) -> RobotRuntime:
    controller = create_simulated_controller(event_sink=output_fn)
    return RobotRuntime(controller, RobotToolPolicy(RobotToolRegistry(), controller))


def create_ollama_provider(config: JarvisConfig) -> OllamaLLM:
    return OllamaLLM(
        OllamaSettings(
            host=config.ollama_host,
            connect_timeout_seconds=config.ollama_connect_timeout_seconds,
            read_timeout_seconds=config.ollama_read_timeout_seconds,
            keep_alive=config.ollama_keep_alive,
        )
    )


def create_conversation(
    config: JarvisConfig,
    *,
    tool_executor: ToolExecutor | None = None,
) -> ConversationService:
    return ConversationService(
        create_ollama_provider(config),
        ConversationSettings(
            model=config.llm_model,
            max_turns=config.conversation_max_turns,
            thinking=config.llm_thinking,
            max_tool_rounds=config.conversation_max_tool_rounds,
        ),
        system_prompt=build_system_prompt(
            configured_prompt=config.system_prompt,
            configured_extras=config.system_prompt_extras,
        ),
        tool_executor=tool_executor,
    )


def _format_status(service: ConversationService) -> str:
    status = service.status()
    return (
        f"Provider: {status.provider}\n"
        f"Endpoint: {status.endpoint}\n"
        f"Model: {status.model}\n"
        f"Thinking: {'on' if status.thinking else 'off'}\n"
        f"History: {status.history_turns}/{status.max_turns} turns\n"
        f"Robot tools: {'enabled' if status.tools_enabled else 'disabled'}\n"
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
) -> int | None:
    try:
        response = service.respond(user_text)
    except LLMInterruptedError as exc:
        output_fn(f"Request interrupted: {exc}")
        return 130
    except LLMError as exc:
        output_fn(f"Error: {exc}")
        return None
    output_fn(f"Jarvis > {response.text}")
    return None


def run_chat(
    service: ConversationService,
    *,
    robot_controller: SafeRobotController | None = None,
    voice_runtime: VoiceInputService | None = None,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    status = service.status()
    output_fn("Jarvis Local")
    output_fn(f"Model: {status.model}")
    output_fn(f"Thinking: {'on' if status.thinking else 'off'}")
    output_fn(
        "Commands: /status, /reset, /think on, /think off, "
        "/robot status, /robot estop, /robot estop-reset, "
        "/talk, /stt status, /mic list, /mic status, /mic use <device>, /quit"
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
            interrupted = _respond(service, transcription.text, output_fn)
            if interrupted is not None:
                return interrupted
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
        if command.startswith("/"):
            output_fn("Unknown command. Use /status for the command list.")
            continue

        interrupted = _respond(service, user_text, output_fn)
        if interrupted is not None:
            return interrupted


def _load_runtime_config() -> JarvisConfig:
    return load_for_paths(JarvisPaths.discover()).config


def chat_command(output_fn: OutputFunction = print) -> int:
    try:
        paths = JarvisPaths.discover()
        config = load_for_paths(paths).config
        runtime = create_robot_runtime(output_fn=output_fn)
        voice = create_voice_runtime(config, paths)
        service = create_conversation(config, tool_executor=runtime.tools)
    except (ConfigValidationError, ValueError) as exc:
        output_fn(f"Configuration error: {exc}")
        return 2
    try:
        return run_chat(
            service,
            robot_controller=runtime.controller,
            voice_runtime=voice,
            output_fn=output_fn,
        )
    finally:
        service.close()


def llm_check_command(output_fn: OutputFunction = print) -> int:
    """Perform an explicit local model check; never download or pull anything."""

    try:
        config = _load_runtime_config()
        provider = create_ollama_provider(config)
    except (ConfigValidationError, ValueError) as exc:
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
                ChatMessage(MessageRole.USER, "Reply with exactly: Jarvis local check OK"),
            ),
            thinking=False,
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
    output_fn("Jarvis local STT check (no LLM call)")
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

    output_fn("Jarvis local STT benchmark (no LLM or network calls)")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m jarvis", description="Jarvis local developer commands")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="start the local text conversation CLI")
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
    benchmark_parser.add_argument(
        "--retain-recordings",
        action="store_true",
        help="keep the temporary benchmark WAV files for developer inspection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "chat":
        return chat_command()
    if args.command == "llm-check":
        return llm_check_command()
    if args.command == "stt-check":
        return stt_check_command(microphone=args.mic)
    if args.command == "stt-benchmark":
        return stt_benchmark_command(
            microphone=args.mic,
            retain_recordings=args.retain_recordings,
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
