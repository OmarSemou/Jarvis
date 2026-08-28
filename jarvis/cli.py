"""Developer CLI for local text conversation and the safe robot simulator."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

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


def run_chat(
    service: ConversationService,
    *,
    robot_controller: SafeRobotController | None = None,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    status = service.status()
    output_fn("Jarvis Local")
    output_fn(f"Model: {status.model}")
    output_fn(f"Thinking: {'on' if status.thinking else 'off'}")
    output_fn(
        "Commands: /status, /reset, /think on, /think off, "
        "/robot status, /robot estop, /robot estop-reset, /quit"
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

        try:
            response = service.respond(user_text)
        except LLMInterruptedError as exc:
            output_fn(f"Request interrupted: {exc}")
            return 130
        except LLMError as exc:
            output_fn(f"Error: {exc}")
            continue
        output_fn(f"Jarvis > {response.text}")


def _load_runtime_config() -> JarvisConfig:
    return load_for_paths(JarvisPaths.discover()).config


def chat_command(output_fn: OutputFunction = print) -> int:
    try:
        runtime = create_robot_runtime(output_fn=output_fn)
        service = create_conversation(_load_runtime_config(), tool_executor=runtime.tools)
    except (ConfigValidationError, ValueError) as exc:
        output_fn(f"Configuration error: {exc}")
        return 2
    try:
        return run_chat(service, robot_controller=runtime.controller, output_fn=output_fn)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m jarvis", description="Jarvis local developer commands")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="start the local text conversation CLI")
    subparsers.add_parser("llm-check", help="explicitly test the configured local Ollama model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "chat":
        return chat_command()
    if args.command == "llm-check":
        return llm_check_command()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
