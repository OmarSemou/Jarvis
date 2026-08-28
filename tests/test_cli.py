from jarvis.cli import run_chat
from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.llm.base import LLMResponse, ModelUnavailableError
from jarvis.robot.controller import create_simulated_controller
from jarvis.tools.policy import RobotToolPolicy
from jarvis.tools.registry import RobotToolRegistry
from jarvis.tools.types import ToolCall


class CLIProvider:
    name = "Mock Ollama"
    endpoint = "http://127.0.0.1:11434"

    def __init__(self, error=None):
        self.requests = []
        self.error = error

    def generate(self, request, *, cancellation=None):
        self.requests.append(request)
        if self.error:
            raise self.error
        return LLMResponse("Hello from local Jarvis.", request.model)

    def close(self):
        pass


def service_for(provider):
    return ConversationService(
        provider,
        ConversationSettings(model="qwen3:8b", max_turns=3),
        system_prompt="Test system policy",
    )


def input_sequence(values):
    iterator = iter(values)
    return lambda prompt: next(iterator)


def test_cli_commands_and_mocked_conversation():
    provider = CLIProvider()
    service = service_for(provider)
    output = []

    result = run_chat(
        service,
        input_fn=input_sequence(["/status", "/think on", "Hello", "/reset", "/quit"]),
        output_fn=output.append,
    )

    assert result == 0
    assert any("Jarvis Local" in line for line in output)
    assert any("Model: qwen3:8b" in line for line in output)
    assert any("Thinking on." in line for line in output)
    assert any("Jarvis > Hello from local Jarvis." in line for line in output)
    assert any("Conversation reset." in line for line in output)
    assert provider.requests[0].thinking is True
    assert service.history == ()


def test_cli_shows_manual_model_command_without_traceback():
    provider = CLIProvider(ModelUnavailableError("qwen3:8b"))
    output = []

    result = run_chat(
        service_for(provider),
        input_fn=input_sequence(["Hello", "/quit"]),
        output_fn=output.append,
    )

    assert result == 0
    assert any("ollama pull qwen3:8b" in line for line in output)


def test_cli_handles_keyboard_interrupt_cleanly():
    def interrupted(prompt):
        raise KeyboardInterrupt

    output = []
    result = run_chat(service_for(CLIProvider()), input_fn=interrupted, output_fn=output.append)

    assert result == 130
    assert output[-1] == "\nInterrupted. Goodbye."


def test_robot_status_and_trusted_estop_commands_are_outside_llm():
    output = []
    controller = create_simulated_controller(event_sink=output.append)
    service = service_for(CLIProvider())

    result = run_chat(
        service,
        robot_controller=controller,
        input_fn=input_sequence(
            [
                "/robot status",
                "/robot estop",
                "/robot status",
                "/robot estop-reset",
                "/robot status",
                "/quit",
            ]
        ),
        output_fn=output.append,
    )

    assert result == 0
    statuses = [line for line in output if line.startswith("Robot simulation")]
    assert "E-stop: clear" in statuses[0]
    assert "E-stop: latched" in statuses[1]
    assert "E-stop: clear" in statuses[2]
    assert any(line == "[ROBOT] estop=latched" for line in output)
    assert any(line == "[ROBOT] estop=clear" for line in output)
    assert controller.state.emergency_stop_latched is False


def test_robot_commands_fail_cleanly_when_simulator_is_not_configured():
    output = []

    result = run_chat(
        service_for(CLIProvider()),
        input_fn=input_sequence(["/robot status", "/robot estop", "/robot estop-reset", "/quit"]),
        output_fn=output.append,
    )

    assert result == 0
    assert output.count("Robot simulation unavailable.") == 3


def test_cli_prints_concise_robot_event_before_natural_response():
    class ToolProvider(CLIProvider):
        def __init__(self):
            super().__init__()
            self.responses = iter(
                [
                    LLMResponse("", "qwen3:8b", tool_calls=(ToolCall("wave", {}),)),
                    LLMResponse("Hey.", "qwen3:8b"),
                ]
            )

        def generate(self, request, *, cancellation=None):
            self.requests.append(request)
            return next(self.responses)

    output = []
    controller = create_simulated_controller(event_sink=output.append)
    policy = RobotToolPolicy(RobotToolRegistry(), controller)
    service = ConversationService(
        ToolProvider(),
        ConversationSettings(model="qwen3:8b", max_turns=3),
        system_prompt="Test system policy",
        tool_executor=policy,
    )

    result = run_chat(
        service,
        robot_controller=controller,
        input_fn=input_sequence(["Wave at me", "/quit"]),
        output_fn=output.append,
    )

    assert result == 0
    assert "[ROBOT] gesture=wave" in output
    assert "Jarvis > Hey." in output
    assert output.index("[ROBOT] gesture=wave") < output.index("Jarvis > Hey.")
