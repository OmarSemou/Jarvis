from jarvis.cli import run_chat
from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.llm.base import LLMResponse, ModelUnavailableError


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
