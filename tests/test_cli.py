from jarvis.cli import run_chat
from jarvis.audio.tts.base import SpeechSynthesisResult, SynthesizedAudio
from jarvis.audio.tts.playback import PlaybackResult
from jarvis.audio.tts.service import TTSService
from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.llm.base import LLMResponse, MessageRole, ModelUnavailableError
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
        return LLMResponse("Hello from local BMO.", request.model)

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
    assert any("BMO Local" in line for line in output)
    assert any("Model: qwen3:8b" in line for line in output)
    assert any("Thinking on." in line for line in output)
    assert any("BMO > Hello from local BMO." in line for line in output)
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
    assert "BMO > Hey." in output
    assert output.index("[ROBOT] gesture=wave") < output.index("BMO > Hey.")


def test_markdown_display_and_history_remain_exact_while_tts_gets_plain_speech():
    display_text = "Use the **Power Stroke**."

    class MarkdownProvider(CLIProvider):
        def generate(self, request, *, cancellation=None):
            self.requests.append(request)
            return LLMResponse(display_text, request.model)

    class SpeechProvider:
        name = "kokoro"
        available_voices = ("am_fenrir",)

        def __init__(self):
            self.texts = []

        def readiness_error(self, _voice):
            return None

        def synthesize(self, text, *, voice, speed, language):
            self.texts.append(text)
            return SpeechSynthesisResult(
                True,
                self.name,
                voice,
                0.01,
                SynthesizedAudio(b"\x00\x00" * 100, 10_000),
            )

    class PiperProvider(SpeechProvider):
        name = "piper"
        available_voices = ("en_US-joe-medium",)

    class Playback:
        def play(self, _audio):
            return PlaybackResult(True, "mock")

        def stop(self):
            pass

    speech_provider = SpeechProvider()
    tts = TTSService(
        {"kokoro": speech_provider, "piper": PiperProvider()},
        Playback(),
        enabled=True,
    )
    service = service_for(MarkdownProvider())
    output = []

    assert run_chat(
        service,
        tts_runtime=tts,
        input_fn=input_sequence(["Explain the cycle", "/quit"]),
        output_fn=output.append,
    ) == 0

    assert f"BMO > {display_text}" in output
    assert service.history[-1].role is MessageRole.ASSISTANT
    assert service.history[-1].content == display_text
    assert speech_provider.texts == ["Use the Power Stroke."]
