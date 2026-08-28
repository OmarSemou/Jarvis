from types import SimpleNamespace

from jarvis.audio.devices import AudioDevice, MicrophoneStatus
from jarvis.audio.recorder import RecordingError, RecordingSession
from jarvis.audio.service import VoiceInputOutcome
from jarvis.audio.stt.base import TranscriptionResult
from jarvis.cli import run_chat
from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.llm.base import LLMResponse
from jarvis.robot.controller import create_simulated_controller
from jarvis.tools.policy import RobotToolPolicy
from jarvis.tools.registry import RobotToolRegistry
from jarvis.tools.types import ToolCall


class Provider:
    name = "mock"
    endpoint = "memory://voice"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, request, *, cancellation=None):
        self.requests.append(request)
        return next(self.responses)

    def close(self):
        pass


class FakeDevices:
    def __init__(self, device):
        self.device = device
        self.configured_device = None

    def list_inputs(self):
        return (self.device,)

    def status(self):
        return MicrophoneStatus(True, self.device, None, "default input")


class FakeVoice:
    def __init__(self, text="Look right.", start_error=None):
        self.device = AudioDevice(2, "Mock Mic", 1, 48_000, True)
        self.devices = FakeDevices(self.device)
        self.text = text
        self.start_error = start_error
        self.started = 0
        self.finished = 0
        self.cancelled = 0

    def start(self):
        self.started += 1
        if self.start_error:
            raise self.start_error
        return RecordingSession(self.device, 48_000)

    def finish(self):
        self.finished += 1
        return VoiceInputOutcome(
            TranscriptionResult(True, self.text, "mock-stt", 0.4, 2.0)
        )

    def cancel(self):
        self.cancelled += 1


def inputs(values):
    iterator = iter(values)
    return lambda _prompt: next(iterator)


def make_service(provider, tools=None):
    return ConversationService(
        provider,
        ConversationSettings(model="qwen3:8b", max_turns=3),
        system_prompt="Test policy",
        tool_executor=tools,
    )


def test_talk_passes_transcript_to_existing_conversation_service():
    provider = Provider([LLMResponse("There.", "qwen3:8b")])
    voice = FakeVoice("Look right.")
    output = []

    result = run_chat(
        make_service(provider),
        voice_runtime=voice,
        input_fn=inputs(["/talk", "", "/quit"]),
        output_fn=output.append,
    )

    assert result == 0
    assert voice.started == voice.finished == 1
    assert provider.requests[0].messages[-1].content == "Look right."
    assert "You (voice) > Look right." in output
    assert "Jarvis > There." in output
    assert any(line.startswith("[STT] audio=2.00s") for line in output)


def test_voice_uses_same_structured_robot_tool_path_as_typing():
    provider = Provider(
        [
            LLMResponse("", "qwen3:8b", tool_calls=(ToolCall("wave", {}),)),
            LLMResponse("Hey.", "qwen3:8b"),
        ]
    )
    output = []
    controller = create_simulated_controller(event_sink=output.append)
    tools = RobotToolPolicy(RobotToolRegistry(), controller)

    run_chat(
        make_service(provider, tools),
        robot_controller=controller,
        voice_runtime=FakeVoice("Wave at me."),
        input_fn=inputs(["/talk", "", "/quit"]),
        output_fn=output.append,
    )

    assert "[ROBOT] gesture=wave" in output
    assert controller.state.last_gesture.value == "wave"


def test_microphone_commands_do_not_call_llm():
    provider = Provider([])
    output = []
    result = run_chat(
        make_service(provider),
        voice_runtime=FakeVoice(),
        input_fn=inputs(["/mic list", "/mic status", "/mic use 2", "/quit"]),
        output_fn=output.append,
    )

    assert result == 0
    assert provider.requests == []
    assert any("Mock Mic" in line for line in output)
    assert any(line.startswith("Microphone ready") for line in output)
    assert any(line.startswith("Microphone selected for this session") for line in output)


def test_invalid_session_microphone_selection_restores_previous_device():
    provider = Provider([])
    voice = FakeVoice()
    voice.devices.configured_device = None

    def status():
        if voice.devices.configured_device == 99:
            return MicrophoneStatus(False, None, 99, "device disconnected")
        return MicrophoneStatus(True, voice.device, None, "default input")

    voice.devices.status = status
    output = []
    run_chat(
        make_service(provider),
        voice_runtime=voice,
        input_fn=inputs(["/mic use 99", "/quit"]),
        output_fn=output.append,
    )

    assert voice.devices.configured_device is None
    assert "Microphone error: device disconnected" in output


def test_microphone_start_error_is_clean_and_does_not_call_llm():
    provider = Provider([])
    voice = FakeVoice(start_error=RecordingError("device disconnected"))
    output = []

    result = run_chat(
        make_service(provider),
        voice_runtime=voice,
        input_fn=inputs(["/talk", "/quit"]),
        output_fn=output.append,
    )

    assert result == 0
    assert provider.requests == []
    assert "Voice input error: device disconnected" in output


def test_typed_chat_still_works_when_voice_runtime_exists():
    provider = Provider([LLMResponse("Typed reply", "qwen3:8b")])
    output = []

    run_chat(
        make_service(provider),
        voice_runtime=FakeVoice(),
        input_fn=inputs(["Typed message", "/quit"]),
        output_fn=output.append,
    )

    assert provider.requests[0].messages[-1].content == "Typed message"
    assert "Jarvis > Typed reply" in output
