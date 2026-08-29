from types import SimpleNamespace

from jarvis.audio.devices import AudioDevice, MicrophoneStatus
from jarvis.audio.recorder import RecordingError, RecordingSession
from jarvis.audio.service import VoiceInputOutcome
from jarvis.audio.stt.base import TranscriptionResult
from jarvis.audio.tts.base import SpeechSynthesisResult, SynthesizedAudio, SynthesisErrorCode, SynthesisFailure
from jarvis.audio.tts.playback import OutputDevice, PlaybackResult, SpeakerStatus
from jarvis.audio.tts.service import SpeechOutputResult, TTSStatus
from jarvis.audio.stt.whisper_cpp import WhisperCppSTT, WhisperCppSettings
from jarvis.cli import create_voice_runtime, run_chat
from jarvis.core.config import JarvisConfig
from jarvis.core.paths import JarvisPaths
from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.llm.base import LLMResponse
from jarvis.memory import MemoryService, SQLiteMemoryStore
from jarvis.memory.tools import MemoryToolExecutor
from jarvis.tools.composite import CompositeToolExecutor
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


class FakePlayback:
    def __init__(self):
        self.configured_device = None
        self.device = OutputDevice(6, "Mock Speakers", 2, 48_000, True)

    def list_outputs(self):
        return (self.device,)

    def status(self):
        return SpeakerStatus(True, self.device, self.configured_device, "default output")


class FakeTTS:
    def __init__(self, *, fail=False):
        self.enabled = False
        self.provider = "kokoro"
        self.voice = "am_michael"
        self.speed = 1.0
        self.language = "en"
        self.playback = FakePlayback()
        self.fail = fail
        self.spoken = []

    def status(self):
        return TTSStatus(self.enabled, self.provider, self.voice, 1.0, "en", True, "ready")

    def set_enabled(self, enabled):
        self.enabled = enabled

    def set_provider(self, provider):
        if provider not in {"kokoro", "piper"}:
            raise ValueError("unknown provider")
        self.provider = provider
        self.voice = "am_michael" if provider == "kokoro" else "en_US-joe-medium"

    def set_voice(self, voice):
        allowed = {
            "kokoro": {"am_michael"},
            "piper": {"en_US-joe-medium", "en_US-john-medium"},
        }
        if voice not in allowed[self.provider]:
            raise ValueError("unknown voice")
        self.voice = voice

    def speak(self, text):
        self.spoken.append(text)
        if self.fail:
            synthesis = SpeechSynthesisResult(
                False,
                self.provider,
                self.voice,
                0.01,
                error=SynthesisFailure(SynthesisErrorCode.SYNTHESIS_FAILED, "mock failure"),
            )
            return SpeechOutputResult(False, synthesis)
        audio = SynthesizedAudio(b"\x00\x00" * 100, 10_000)
        synthesis = SpeechSynthesisResult(
            True, self.provider, self.voice, 0.01, audio, 0.01
        )
        return SpeechOutputResult(True, synthesis, PlaybackResult(True, "Mock Speakers"))


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
    assert "BMO > There." in output
    assert any(line.startswith("[STT] audio=2.00s") for line in output)


def test_voice_runtime_factory_returns_stt_service(tmp_path):
    paths = JarvisPaths.from_repository_root(tmp_path.resolve())
    runtime = create_voice_runtime(JarvisConfig(), paths)

    assert runtime is not None
    assert runtime.stt.name == "whisper.cpp"
    assert runtime.stt.settings.model_name == "base"


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


def test_voice_path_persists_explicit_memory_and_new_session_retrieves_it(tmp_path):
    db = tmp_path / "data" / "bmo.db"
    memory = MemoryService(SQLiteMemoryStore(db))
    provider = Provider([LLMResponse("Noted.", "qwen3:8b")])
    conversation = make_service(provider, CompositeToolExecutor(MemoryToolExecutor(memory)))
    output = []

    result = run_chat(
        conversation,
        voice_runtime=FakeVoice("Remember that I prefer pistachio ice cream."),
        memory_service=memory,
        input_fn=inputs(["/talk", "", "/quit"]),
        output_fn=output.append,
    )

    assert result == 0
    assert len(memory.list()) == 1
    assert memory.list()[0].value == "pistachio"
    memory.close()

    restarted = MemoryService(SQLiteMemoryStore(db))
    query_provider = Provider([LLMResponse("Pistachio.", "qwen3:8b")])
    # Use the same provider-neutral retrieval seam as a fresh application
    # session; no history from the first conversation is reused.
    restarted_conversation = ConversationService(
        query_provider,
        ConversationSettings(model="qwen3:8b", max_turns=3),
        system_prompt="Test policy",
        memory_service=restarted,
    )
    restarted_conversation.respond("What's my favorite ice cream?")
    assert any("pistachio" in message.content.lower() for message in query_provider.requests[0].messages)
    restarted.close()


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
    assert "BMO > Typed reply" in output


def test_stt_status_is_local_and_reports_selected_model(tmp_path):
    provider = Provider([])
    voice = FakeVoice()
    executable = (tmp_path / "whisper-cli.exe").resolve()
    model = (tmp_path / "ggml-base.bin").resolve()
    executable.write_bytes(b"exe")
    model.write_bytes(b"model")
    voice.stt = WhisperCppSTT(
        WhisperCppSettings(
            executable,
            model,
            (tmp_path / "stt").resolve(),
            model_name="base",
            language="auto",
            use_gpu=False,
        )
    )
    output = []

    result = run_chat(
        make_service(provider),
        voice_runtime=voice,
        input_fn=inputs(["/stt status", "/quit"]),
        output_fn=output.append,
    )

    assert result == 0
    assert provider.requests == []
    status = next(line for line in output if line.startswith("STT provider:"))
    assert "Version: 1.9.1" in status
    assert "Model: base" in status
    assert "Backend: CPU" in status
    assert "Language: auto" in status
    assert "Ready: yes" in status


def test_voice_and_speaker_commands_are_session_only_and_do_not_call_llm():
    provider = Provider([])
    tts = FakeTTS()
    output = []

    result = run_chat(
        make_service(provider),
        tts_runtime=tts,
        input_fn=inputs(
            [
                "/voice status",
                "/voice on",
                "/voice provider piper",
                "/voice use en_US-john-medium",
                "/speaker list",
                "/speaker status",
                "/speaker use 6",
                "/voice off",
                "/quit",
            ]
        ),
        output_fn=output.append,
    )

    assert result == 0
    assert provider.requests == []
    assert tts.enabled is False
    assert tts.provider == "piper"
    assert tts.voice == "en_US-john-medium"
    assert tts.playback.configured_device == 6
    assert any(line.startswith("Voice output:") for line in output)
    assert any(line.startswith("Speaker outputs") for line in output)
    assert any(line.startswith("Speaker ready") for line in output)


def test_typed_and_talk_responses_both_use_tts_when_enabled():
    provider = Provider([LLMResponse("Typed.", "qwen3:8b"), LLMResponse("Spoken.", "qwen3:8b")])
    tts = FakeTTS()
    tts.enabled = True
    output = []

    run_chat(
        make_service(provider),
        voice_runtime=FakeVoice("Voice request"),
        tts_runtime=tts,
        input_fn=inputs(["Typed request", "/talk", "", "/quit"]),
        output_fn=output.append,
    )

    assert tts.spoken == ["Typed.", "Spoken."]
    assert output.count("[TTS] kokoro/am_michael synthesis=0.01s speech=0.01s RTF=1.00") == 2


def test_tts_failure_keeps_visible_assistant_text_and_chat_alive():
    provider = Provider([LLMResponse("Still visible.", "qwen3:8b")])
    tts = FakeTTS(fail=True)
    tts.enabled = True
    output = []

    result = run_chat(
        make_service(provider),
        tts_runtime=tts,
        input_fn=inputs(["Hello", "/quit"]),
        output_fn=output.append,
    )

    assert result == 0
    assert "BMO > Still visible." in output
    assert any(line.startswith("Voice output error: mock failure") for line in output)
