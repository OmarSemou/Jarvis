import pytest

from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.llm.base import (
    LLMRequest,
    LLMResponse,
    MessageRole,
    ProviderUnavailableError,
)
from jarvis.personality.prompt import build_system_prompt


class FakeProvider:
    name = "Fake local"
    endpoint = "memory://test"

    def __init__(self, responses=()):
        self.requests: list[LLMRequest] = []
        self.responses = iter(responses)
        self.closed = False

    def generate(self, request, *, cancellation=None):
        self.requests.append(request)
        try:
            text = next(self.responses)
        except StopIteration:
            text = f"reply {len(self.requests)}"
        if isinstance(text, Exception):
            raise text
        return LLMResponse(text, request.model)

    def close(self):
        self.closed = True


def make_service(provider, *, max_turns=12, configured_prompt=None):
    return ConversationService(
        provider,
        ConversationSettings(model="qwen3:8b", max_turns=max_turns),
        system_prompt=build_system_prompt(configured_prompt=configured_prompt),
    )


def test_system_and_personality_prompt_reach_provider():
    provider = FakeProvider(["Hello."])
    service = make_service(provider, configured_prompt="Address the user as Captain.")

    service.respond("Hello")

    system = provider.requests[0].messages[0]
    assert system.role is MessageRole.SYSTEM
    assert "<immutable_system_policy>" in system.content
    assert "<personality_profile>" in system.content
    assert "Name: Jarvis" in system.content
    assert "Address the user as Captain." in system.content


def test_turn_ordering_and_complete_history_recording():
    provider = FakeProvider(["First answer", "Second answer"])
    service = make_service(provider)

    service.respond("First question")
    service.respond("Second question")

    roles = [message.role for message in provider.requests[1].messages]
    contents = [message.content for message in provider.requests[1].messages]
    assert roles == [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER]
    assert contents[1:] == ["First question", "First answer", "Second question"]
    assert [message.role for message in service.history] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]


def test_reset_clears_turns_but_preserves_system_prompt():
    provider = FakeProvider(["Answer"])
    service = make_service(provider)
    original_system = service.system_prompt
    service.respond("Question")

    service.reset()

    assert service.history == ()
    assert service.messages[0].role is MessageRole.SYSTEM
    assert service.system_prompt == original_system


def test_history_limit_keeps_recent_complete_turns():
    provider = FakeProvider(["a1", "a2", "a3"])
    service = make_service(provider, max_turns=2)

    service.respond("q1")
    service.respond("q2")
    service.respond("q3")

    assert [message.content for message in service.history] == ["q2", "a2", "q3", "a3"]
    assert [message.content for message in provider.requests[2].messages[1:]] == [
        "q1",
        "a1",
        "q2",
        "a2",
        "q3",
    ]
    assert service.status().history_turns == 2


def test_failed_request_does_not_corrupt_history():
    provider = FakeProvider([ProviderUnavailableError("offline")])
    service = make_service(provider)

    with pytest.raises(ProviderUnavailableError, match="offline"):
        service.respond("Do not record this")

    assert service.history == ()


def test_thinking_can_be_toggled_per_runtime():
    provider = FakeProvider(["answer"])
    service = make_service(provider)

    service.set_thinking(True)
    service.respond("Think")

    assert provider.requests[0].thinking is True


def test_conservative_temperature_is_provider_neutral_and_reported_in_status():
    provider = FakeProvider(["answer"])
    service = ConversationService(
        provider,
        ConversationSettings(model="qwen3:8b", temperature=0.3),
        system_prompt=build_system_prompt(),
    )

    service.respond("Question")

    assert provider.requests[0].temperature == 0.3
    assert service.status().temperature == 0.3


@pytest.mark.parametrize("temperature", (-0.1, 2.1, float("nan"), True))
def test_invalid_conversation_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        ConversationSettings(temperature=temperature)


def test_close_delegates_to_provider():
    provider = FakeProvider()
    service = make_service(provider)

    service.close()

    assert provider.closed is True
