import pytest

from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.llm.base import (
    LLMInterruptedError,
    LLMRequest,
    LLMResponse,
    MessageRole,
    ProviderUnavailableError,
)
from jarvis.robot.controller import create_simulated_controller
from jarvis.robot.safety import SafetyAuthority
from jarvis.tools.policy import RobotToolPolicy
from jarvis.tools.registry import RobotToolRegistry
from jarvis.tools.types import (
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultStatus,
)


MODEL = "qwen3:8b"
WAVE = ToolCall("wave", {})
WAVE_DEFINITION = ToolDefinition("wave", "Wave in simulation.")


class SequenceProvider:
    name = "Fake local"
    endpoint = "memory://tools"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests: list[LLMRequest] = []

    def generate(self, request, *, cancellation=None):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        pass


class RecordingExecutor:
    definitions = (WAVE_DEFINITION,)

    def __init__(self, *, denied=False):
        self.batches = []
        self.denied = denied

    def execute(self, calls):
        self.batches.append(calls)
        if self.denied:
            return tuple(
                ToolResult(
                    call,
                    ToolResultStatus.DENIED,
                    "Obstacle vetoed the action.",
                    "obstacle_ahead",
                )
                for call in calls
            )
        return tuple(
            ToolResult(call, ToolResultStatus.SUCCESS, "Wave completed.")
            for call in calls
        )


def service(provider, executor=None, *, max_rounds=3):
    return ConversationService(
        provider,
        ConversationSettings(model=MODEL, max_turns=4, max_tool_rounds=max_rounds),
        system_prompt="Trusted test policy",
        tool_executor=executor,
    )


def tool_response(*calls):
    return LLMResponse("", MODEL, tool_calls=tuple(calls))


def test_plain_conversation_still_uses_one_provider_call():
    provider = SequenceProvider([LLMResponse("Hey.", MODEL)])
    conversation = service(provider, RecordingExecutor())

    response = conversation.respond("Hello Jarvis")

    assert response.text == "Hey."
    assert len(provider.requests) == 1
    assert provider.requests[0].tools == (WAVE_DEFINITION,)
    assert [message.role for message in conversation.history] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]


def test_tool_call_executes_then_receives_natural_final_response():
    provider = SequenceProvider([tool_response(WAVE), LLMResponse("Hey.", MODEL)])
    executor = RecordingExecutor()
    conversation = service(provider, executor)

    response = conversation.respond("Wave at me")

    assert response.text == "Hey."
    assert executor.batches == [(WAVE,)]
    second_messages = provider.requests[1].messages
    assert [message.role for message in second_messages[-3:]] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert second_messages[-1].tool_result is not None
    assert second_messages[-1].tool_result.status is ToolResultStatus.SUCCESS
    assert [message.role for message in conversation.history] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]


def test_denied_tool_result_reaches_model_and_disables_retries():
    provider = SequenceProvider(
        [tool_response(ToolCall("move_forward", {})), LLMResponse("Can't—there's an obstacle.", MODEL)]
    )
    executor = RecordingExecutor(denied=True)
    conversation = service(provider, executor)

    response = conversation.respond("Move forward")

    result_message = provider.requests[1].messages[-1]
    assert response.text == "Can't—there's an obstacle."
    assert result_message.role is MessageRole.TOOL
    assert result_message.tool_result is not None
    assert result_message.tool_result.denial_reason == "obstacle_ahead"
    assert provider.requests[1].tools == ()
    assert len(executor.batches) == 1


def test_multiple_successful_tool_rounds_are_bounded_then_tools_are_removed():
    provider = SequenceProvider(
        [tool_response(WAVE), tool_response(WAVE), LLMResponse("Done.", MODEL)]
    )
    executor = RecordingExecutor()
    conversation = service(provider, executor, max_rounds=2)

    response = conversation.respond("Keep waving")

    assert response.text == "Done."
    assert len(executor.batches) == 2
    assert provider.requests[0].tools == (WAVE_DEFINITION,)
    assert provider.requests[1].tools == (WAVE_DEFINITION,)
    assert provider.requests[2].tools == ()


def test_tool_calls_from_the_closed_final_round_never_execute():
    provider = SequenceProvider([tool_response(WAVE), tool_response(WAVE)])
    executor = RecordingExecutor()
    conversation = service(provider, executor, max_rounds=1)

    response = conversation.respond("Loop forever")

    assert response.text == "I stopped the robot action loop at its safety limit."
    assert len(executor.batches) == 1
    assert provider.requests[1].tools == ()
    assert conversation.history[-2].tool_result is not None
    assert conversation.history[-2].tool_result.denial_reason == "tool_round_limit"


def test_provider_failure_after_execution_preserves_truthful_tool_transcript():
    provider = SequenceProvider(
        [tool_response(WAVE), ProviderUnavailableError("offline after action")]
    )
    conversation = service(provider, RecordingExecutor())

    with pytest.raises(ProviderUnavailableError, match="offline after action"):
        conversation.respond("Wave")

    assert [message.role for message in conversation.history] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert "results were recorded" in conversation.history[-1].content


def test_interruption_after_execution_also_preserves_tool_transcript():
    provider = SequenceProvider([tool_response(WAVE), LLMInterruptedError("cancelled")])
    conversation = service(provider, RecordingExecutor())

    with pytest.raises(LLMInterruptedError, match="cancelled"):
        conversation.respond("Wave")

    assert conversation.history[-2].role is MessageRole.TOOL
    assert "results were recorded" in conversation.history[-1].content


def test_failure_before_any_tool_execution_does_not_change_history():
    provider = SequenceProvider([ProviderUnavailableError("offline")])
    conversation = service(provider, RecordingExecutor())

    with pytest.raises(ProviderUnavailableError):
        conversation.respond("Hello")

    assert conversation.history == ()


def test_reset_clears_transcript_but_never_clears_trusted_estop():
    controller = create_simulated_controller()
    controller.latch_emergency_stop(authority=SafetyAuthority.LOCAL_OPERATOR)
    policy = RobotToolPolicy(RobotToolRegistry(), controller)
    provider = SequenceProvider([LLMResponse("Noted.", MODEL)])
    conversation = service(provider, policy)
    conversation.respond("Hello")

    conversation.reset()

    assert conversation.history == ()
    assert controller.state.emergency_stop_latched is True
