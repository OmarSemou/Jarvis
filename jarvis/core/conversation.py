"""Provider-independent, in-memory conversation and bounded tool orchestration."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from jarvis.llm.base import (
    CancellationToken,
    ChatMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MessageRole,
)
from jarvis.tools.types import ToolCall, ToolExecutor, ToolResult, ToolResultStatus
from jarvis.memory.intent import is_persistent_memory_query


class MemoryContextProvider(Protocol):
    """Minimal provider-neutral retrieval seam used by the conversation core."""

    def retrieve_context(self, query: str) -> str: ...


@dataclass(frozen=True, slots=True)
class ConversationSettings:
    model: str = "qwen3:8b"
    max_turns: int = 12
    thinking: bool = False
    max_tool_rounds: int = 3
    temperature: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("conversation model must be a non-empty string")
        if not isinstance(self.max_turns, int) or isinstance(self.max_turns, bool) or self.max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if not isinstance(self.thinking, bool):
            raise TypeError("thinking must be a boolean")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("conversation temperature must be a finite number from 0 to 2")
        object.__setattr__(self, "temperature", float(self.temperature))
        if (
            not isinstance(self.max_tool_rounds, int)
            or isinstance(self.max_tool_rounds, bool)
            or not 1 <= self.max_tool_rounds <= 10
        ):
            raise ValueError("max_tool_rounds must be an integer from 1 to 10")


@dataclass(frozen=True, slots=True)
class ConversationStatus:
    provider: str
    endpoint: str
    model: str
    thinking: bool
    temperature: float
    history_turns: int
    max_turns: int
    tools_enabled: bool
    max_tool_rounds: int
    memory_enabled: bool = False


class ConversationService:
    """Own complete turns and execute structured tools through one bounded policy."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: ConversationSettings,
        *,
        system_prompt: str,
        tool_executor: ToolExecutor | None = None,
        memory_service: MemoryContextProvider | None = None,
    ) -> None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        self._provider = provider
        self._settings = settings
        self._system_message = ChatMessage(MessageRole.SYSTEM, system_prompt.strip())
        self._history_turns: list[tuple[ChatMessage, ...]] = []
        self._thinking = settings.thinking
        self._tool_executor = tool_executor
        self._memory_service = memory_service
        self._lock = RLock()

    @property
    def system_prompt(self) -> str:
        return self._system_message.content

    def _flat_history(self) -> tuple[ChatMessage, ...]:
        return tuple(message for turn in self._history_turns for message in turn)

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        with self._lock:
            return self._flat_history()

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        with self._lock:
            return (self._system_message, *self._flat_history())

    @property
    def thinking(self) -> bool:
        with self._lock:
            return self._thinking

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Names exposed to the provider, without leaking provider objects."""

        with self._lock:
            if self._tool_executor is None:
                return ()
            return tuple(definition.name for definition in self._tool_executor.definitions)

    def set_thinking(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        with self._lock:
            self._thinking = enabled

    def reset(self) -> None:
        """Clear conversation/tool transcripts, never simulator or safety state."""

        with self._lock:
            self._history_turns.clear()

    def _commit_turn(self, messages: list[ChatMessage]) -> None:
        self._history_turns.append(tuple(messages))
        if len(self._history_turns) > self._settings.max_turns:
            del self._history_turns[: len(self._history_turns) - self._settings.max_turns]

    @staticmethod
    def _assistant_message(response: LLMResponse) -> ChatMessage:
        return ChatMessage(
            MessageRole.ASSISTANT,
            response.text,
            tool_calls=response.tool_calls,
        )

    def _request(
        self,
        turn: list[ChatMessage],
        *,
        expose_tools: bool,
        cancellation: CancellationToken | None,
    ) -> LLMResponse:
        definitions = self._tool_executor.definitions if expose_tools and self._tool_executor else ()
        memory_messages: tuple[ChatMessage, ...] = ()
        if self._memory_service is not None:
            try:
                context = self._memory_service.retrieve_context(turn[0].content)
            except Exception:
                context = ""
            if context.strip():
                memory_messages = (ChatMessage(MessageRole.USER, context.strip()),)
        request = LLMRequest(
            model=self._settings.model,
            messages=(self._system_message, *self._flat_history(), *memory_messages, *turn),
            thinking=self._thinking,
            tools=definitions,
            temperature=self._settings.temperature,
        )
        try:
            return self._provider.generate(request, cancellation=cancellation)
        except Exception:
            if any(message.role is MessageRole.TOOL for message in turn):
                turn.append(
                    ChatMessage(
                        MessageRole.ASSISTANT,
                        "Robot tool results were recorded, but the final language response was unavailable.",
                    )
                )
                self._commit_turn(turn)
            raise

    @staticmethod
    def _execution_failure_results(calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        return tuple(
            ToolResult(
                call,
                ToolResultStatus.ERROR,
                "The trusted tool execution pipeline failed safely.",
                "tool_execution_failure",
            )
            for call in calls
        )

    def _execute_calls(
        self, calls: tuple[ToolCall, ...], *, user_text: str | None = None
    ) -> tuple[ToolResult, ...]:
        if self._tool_executor is None:
            return tuple(
                ToolResult(
                    call,
                    ToolResultStatus.DENIED,
                    "No robot tool executor is configured.",
                    "tools_unavailable",
                )
                for call in calls
            )
        try:
            setter = getattr(self._tool_executor, "set_user_text", None)
            if callable(setter) and user_text is not None:
                setter(user_text)
            results = tuple(self._tool_executor.execute(calls))
        except Exception:
            return self._execution_failure_results(calls)
        if len(results) != len(calls) or any(
            result.call != call for call, result in zip(calls, results, strict=True)
        ):
            return self._execution_failure_results(calls)
        return results

    def _final_response(
        self,
        turn: list[ChatMessage],
        *,
        cancellation: CancellationToken | None,
    ) -> LLMResponse:
        response = self._request(turn, expose_tools=False, cancellation=cancellation)
        if response.tool_calls:
            turn.append(self._assistant_message(response))
            limit_results = tuple(
                ToolResult(
                    call,
                    ToolResultStatus.DENIED,
                    "No further robot actions were executed because the tool loop is closed.",
                    "tool_round_limit",
                )
                for call in response.tool_calls
            )
            turn.extend(
                ChatMessage(MessageRole.TOOL, result.message, tool_result=result)
                for result in limit_results
            )
            response = LLMResponse(
                "I stopped the robot action loop at its safety limit.",
                self._settings.model,
            )
        response = self._truthful_side_effect_response(turn, response)
        response = self._truthful_memory_response(turn[0].content, response)
        turn.append(self._assistant_message(response))
        self._commit_turn(turn)
        return response

    def _execute_explicit_memory(
        self, user_text: str
    ) -> tuple[ToolCall, ToolResult] | None:
        if self._tool_executor is None:
            return None
        handler = getattr(self._tool_executor, "execute_explicit", None)
        if not callable(handler):
            return None
        try:
            outcome = handler(user_text)
        except Exception:
            return None
        if outcome is None:
            return None
        call, result = outcome
        if not isinstance(call, ToolCall) or not isinstance(result, ToolResult) or result.call != call:
            return None
        return call, result

    def _truthful_memory_response(self, user_text: str, response: LLMResponse) -> LLMResponse:
        """Prevent a positive persistent-memory claim when context is empty."""

        if self._memory_service is None or not is_persistent_memory_query(user_text):
            return response
        try:
            context = self._memory_service.retrieve_context(user_text)
        except Exception:
            return response
        if "No persistent memories were retrieved" not in context and "Persistent memory is unavailable" not in context:
            return response
        claim = re.search(
            r"\b(?:i\s+remember|i\s+have\s+(?:that|your)|your\s+.+\s+is|"
            r"you\s+(?:like|love|prefer|enjoy)|in\s+(?:my|your)\s+(?:persistent\s+)?memory)\b",
            response.text,
            re.IGNORECASE,
        )
        safe_empty = re.search(
            r"\b(?:don't|do not|cannot|can't)\s+(?:have|remember)\b.*\b(?:persistent\s+)?memor"
            r"|\b(?:no|none|nothing)\s+(?:persistent\s+)?memor",
            response.text,
            re.IGNORECASE,
        )
        positive_fact = re.search(
            r"\b(?:your\s+.+\s+is|you\s+(?:like|love|prefer|enjoy))\b",
            response.text,
            re.IGNORECASE,
        )
        if claim and (not safe_empty or positive_fact is not None):
            return LLMResponse(
                "I don't have any persistent memories for that yet.",
                response.model,
                total_duration_ns=response.total_duration_ns,
                load_duration_ns=response.load_duration_ns,
                prompt_eval_count=response.prompt_eval_count,
                eval_count=response.eval_count,
                eval_duration_ns=response.eval_duration_ns,
            )
        return response

    @staticmethod
    def _truthful_side_effect_response(
        turn: list[ChatMessage], response: LLMResponse
    ) -> LLMResponse:
        """Reject a positive memory claim when its tool result was not successful."""

        failed_memory = any(
            message.tool_result is not None
            and message.tool_result.call.name in {"remember_memory", "forget_memory"}
            and not message.tool_result.success
            for message in turn
        )
        if not failed_memory:
            return response
        if not re.search(
            r"\b(?:i\s+(?:remember(?:ed)?|forgot|saved|stored|deleted|removed)|"
            r"(?:it|that)\s+(?:is|was)\s+(?:in|from)\s+(?:my\s+)?(?:persistent\s+)?memory)\b",
            response.text,
            re.IGNORECASE,
        ):
            return response
        return LLMResponse(
            "I couldn't change that persistent memory; nothing was changed.",
            response.model,
            total_duration_ns=response.total_duration_ns,
            load_duration_ns=response.load_duration_ns,
            prompt_eval_count=response.prompt_eval_count,
            eval_count=response.eval_count,
            eval_duration_ns=response.eval_duration_ns,
        )

    def respond(
        self,
        user_text: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> LLMResponse:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be a non-empty string")
        with self._lock:
            turn = [ChatMessage(MessageRole.USER, user_text.strip())]
            explicit_memory = self._execute_explicit_memory(turn[0].content)
            if explicit_memory is not None:
                call, result = explicit_memory
                turn.append(ChatMessage(MessageRole.ASSISTANT, "", tool_calls=(call,)))
                turn.append(ChatMessage(MessageRole.TOOL, result.message, tool_result=result))
                return self._final_response(turn, cancellation=cancellation)
            for _ in range(self._settings.max_tool_rounds):
                response = self._request(
                    turn,
                    expose_tools=self._tool_executor is not None,
                    cancellation=cancellation,
                )
                if not response.tool_calls:
                    response = self._truthful_side_effect_response(turn, response)
                    response = self._truthful_memory_response(turn[0].content, response)
                    turn.append(self._assistant_message(response))
                    self._commit_turn(turn)
                    return response

                turn.append(self._assistant_message(response))
                results = self._execute_calls(response.tool_calls, user_text=turn[0].content)
                turn.extend(
                    ChatMessage(MessageRole.TOOL, result.message, tool_result=result)
                    for result in results
                )
                if any(not result.success for result in results):
                    return self._final_response(turn, cancellation=cancellation)

            return self._final_response(turn, cancellation=cancellation)

    def status(self) -> ConversationStatus:
        with self._lock:
            return ConversationStatus(
                provider=self._provider.name,
                endpoint=self._provider.endpoint,
                model=self._settings.model,
                thinking=self._thinking,
                temperature=self._settings.temperature,
                history_turns=len(self._history_turns),
                max_turns=self._settings.max_turns,
                tools_enabled=self._tool_executor is not None,
                max_tool_rounds=self._settings.max_tool_rounds,
                memory_enabled=(
                    self._memory_service is not None
                    and bool(getattr(self._memory_service, "enabled", True))
                ),
            )

    def close(self) -> None:
        self._provider.close()

    def __enter__(self) -> "ConversationService":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
