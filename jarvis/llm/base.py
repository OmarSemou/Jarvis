"""Provider-independent language-model messages, responses, and errors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event
from typing import Protocol, runtime_checkable

from jarvis.tools.types import ToolCall, ToolDefinition, ToolResult


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_result: ToolResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("message role must be a MessageRole")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if any(not isinstance(call, ToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must contain only ToolCall values")
        if self.role in {MessageRole.SYSTEM, MessageRole.USER}:
            if not self.content.strip() or self.tool_calls or self.tool_result is not None:
                raise ValueError("system and user messages require text and cannot contain tool data")
        elif self.role is MessageRole.ASSISTANT:
            if not self.content.strip() and not self.tool_calls:
                raise ValueError("assistant messages require text or tool calls")
            if self.tool_result is not None:
                raise ValueError("assistant messages cannot contain a tool result")
        elif self.role is MessageRole.TOOL:
            if self.tool_result is None or self.tool_calls:
                raise ValueError("tool messages require exactly one ToolResult")
            if not self.content.strip():
                object.__setattr__(self, "content", self.tool_result.message)

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True, slots=True)
class LLMRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    thinking: bool = False
    tools: tuple[ToolDefinition, ...] = ()
    temperature: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.thinking, bool):
            raise TypeError("thinking must be a boolean")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be a finite number from 0 to 2")
        object.__setattr__(self, "temperature", float(self.temperature))
        if not self.messages:
            raise ValueError("at least one message is required")
        object.__setattr__(self, "tools", tuple(self.tools))
        if any(not isinstance(tool, ToolDefinition) for tool in self.tools):
            raise TypeError("tools must contain only ToolDefinition values")
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool definition names must be unique")


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    tool_calls: tuple[ToolCall, ...] = ()
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("response text must be a string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("response model must be a non-empty string")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if any(not isinstance(call, ToolCall) for call in self.tool_calls):
            raise TypeError("response tool_calls must contain only ToolCall values")
        if not self.text.strip() and not self.tool_calls:
            raise ValueError("response must contain text or tool calls")


@dataclass(slots=True)
class CancellationToken:
    """Cooperative cancellation signal checked around bounded provider calls."""

    _event: Event = field(default_factory=Event, repr=False)

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


class LLMError(RuntimeError):
    """Base class for expected provider-facing failures."""


class ProviderUnavailableError(LLMError):
    """The configured local provider cannot be reached."""


class ModelUnavailableError(LLMError):
    """The configured model is not installed in the local provider."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.manual_command = f"ollama pull {model}"
        super().__init__(
            f"The local Ollama model '{model}' is unavailable. "
            f"Install it manually with: {self.manual_command}"
        )


class LLMTimeoutError(LLMError):
    """A bounded provider call exceeded its configured timeout."""


class LLMInterruptedError(LLMError):
    """A request was cancelled or interrupted before completion."""


class ProviderResponseError(LLMError):
    """The provider returned an error that is safe to show to a user."""


class LLMProtocolError(LLMError):
    """The provider returned a malformed or empty response."""


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal provider boundary used by the conversation service."""

    @property
    def name(self) -> str: ...

    @property
    def endpoint(self) -> str: ...

    def generate(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> LLMResponse: ...

    def close(self) -> None: ...
