"""Provider-independent language-model messages, responses, and errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event
from typing import Protocol, runtime_checkable


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("message role must be a MessageRole")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("message content must be a non-empty string")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True, slots=True)
class LLMRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    thinking: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.thinking, bool):
            raise TypeError("thinking must be a boolean")
        if not self.messages:
            raise ValueError("at least one message is required")


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("response text must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("response model must be a non-empty string")


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
