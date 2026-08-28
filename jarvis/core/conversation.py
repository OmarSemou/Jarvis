"""Provider-independent, in-memory text conversation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from jarvis.llm.base import (
    CancellationToken,
    ChatMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MessageRole,
)


@dataclass(frozen=True, slots=True)
class ConversationSettings:
    model: str = "qwen3:8b"
    max_turns: int = 12
    thinking: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("conversation model must be a non-empty string")
        if not isinstance(self.max_turns, int) or isinstance(self.max_turns, bool) or self.max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if not isinstance(self.thinking, bool):
            raise TypeError("thinking must be a boolean")


@dataclass(frozen=True, slots=True)
class ConversationStatus:
    provider: str
    endpoint: str
    model: str
    thinking: bool
    history_turns: int
    max_turns: int


class ConversationService:
    """Own complete in-memory turns and submit immutable request snapshots."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: ConversationSettings,
        *,
        system_prompt: str,
    ) -> None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        self._provider = provider
        self._settings = settings
        self._system_message = ChatMessage(MessageRole.SYSTEM, system_prompt.strip())
        self._history: list[ChatMessage] = []
        self._thinking = settings.thinking
        self._lock = RLock()

    @property
    def system_prompt(self) -> str:
        return self._system_message.content

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        with self._lock:
            return (self._system_message, *self._history)

    @property
    def thinking(self) -> bool:
        with self._lock:
            return self._thinking

    def set_thinking(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        with self._lock:
            self._thinking = enabled

    def reset(self) -> None:
        with self._lock:
            self._history.clear()

    def respond(
        self,
        user_text: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> LLMResponse:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be a non-empty string")
        user_message = ChatMessage(MessageRole.USER, user_text.strip())
        with self._lock:
            request = LLMRequest(
                model=self._settings.model,
                messages=(self._system_message, *self._history, user_message),
                thinking=self._thinking,
            )
            response = self._provider.generate(request, cancellation=cancellation)
            assistant_message = ChatMessage(MessageRole.ASSISTANT, response.text.strip())
            self._history.extend((user_message, assistant_message))
            maximum_messages = self._settings.max_turns * 2
            if len(self._history) > maximum_messages:
                del self._history[: len(self._history) - maximum_messages]
            return response

    def status(self) -> ConversationStatus:
        with self._lock:
            return ConversationStatus(
                provider=self._provider.name,
                endpoint=self._provider.endpoint,
                model=self._settings.model,
                thinking=self._thinking,
                history_turns=len(self._history) // 2,
                max_turns=self._settings.max_turns,
            )

    def close(self) -> None:
        self._provider.close()

    def __enter__(self) -> "ConversationService":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
