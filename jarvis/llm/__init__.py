"""Provider-neutral local language-model contracts."""

from .base import (
    CancellationToken,
    ChatMessage,
    LLMError,
    LLMInterruptedError,
    LLMProtocolError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    MessageRole,
    ModelUnavailableError,
    ProviderResponseError,
    ProviderUnavailableError,
)

__all__ = [
    "CancellationToken",
    "ChatMessage",
    "LLMError",
    "LLMInterruptedError",
    "LLMProtocolError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMTimeoutError",
    "MessageRole",
    "ModelUnavailableError",
    "ProviderResponseError",
    "ProviderUnavailableError",
]
