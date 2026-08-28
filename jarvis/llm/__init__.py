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
from jarvis.tools.types import ToolCall, ToolDefinition, ToolResult, ToolResultStatus

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
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "ToolResultStatus",
]
