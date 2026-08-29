"""Bounded loopback-only adapter for the official Ollama Python client."""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit

import httpx
from ollama import Client, ResponseError

from .base import (
    CancellationToken,
    ChatMessage,
    LLMInterruptedError,
    LLMProtocolError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    ModelUnavailableError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from jarvis.tools.types import ToolCall, ToolDefinition, ToolParameter


def validate_loopback_endpoint(value: str) -> str:
    """Validate and normalize an HTTP Ollama endpoint on a loopback address."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Ollama endpoint must be a non-empty URL")
    try:
        parsed = urlsplit(value.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid Ollama endpoint: {exc}") from exc
    if parsed.scheme.lower() != "http":
        raise ValueError("Ollama endpoint must use http on a loopback host")
    if hostname is None:
        raise ValueError("Ollama endpoint must include a loopback host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Ollama endpoint must not include credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Ollama endpoint must not include a path, query, or fragment")
    port = 11434 if port is None else port
    if not 1 <= port <= 65_535:
        raise ValueError("Ollama endpoint port must be between 1 and 65535")

    normalized_host = hostname.lower()
    is_loopback = normalized_host == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(normalized_host).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ValueError(
            "Ollama endpoint must be loopback-only (127.0.0.1, localhost, or ::1); "
            f"remote host '{hostname}' was rejected"
        )

    authority = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    return f"http://{authority}:{port}"


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    host: str = "http://127.0.0.1:11434"
    connect_timeout_seconds: float = 3.0
    read_timeout_seconds: float = 120.0
    keep_alive: str = "5m"

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", validate_loopback_endpoint(self.host))
        for name, value, maximum in (
            ("connect_timeout_seconds", self.connect_timeout_seconds, 60.0),
            ("read_timeout_seconds", self.read_timeout_seconds, 600.0),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= maximum:
                raise ValueError(f"{name} must be greater than 0 and at most {maximum:g}")
        if not isinstance(self.keep_alive, str) or not self.keep_alive.strip():
            raise ValueError("keep_alive must be a non-empty Ollama duration")
        object.__setattr__(self, "connect_timeout_seconds", float(self.connect_timeout_seconds))
        object.__setattr__(self, "read_timeout_seconds", float(self.read_timeout_seconds))
        object.__setattr__(self, "keep_alive", self.keep_alive.strip())


class OllamaClientProtocol(Protocol):
    def chat(self, **kwargs: Any) -> Any: ...

    def show(self, model: str) -> Any: ...

    def close(self) -> None: ...


ClientFactory = Callable[..., OllamaClientProtocol]


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_int(value: Any, name: str) -> int | None:
    item = _field(value, name)
    return item if isinstance(item, int) and not isinstance(item, bool) else None


def _parameter_schema(parameter: ToolParameter) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": parameter.kind.value,
        "description": parameter.description,
    }
    if parameter.allowed_values:
        schema["enum"] = list(parameter.allowed_values)
    return schema


def _tool_definition_to_ollama(definition: ToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": {
                "type": "object",
                "properties": {
                    parameter.name: _parameter_schema(parameter)
                    for parameter in definition.parameters
                },
                "required": [parameter.name for parameter in definition.parameters if parameter.required],
                "additionalProperties": False,
            },
        },
    }


def _tool_call_to_ollama(call: ToolCall) -> dict[str, object]:
    return {
        "function": {
            "name": call.name,
            "arguments": dict(call.arguments),
        }
    }


def _message_to_ollama(message: ChatMessage) -> dict[str, object]:
    payload: dict[str, object] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [_tool_call_to_ollama(call) for call in message.tool_calls]
    if message.tool_result is not None:
        payload["tool_name"] = message.tool_result.call.name
        payload["content"] = json.dumps(
            message.tool_result.model_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return payload


def _tool_call_from_ollama(value: Any) -> ToolCall:
    function = _field(value, "function")
    name = _field(function, "name") if function is not None else None
    arguments = _field(function, "arguments") if function is not None else None
    if not isinstance(name, str) or not name.strip():
        raise LLMProtocolError("Ollama returned a tool call without a valid function name.")
    if not isinstance(arguments, Mapping):
        raise LLMProtocolError(f"Ollama returned invalid arguments for tool '{name}'.")
    try:
        return ToolCall(name, arguments)
    except (TypeError, ValueError):
        raise LLMProtocolError(f"Ollama returned an invalid tool name '{name}'.") from None


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_PROVIDER_CONTROL_SYNTAX = re.compile(
    r"(?:/(?:no[_-]?think|think)\b|<\|[^|\n]+\|>|\[/?INST\]|</?s>)",
    flags=re.IGNORECASE,
)


def _control_configuration_requested(messages: tuple[ChatMessage, ...]) -> bool:
    user_messages = [message.content for message in messages if message.role.value == "user"]
    if not user_messages:
        return False
    latest = user_messages[-1].casefold()
    return any(
        phrase in latest
        for phrase in (
            "developer configuration",
            "provider control",
            "control token",
            "thinking mode",
            "/think",
            "/no_think",
            "ollama thinking",
            "qwen thinking",
        )
    )


def _visible_answer(text: str, *, allow_provider_controls: bool = False) -> str:
    """Discard tagged reasoning if an older Ollama/model combination emits it."""

    answer = _THINK_BLOCK.sub("", text).strip()
    if not answer or re.search(r"</?think>", answer, flags=re.IGNORECASE):
        raise LLMProtocolError("Ollama returned reasoning without a usable final answer.")
    if not allow_provider_controls and _PROVIDER_CONTROL_SYNTAX.search(answer):
        raise LLMProtocolError(
            "Ollama returned internal control syntax instead of a usable final answer."
        )
    return answer


class OllamaLLM:
    """Official Ollama client constrained to explicit local transport."""

    def __init__(
        self,
        settings: OllamaSettings | None = None,
        *,
        client: OllamaClientProtocol | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        if client is not None and client_factory is not None:
            raise ValueError("provide either client or client_factory, not both")
        self._settings = settings or OllamaSettings()
        self._closed = False
        if client is not None:
            self._client = client
        else:
            factory = client_factory or Client
            timeout = httpx.Timeout(
                self._settings.read_timeout_seconds,
                connect=self._settings.connect_timeout_seconds,
                write=self._settings.connect_timeout_seconds,
                pool=self._settings.connect_timeout_seconds,
            )
            self._client = factory(
                host=self._settings.host,
                timeout=timeout,
                trust_env=False,
            )

    @property
    def name(self) -> str:
        return "Ollama"

    @property
    def endpoint(self) -> str:
        return self._settings.host

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProviderUnavailableError("The local Ollama client is closed.")

    @staticmethod
    def _check_cancellation(cancellation: CancellationToken | None) -> None:
        if cancellation is not None and cancellation.is_cancelled:
            raise LLMInterruptedError("The local model request was cancelled.")

    @staticmethod
    def _raise_response_error(exc: ResponseError, model: str) -> None:
        detail = str(getattr(exc, "error", exc)).strip()
        lowered = detail.lower()
        if getattr(exc, "status_code", None) == 404 or "model" in lowered and "not found" in lowered:
            raise ModelUnavailableError(model) from None
        if "timed out" in lowered or "timeout" in lowered:
            raise LLMTimeoutError(f"The local Ollama request timed out: {detail}") from None
        raise ProviderResponseError(f"Ollama could not complete the request: {detail}") from None

    def ensure_model_available(self, model: str) -> None:
        """Confirm local model metadata exists without pulling or starting inference."""

        self._ensure_open()
        try:
            self._client.show(model)
        except KeyboardInterrupt:
            raise LLMInterruptedError("The local Ollama check was interrupted.") from None
        except httpx.TimeoutException:
            raise LLMTimeoutError("Timed out while checking the local Ollama model.") from None
        except (ConnectionError, httpx.NetworkError):
            raise ProviderUnavailableError(
                f"Ollama is unavailable at {self.endpoint}. Start the local Ollama application and try again."
            ) from None
        except ResponseError as exc:
            self._raise_response_error(exc, model)

    def generate(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> LLMResponse:
        self._ensure_open()
        self._check_cancellation(cancellation)
        request_options: dict[str, object] = {
            "model": request.model,
            "messages": [_message_to_ollama(message) for message in request.messages],
            "stream": False,
            "think": request.thinking,
            "keep_alive": self._settings.keep_alive,
            "options": {"temperature": request.temperature},
        }
        if request.tools:
            request_options["tools"] = [_tool_definition_to_ollama(tool) for tool in request.tools]
        try:
            raw_response = self._client.chat(
                **request_options,
            )
        except KeyboardInterrupt:
            raise LLMInterruptedError("The local model request was interrupted.") from None
        except httpx.TimeoutException:
            raise LLMTimeoutError(
                f"The local Ollama request exceeded the {self._settings.read_timeout_seconds:g}s read timeout."
            ) from None
        except (ConnectionError, httpx.NetworkError):
            raise ProviderUnavailableError(
                f"Ollama is unavailable at {self.endpoint}. Start the local Ollama application and try again."
            ) from None
        except ResponseError as exc:
            self._raise_response_error(exc, request.model)

        self._check_cancellation(cancellation)
        message = _field(raw_response, "message")
        text = _field(message, "content") if message is not None else None
        raw_tool_calls = _field(message, "tool_calls", ()) if message is not None else ()
        if raw_tool_calls is None:
            raw_tool_calls = ()
        if not isinstance(raw_tool_calls, (list, tuple)):
            raise LLMProtocolError("Ollama returned malformed tool calls.")
        tool_calls = tuple(_tool_call_from_ollama(call) for call in raw_tool_calls)
        visible_text = ""
        if isinstance(text, str) and text.strip():
            visible_text = _visible_answer(
                text,
                allow_provider_controls=_control_configuration_requested(
                    request.messages
                ),
            )
        elif text is not None and not isinstance(text, str):
            raise LLMProtocolError("Ollama returned malformed assistant text.")
        if not visible_text and not tool_calls:
            raise LLMProtocolError("Ollama returned an empty assistant response.")
        response_model = _field(raw_response, "model", request.model)
        if not isinstance(response_model, str) or not response_model.strip():
            response_model = request.model
        return LLMResponse(
            text=visible_text,
            model=response_model,
            tool_calls=tool_calls,
            total_duration_ns=_optional_int(raw_response, "total_duration"),
            load_duration_ns=_optional_int(raw_response, "load_duration"),
            prompt_eval_count=_optional_int(raw_response, "prompt_eval_count"),
            eval_count=_optional_int(raw_response, "eval_count"),
            eval_duration_ns=_optional_int(raw_response, "eval_duration"),
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.close()

    def __enter__(self) -> "OllamaLLM":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
