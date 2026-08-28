from types import SimpleNamespace

import httpx
import pytest
from ollama import ResponseError

from jarvis.llm.base import (
    CancellationToken,
    ChatMessage,
    LLMInterruptedError,
    LLMRequest,
    LLMTimeoutError,
    MessageRole,
    ModelUnavailableError,
    ProviderUnavailableError,
)
from jarvis.llm.ollama import OllamaLLM, OllamaSettings, validate_loopback_endpoint


class FakeClient:
    def __init__(self, *, response=None, error=None):
        self.response = response or {
            "model": "qwen3:8b",
            "message": {"content": "Local answer"},
            "eval_count": 2,
            "eval_duration": 1_000_000_000,
        }
        self.error = error
        self.chat_kwargs = None
        self.shown_model = None
        self.closed = False

    def chat(self, **kwargs):
        self.chat_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response

    def show(self, model):
        self.shown_model = model
        if self.error is not None:
            raise self.error
        return {"model": model}

    def close(self):
        self.closed = True


def request(*, thinking=False):
    return LLMRequest(
        model="qwen3:8b",
        messages=(ChatMessage(MessageRole.USER, "Hello"),),
        thinking=thinking,
    )


@pytest.mark.parametrize(
    ("value", "normalized"),
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://127.9.8.7", "http://127.9.8.7:11434"),
        ("http://LOCALHOST:11434/", "http://localhost:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_loopback_endpoints_are_accepted(value, normalized):
    assert validate_loopback_endpoint(value) == normalized


@pytest.mark.parametrize(
    "value",
    [
        "https://127.0.0.1:11434",
        "http://192.168.1.10:11434",
        "http://example.com:11434",
        "http://localhost:11434/api/chat",
        "http://user:pass@localhost:11434",
        "http://127.0.0.1:0",
        "not-a-url",
    ],
)
def test_remote_or_unsafe_endpoints_are_rejected(value):
    with pytest.raises(ValueError):
        validate_loopback_endpoint(value)


def test_official_client_is_constructed_with_explicit_local_settings(monkeypatch):
    captured = {}
    monkeypatch.setenv("OLLAMA_HOST", "https://remote.example")

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeClient()

    OllamaLLM(OllamaSettings(), client_factory=factory)

    assert captured["host"] == "http://127.0.0.1:11434"
    assert captured["trust_env"] is False
    assert isinstance(captured["timeout"], httpx.Timeout)
    assert captured["timeout"].connect == 3.0
    assert captured["timeout"].read == 120.0


def test_tagged_reasoning_is_not_returned_to_conversation():
    client = FakeClient(
        response={"model": "qwen3:8b", "message": {"content": "<think>private steps</think>Final answer"}}
    )
    provider = OllamaLLM(client=client)

    response = provider.generate(request())

    assert response.text == "Final answer"


def test_request_uses_top_level_think_and_keep_alive():
    client = FakeClient()
    provider = OllamaLLM(OllamaSettings(keep_alive="10m"), client=client)

    response = provider.generate(request())

    assert response.text == "Local answer"
    assert response.eval_count == 2
    assert client.chat_kwargs["think"] is False
    assert client.chat_kwargs["keep_alive"] == "10m"
    assert "options" not in client.chat_kwargs
    assert client.chat_kwargs["messages"] == [{"role": "user", "content": "Hello"}]


def test_thinking_can_be_enabled_explicitly():
    client = FakeClient()
    provider = OllamaLLM(client=client)

    provider.generate(request(thinking=True))

    assert client.chat_kwargs["think"] is True


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ConnectionError("no connection"), ProviderUnavailableError),
        (httpx.ReadTimeout("slow"), LLMTimeoutError),
        (ResponseError("model 'qwen3:8b' not found", 404), ModelUnavailableError),
        (KeyboardInterrupt(), LLMInterruptedError),
    ],
)
def test_expected_errors_are_mapped(error, expected):
    provider = OllamaLLM(client=FakeClient(error=error))

    with pytest.raises(expected) as raised:
        provider.generate(request())

    if expected is ModelUnavailableError:
        assert raised.value.manual_command == "ollama pull qwen3:8b"


def test_model_check_maps_missing_model_without_pull():
    provider = OllamaLLM(client=FakeClient(error=ResponseError("not found", 404)))

    with pytest.raises(ModelUnavailableError, match="ollama pull qwen3:8b"):
        provider.ensure_model_available("qwen3:8b")


def test_pre_cancelled_request_never_calls_client():
    client = FakeClient()
    provider = OllamaLLM(client=client)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(LLMInterruptedError):
        provider.generate(request(), cancellation=token)

    assert client.chat_kwargs is None


def test_object_response_is_supported_and_client_closes():
    raw = SimpleNamespace(
        model="qwen3:8b",
        message=SimpleNamespace(content="Object answer"),
        total_duration=10,
    )
    client = FakeClient(response=raw)
    provider = OllamaLLM(client=client)

    response = provider.generate(request())
    provider.close()

    assert response.text == "Object answer"
    assert response.total_duration_ns == 10
    assert client.closed is True
