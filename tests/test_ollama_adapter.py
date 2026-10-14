"""Ollama adapter behaviour, tested against httpx.MockTransport (no live server).

The final test talks to a real local Ollama and is skipped unless
TRACELAB_OLLAMA_TESTS=1 is set.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

from tracelab.adapters.base import (
    Message,
    ModelRequest,
    ModelResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from tracelab.adapters.factory import available_providers, create_adapter
from tracelab.adapters.ollama import OllamaAdapter
from tracelab.config import AdapterConfig, Settings

pytestmark = pytest.mark.anyio

# Captured at import time because the autouse fixture clears TRACELAB_* variables.
LIVE_MODEL = os.environ.get("TRACELAB_OLLAMA_MODEL", "llama3")
LIVE_URL = os.environ.get("TRACELAB_OLLAMA_BASE_URL", "http://127.0.0.1:11434")

REQUEST = ModelRequest(
    messages=[Message("system", "be brief"), Message("user", "hello")],
    purpose="plan",
    response_format="json",
    json_schema={"type": "object"},
)


def adapter_with(handler) -> OllamaAdapter:
    return OllamaAdapter("test-model", transport=httpx.MockTransport(handler), timeout_s=5)


async def test_successful_call_sends_expected_payload_and_parses_usage() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "message": {"role": "assistant", "content": '{"ok": true}'},
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 5,
            },
        )

    response = await adapter_with(handler).generate(REQUEST)

    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    body = seen["body"]
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body["format"] == {"type": "object"}
    assert body["options"] == {"temperature": 0.0, "seed": 0}
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert response.text == '{"ok": true}'
    assert response.token_usage is not None
    assert response.token_usage.to_dict() == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
        "source": "ollama",
    }
    assert response.metadata == {"done_reason": "stop"}


async def test_text_requests_do_not_constrain_the_format() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "hi"}})

    text_request = ModelRequest(messages=[Message("user", "hi")], purpose="compose")
    await adapter_with(handler).generate(text_request)
    assert "format" not in seen


async def test_missing_prompt_count_stays_none_instead_of_zero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "x"}, "eval_count": 3})

    usage = (await adapter_with(handler).generate(REQUEST)).token_usage
    assert usage is not None
    assert usage.prompt_tokens is None
    assert usage.completion_tokens == 3
    assert usage.total_tokens is None


async def test_no_counts_means_no_token_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "x"}})

    assert (await adapter_with(handler).generate(REQUEST)).token_usage is None


async def test_connection_failure_becomes_model_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ModelUnavailableError, match="ollama serve"):
        await adapter_with(handler).generate(REQUEST)


async def test_http_timeout_becomes_model_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ModelTimeoutError):
        await adapter_with(handler).generate(REQUEST)


async def test_missing_model_explains_how_to_pull_it_without_downloading() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(404, json={"error": "model 'test-model' not found"})

    with pytest.raises(ModelResponseError, match="ollama pull test-model"):
        await adapter_with(handler).generate(REQUEST)
    assert calls == ["/api/chat"]  # no call to /api/pull


async def test_server_error_includes_the_error_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "out of memory"})

    with pytest.raises(ModelResponseError, match="HTTP 500: out of memory"):
        await adapter_with(handler).generate(REQUEST)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"message": {}}),
        httpx.Response(200, json=["unexpected"]),
    ],
)
async def test_unusable_bodies_raise_model_response_error(response: httpx.Response) -> None:
    with pytest.raises(ModelResponseError):
        await adapter_with(lambda request: response).generate(REQUEST)


def test_factory_uses_configured_model_or_environment_default(settings: Settings) -> None:
    configured = create_adapter(AdapterConfig(provider="ollama", model="qwen2.5:3b"), settings)
    defaulted = create_adapter(AdapterConfig(provider="ollama"), settings)
    assert configured.model == "qwen2.5:3b"
    assert defaulted.model == settings.ollama_model


def test_factory_rejects_unknown_provider(settings: Settings) -> None:
    assert available_providers() == ["mock", "ollama"]
    with pytest.raises(ValueError, match="Unknown model provider 'openai'"):
        create_adapter(AdapterConfig(provider="openai"), settings)


def test_settings_read_model_name_from_environment() -> None:
    settings = Settings.from_env({"TRACELAB_OLLAMA_MODEL": "phi3:mini"})
    assert settings.ollama_model == "phi3:mini"


@pytest.mark.ollama
async def test_live_ollama_returns_text_and_reported_usage() -> None:
    adapter = OllamaAdapter(LIVE_MODEL, base_url=LIVE_URL, timeout_s=300)
    response = await adapter.generate(
        ModelRequest(
            messages=[Message("user", 'Reply with the JSON object {"ok": true} and nothing else.')],
            purpose="plan",
            response_format="json",
        )
    )
    assert isinstance(json.loads(response.text), dict)
    assert response.provider == "ollama"
