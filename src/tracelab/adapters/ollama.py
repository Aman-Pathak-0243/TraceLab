"""Optional adapter for a locally running Ollama server.

TraceLab never downloads models. If the configured model is missing, the
adapter raises a `ModelResponseError` that tells the user to pull it manually.
Uses Ollama's /api/chat endpoint with `stream: false`.
"""

from __future__ import annotations

from typing import Any

import httpx

from tracelab.adapters.base import (
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
    TokenUsage,
)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


class OllamaAdapter:
    provider = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout_s: float = 60.0,
        seed: int | None = 0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not model:
            raise ValueError("An Ollama model name is required (e.g. TRACELAB_OLLAMA_MODEL=llama3)")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.seed = seed
        # Injected in tests (httpx.MockTransport) so no live server is needed.
        self._transport = transport

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": request.temperature}
        if self.seed is not None:
            options["seed"] = self.seed
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": options,
        }
        if request.response_format == "json":
            # Ollama accepts either "json" or a JSON schema for structured output.
            # The application still validates the result; this only constrains decoding.
            payload["format"] = request.json_schema or "json"
        return payload

    async def generate(self, request: ModelRequest) -> ModelResponse:
        url = f"{self.base_url}/api/chat"
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self.timeout_s
            ) as client:
                response = await client.post(url, json=self._payload(request))
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(
                f"Ollama did not respond within {self.timeout_s}s ({type(exc).__name__})"
            ) from exc
        except httpx.TransportError as exc:
            raise ModelUnavailableError(
                f"Could not reach Ollama at {self.base_url} ({type(exc).__name__}). "
                "Is `ollama serve` running?"
            ) from exc

        if response.status_code == 404:
            raise ModelResponseError(
                f"Ollama model {self.model!r} was not found (HTTP 404). Pull it yourself with "
                f"`ollama pull {self.model}`; TraceLab never downloads models."
            )
        if response.status_code >= 400:
            raise ModelResponseError(
                f"Ollama returned HTTP {response.status_code}: {_error_text(response)}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ModelResponseError("Ollama returned a non-JSON response body") from exc

        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ModelResponseError("Ollama response is missing message.content")

        return ModelResponse(
            text=content,
            provider=self.provider,
            model=str(data.get("model") or self.model),
            token_usage=_token_usage(data),
            metadata={
                key: data[key]
                for key in ("done_reason", "total_duration", "load_duration", "eval_duration")
                if key in data
            },
        )


def _token_usage(data: dict[str, Any]) -> TokenUsage | None:
    """Map Ollama's eval counters to TokenUsage, keeping missing values as None.

    Ollama can omit `prompt_eval_count` (e.g. when the prompt is served from its
    cache), so each field is passed through independently instead of defaulted.
    """
    prompt = data.get("prompt_eval_count")
    completion = data.get("eval_count")
    prompt = prompt if isinstance(prompt, int) else None
    completion = completion if isinstance(completion, int) else None
    if prompt is None and completion is None:
        return None
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, source="ollama")


def _error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict) and "error" in body:
            return str(body["error"])[:300]
    except ValueError:
        pass
    return response.text[:300]
