"""Model adapter interface shared by every model provider.

The demo application and the LLM judge depend only on `ModelAdapter`, so the
provider (deterministic mock, local Ollama, or a future hosted API) can be
swapped through configuration without touching application or evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class ModelRequest:
    """A provider-neutral model call.

    `purpose` ("plan", "compose", "judge") is recorded in traces. Real providers
    ignore it; the mock adapter uses it to pick which scripted behaviour to run.
    """

    messages: list[Message]
    purpose: str
    response_format: Literal["text", "json"] = "text"
    json_schema: dict[str, Any] | None = None
    temperature: float = 0.0


@dataclass(frozen=True)
class TokenUsage:
    """Token counts exactly as reported by the provider.

    Fields are None when the provider did not report them. TraceLab never
    estimates or invents token counts.
    """

    prompt_tokens: int | None
    completion_tokens: int | None
    source: str

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "source": self.source,
        }


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider: str
    model: str
    token_usage: TokenUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelError(Exception):
    """Base class for provider failures. `code` is a stable machine-readable label."""

    code = "model_error"


class ModelUnavailableError(ModelError):
    """The provider could not be reached (e.g. Ollama is not running)."""

    code = "model_unavailable"


class ModelTimeoutError(ModelError):
    """The provider did not answer within the configured timeout."""

    code = "model_timeout"


class ModelResponseError(ModelError):
    """The provider answered, but with an error or an unusable payload."""

    code = "model_response_error"


@runtime_checkable
class ModelAdapter(Protocol):
    """Anything that can turn a `ModelRequest` into a `ModelResponse`.

    Implementations must raise a `ModelError` subclass for provider failures
    instead of returning partial or placeholder responses.
    """

    provider: str
    model: str

    async def generate(self, request: ModelRequest) -> ModelResponse: ...
