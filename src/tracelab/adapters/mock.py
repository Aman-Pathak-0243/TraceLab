"""Deterministic, offline mock model.

The mock is NOT a language model and does not imitate the quality of one. It is
a scripted stand-in that makes application behaviour reproducible so that the
tracing, evaluation and regression workflow can be exercised without network
access, API keys, model downloads or Ollama.

A `MockScript` supplies the "brain" (a pure function from request to text);
`MockAdapter` adds controlled failure modes on top of it.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, get_args

from tracelab.adapters.base import (
    ModelRequest,
    ModelResponse,
    ModelTimeoutError,
    ModelUnavailableError,
)

MockFault = Literal[
    "malformed_json",  # output is truncated mid-JSON
    "fenced_json",  # valid JSON wrapped in a ```json markdown fence
    "wrong_schema",  # valid JSON that does not match the expected structure
    "hallucinated_tool",  # asks for a tool that does not exist
    "empty_response",  # returns an empty string
    "timeout",  # never answers; the caller's timeout must fire
    "timeout_once",  # hangs on the first call per purpose, then behaves normally
    "unavailable",  # raises ModelUnavailableError
]
MOCK_FAULTS: tuple[str, ...] = get_args(MockFault)

# Faults that replace the scripted output with a script-specific payload.
CONTENT_FAULTS = ("wrong_schema", "hallucinated_tool")

Responder = Callable[[ModelRequest], str]


@dataclass(frozen=True)
class MockScript:
    """Scripted behaviour for one application.

    `responder` must be a pure function of the request so results are
    reproducible. `fault_outputs` holds the payloads for CONTENT_FAULTS, because
    a "wrong schema" only makes sense relative to the application's schema.
    """

    responder: Responder
    fault_outputs: Mapping[str, str] = field(default_factory=dict)


class MockAdapter:
    provider = "mock"

    def __init__(
        self,
        script: MockScript,
        *,
        model: str = "mock-model",
        fault: MockFault | None = None,
        fault_purposes: tuple[str, ...] = ("plan",),
        hang_seconds: float = 30.0,
    ) -> None:
        if fault is not None and fault not in MOCK_FAULTS:
            raise ValueError(f"Unknown mock fault {fault!r}; expected one of {MOCK_FAULTS}")
        if fault in CONTENT_FAULTS and fault not in script.fault_outputs:
            raise ValueError(f"Mock script does not define an output for fault {fault!r}")
        self.script = script
        self.model = model
        self.fault = fault
        self.fault_purposes = fault_purposes
        self.hang_seconds = hang_seconds
        self.requests: list[ModelRequest] = []  # inspected by tests
        self._calls_by_purpose: Counter[str] = Counter()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        call_number = self._calls_by_purpose[request.purpose]
        self._calls_by_purpose[request.purpose] += 1
        fault = self.fault if request.purpose in self.fault_purposes else None

        if fault == "unavailable":
            raise ModelUnavailableError("Simulated: mock model is unavailable")
        if fault == "timeout" or (fault == "timeout_once" and call_number == 0):
            # Simulates a provider that stops responding. Callers are expected to
            # enforce their own timeout and cancel this coroutine.
            await asyncio.sleep(self.hang_seconds)
            raise ModelTimeoutError(f"Simulated: mock model hung for {self.hang_seconds}s")

        text = self.script.responder(request)
        if fault == "malformed_json":
            text = text[: max(1, len(text) // 2)]
        elif fault == "fenced_json":
            text = f"```json\n{text}\n```"
        elif fault == "empty_response":
            text = ""
        elif fault in CONTENT_FAULTS:
            text = self.script.fault_outputs[fault]

        return ModelResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            token_usage=None,  # the mock has no tokenizer; never invent counts
            metadata={"simulated": True, "fault": fault},
        )
