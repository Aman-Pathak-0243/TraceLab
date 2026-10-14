"""Step-level execution tracing.

One `Tracer` records one execution (one trace). Each step is a `Span` with a
parent, a sequence number, wall-clock timestamps, a monotonic duration, a
status and sanitized attributes.

Usage:

    tracer = Tracer()
    with tracer.span("tool.lookup_appointment", "tool", arguments=args) as span:
        result = lookup(**args)
        span.set(output=result)

Exceptions raised inside a span are recorded on the span and then re-raised;
the tracer never swallows errors.
"""

from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from tracelab.adapters.base import TokenUsage
from tracelab.tracing.sanitize import sanitize, truncate

SpanKind = Literal["agent", "step", "policy", "model", "validation", "tool"]
SpanStatus = Literal["running", "ok", "error", "cancelled"]

# (trace_id, span_id) of the innermost open span in the current task/context.
# A ContextVar keeps parent tracking correct when several traces run
# concurrently in different asyncio tasks.
_current_span: ContextVar[tuple[str, str] | None] = ContextVar(
    "tracelab_current_span", default=None
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_span_id: str | None
    seq: int
    name: str
    kind: str
    started_at: str
    ended_at: str | None = None
    duration_ms: float | None = None
    status: str = "running"
    attributes: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    token_usage: dict[str, Any] | None = None
    _start_ns: int = field(default=0, repr=False)

    def set(self, **attributes: Any) -> None:
        """Attach sanitized attributes to the span."""
        self.attributes.update(sanitize(attributes))

    def record_usage(self, usage: TokenUsage | None) -> None:
        """Record token usage only if the provider actually reported it."""
        self.token_usage = usage.to_dict() if usage is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "seq": self.seq,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "error": self.error,
            "token_usage": self.token_usage,
        }


def _error_info(exc: BaseException) -> dict[str, Any]:
    info: dict[str, Any] = {"type": type(exc).__name__, "message": truncate(str(exc), 1000)}
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        info["code"] = code
    return sanitize(info)


class Tracer:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or new_id("tr")
        self._spans: list[Span] = []
        self._seq = itertools.count()

    @property
    def spans(self) -> list[Span]:
        """Spans in start order (seq), parents always before their children."""
        return list(self._spans)

    @contextmanager
    def span(self, name: str, kind: SpanKind, **attributes: Any) -> Iterator[Span]:
        current = _current_span.get()
        parent_id = current[1] if current and current[0] == self.trace_id else None
        span = Span(
            span_id=new_id("sp"),
            trace_id=self.trace_id,
            parent_span_id=parent_id,
            seq=next(self._seq),
            name=name,
            kind=kind,
            started_at=utc_now(),
            _start_ns=time.perf_counter_ns(),
        )
        span.set(**attributes)
        self._spans.append(span)
        token = _current_span.set((self.trace_id, span.span_id))
        try:
            yield span
        except asyncio.CancelledError as exc:
            span.status = "cancelled"
            span.error = _error_info(exc) | {
                "message": "task was cancelled before the step finished"
            }
            raise
        except BaseException as exc:
            span.status = "error"
            span.error = _error_info(exc)
            raise
        else:
            if span.status == "running":
                span.status = "ok"
        finally:
            span.ended_at = utc_now()
            span.duration_ms = round((time.perf_counter_ns() - span._start_ns) / 1_000_000, 3)
            _current_span.reset(token)

    def root(self) -> Span | None:
        return next((s for s in self._spans if s.parent_span_id is None), None)

    def tool_calls(self) -> list[dict[str, Any]]:
        return tool_calls_from_spans([s.to_dict() for s in self._spans])


def tool_calls_from_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract executed tool calls (in order) from serialized spans."""
    calls = []
    for span in sorted(spans, key=lambda s: s["seq"]):
        if span["kind"] != "tool":
            continue
        attrs = span.get("attributes") or {}
        calls.append(
            {
                "tool": attrs.get("tool_name"),
                "arguments": attrs.get("arguments", {}),
                "idempotent": attrs.get("idempotent"),
                "status": span["status"],
                "error": span.get("error"),
                "span_id": span["span_id"],
            }
        )
    return calls
