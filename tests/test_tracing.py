"""Span recording, error capture, cancellation and sanitization."""

from __future__ import annotations

import asyncio

import pytest

from tracelab.adapters.base import TokenUsage
from tracelab.tracing.render import format_trace
from tracelab.tracing.sanitize import REDACTED, sanitize
from tracelab.tracing.tracer import Tracer, tool_calls_from_spans


def test_spans_record_parents_order_and_timing() -> None:
    tracer = Tracer()
    with (
        tracer.span("root", "agent"),
        tracer.span("child", "step"),
        tracer.span("grandchild", "tool", tool_name="t"),
    ):
        pass
    with tracer.span("second_root", "agent"):
        pass

    root, child, grandchild, second = tracer.spans
    assert [s.seq for s in tracer.spans] == [0, 1, 2, 3]
    assert root.parent_span_id is None
    assert child.parent_span_id == root.span_id
    assert grandchild.parent_span_id == child.span_id
    assert second.parent_span_id is None
    for span in tracer.spans:
        assert span.status == "ok"
        assert span.trace_id == tracer.trace_id
        assert span.ended_at is not None and span.ended_at >= span.started_at
        assert span.duration_ms is not None and span.duration_ms >= 0


def test_exceptions_are_recorded_on_every_open_span_and_reraised() -> None:
    tracer = Tracer()
    with pytest.raises(KeyError), tracer.span("outer", "step"), tracer.span("inner", "tool"):
        raise KeyError("boom")
    outer, inner = tracer.spans
    assert inner.status == outer.status == "error"
    assert inner.error == {"type": "KeyError", "message": "'boom'"}


def test_error_code_attribute_is_captured() -> None:
    class CodedError(Exception):
        code = "tool_unavailable"

    tracer = Tracer()
    with pytest.raises(CodedError), tracer.span("tool.x", "tool"):
        raise CodedError("down")
    assert tracer.spans[0].error == {
        "type": "CodedError",
        "message": "down",
        "code": "tool_unavailable",
    }


@pytest.mark.anyio
async def test_cancellation_is_recorded_as_cancelled() -> None:
    tracer = Tracer()

    async def slow() -> None:
        with tracer.span("waiting", "model"):
            await asyncio.sleep(10)

    task = asyncio.create_task(slow())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (span,) = tracer.spans
    assert span.status == "cancelled"
    assert span.error is not None and "cancelled" in span.error["message"]


@pytest.mark.anyio
async def test_concurrent_traces_do_not_share_parents() -> None:
    tracers = [Tracer(), Tracer()]

    async def work(tracer: Tracer) -> None:
        with tracer.span("root", "agent"):
            await asyncio.sleep(0.01)
            with tracer.span("child", "step"):
                await asyncio.sleep(0.01)

    await asyncio.gather(*(work(t) for t in tracers))
    for tracer in tracers:
        root, child = tracer.spans
        assert child.parent_span_id == root.span_id


def test_token_usage_is_recorded_only_when_reported() -> None:
    tracer = Tracer()
    with tracer.span("a", "model") as with_usage:
        with_usage.record_usage(TokenUsage(10, 4, source="ollama"))
    with tracer.span("b", "model") as without_usage:
        without_usage.record_usage(None)
    assert with_usage.token_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "source": "ollama",
    }
    assert without_usage.token_usage is None


def test_sanitize_redacts_secrets_and_truncates() -> None:
    clean = sanitize(
        {
            "api_key": "abc",
            "Authorization": "Bearer abcdefghijkl",
            "nested": {"password": "hunter2", "note": "token Bearer abcdefghijklmnop end"},
            "openai": "sk-abcdefghijklmnopqrstuvwxyz",
            "long": "x" * 600,
            "items": list(range(60)),
            "ok": "M-1001",
        }
    )
    assert clean["api_key"] == REDACTED
    assert clean["Authorization"] == REDACTED
    assert clean["nested"]["password"] == REDACTED
    assert clean["nested"]["note"] == f"token {REDACTED} end"
    assert clean["openai"] == REDACTED
    assert clean["long"].endswith("[truncated 100 chars]")
    assert len(clean["items"]) == 51
    assert clean["ok"] == "M-1001"


def test_span_attributes_are_sanitized() -> None:
    tracer = Tracer()
    with tracer.span("call", "tool", arguments={"secret": "s3cr3t", "x": 1}) as span:
        span.set(output={"access_token": "abc"})
    assert span.attributes == {
        "arguments": {"secret": REDACTED, "x": 1},
        "output": {"access_token": REDACTED},
    }


def test_tool_calls_are_extracted_in_order_with_status() -> None:
    tracer = Tracer()
    with tracer.span("tool.a", "tool", tool_name="a", arguments={"x": 1}, idempotent=True):
        pass
    with (
        pytest.raises(RuntimeError),
        tracer.span("tool.b", "tool", tool_name="b", idempotent=False),
    ):
        raise RuntimeError("fail")
    calls = tool_calls_from_spans([s.to_dict() for s in tracer.spans])
    assert [(c["tool"], c["status"], c["idempotent"]) for c in calls] == [
        ("a", "ok", True),
        ("b", "error", False),
    ]


def test_format_trace_renders_tree() -> None:
    tracer = Tracer()
    with tracer.span("root", "agent"), tracer.span("child", "tool"):
        pass
    text = format_trace([s.to_dict() for s in tracer.spans])
    assert "[ok ] root (agent)" in text
    assert "[ok ]   child (tool)" in text
