"""CareDesk agent behaviour end to end with the mock model, including failures."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tracelab.adapters.base import ModelRequest, ModelResponse, ModelUnavailableError
from tracelab.adapters.mock import MockAdapter
from tracelab.datasets.schema import ToolFault
from tracelab.demo_app.agent import CareDeskAgent, InvalidModelOutputError, parse_plan
from tracelab.demo_app.mock_script import CAREDESK_MOCK_SCRIPT
from tracelab.demo_app.schemas import AgentRequest, AgentResponse
from tracelab.demo_app.tools import ToolRegistry, ToolSpec
from tracelab.tracing.tracer import Tracer

pytestmark = pytest.mark.anyio

MEMBER = {"member_id": "M-1001"}


async def run_agent(
    message: str,
    context: dict[str, str] | None = None,
    *,
    fault: str | None = None,
    tool_fault: ToolFault | None = None,
    adapter: Any = None,
    tools: ToolRegistry | None = None,
    timeout_s: float = 0.05,
) -> tuple[AgentResponse, Tracer]:
    agent = CareDeskAgent(
        adapter or MockAdapter(CAREDESK_MOCK_SCRIPT, fault=fault),
        tools or ToolRegistry(fault=tool_fault),
        model_timeout_s=timeout_s,
    )
    tracer = Tracer()
    response = await agent.handle(AgentRequest(message=message, context=context or {}), tracer)
    return response, tracer


def span_names(tracer: Tracer) -> list[str]:
    return [s.name for s in tracer.spans]


@pytest.mark.parametrize(
    ("message", "context", "action", "tools_used", "error_code"),
    [
        ("What packages do you offer?", {}, "answer", ["list_packages"], None),
        ("When is my appointment APT-1001?", MEMBER, "answer", ["lookup_appointment"], None),
        (
            "Book the Basic Wellness Screening on 2026-11-20",
            MEMBER,
            "answer",
            ["book_appointment"],
            None,
        ),
        ("What is your cancellation policy?", {}, "answer", ["get_policy"], None),
        ("Tell me a joke", {}, "answer", [], None),
        ("When is my appointment?", MEMBER, "request_info", [], None),
        ("Book the Heart Health Check", MEMBER, "request_info", [], None),
        ("Book the Heart Health Check on 2026-11-20", {}, "request_info", [], None),
        ("Status of APT-9999?", MEMBER, "error", [], "not_found"),
        ("Book the Heart Health Check on 2026-10-01", MEMBER, "error", [], "date_in_past"),
        ("Which medication should I take?", {}, "refuse", [], None),
    ],
)
async def test_scenarios(
    message: str,
    context: dict[str, str],
    action: str,
    tools_used: list[str],
    error_code: str | None,
) -> None:
    response, _ = await run_agent(message, context)
    assert response.action == action
    assert response.tools_used == tools_used
    assert (response.error.code if response.error else None) == error_code


async def test_missing_date_produces_request_info_without_calling_the_tool() -> None:
    response, tracer = await run_agent("Please book the Heart Health Check", MEMBER)
    assert response.missing_fields == ["preferred_date"]
    assert "preferred date" in response.answer
    assert tracer.tool_calls() == []
    validation = next(s for s in tracer.spans if s.name == "validate.tool_call")
    assert validation.attributes["outcome"] == "missing_arguments"
    assert validation.attributes["missing"] == ["preferred_date"]


async def test_refusal_happens_before_any_model_call() -> None:
    adapter = MockAdapter(CAREDESK_MOCK_SCRIPT)
    response, tracer = await run_agent("Is it serious? What dose should I take?", adapter=adapter)
    assert response.action == "refuse"
    assert adapter.requests == []
    assert span_names(tracer) == ["agent.handle", "policy.screen_input"]


@pytest.mark.parametrize(
    ("fault", "code"),
    [
        ("malformed_json", "invalid_model_output"),
        ("wrong_schema", "invalid_model_output"),
        ("empty_response", "invalid_model_output"),
        ("hallucinated_tool", "unknown_tool"),
        ("timeout", "model_timeout"),
        ("unavailable", "model_unavailable"),
    ],
)
async def test_model_failures_become_controlled_errors(fault: str, code: str) -> None:
    response, tracer = await run_agent("What packages do you offer?", fault=fault)
    assert response.action == "error"
    assert response.error is not None and response.error.code == code
    assert tracer.tool_calls() == []  # nothing executed on bad model output
    assert all(s.status != "running" for s in tracer.spans)


async def test_hallucinated_tool_is_recorded_but_never_executed() -> None:
    _, tracer = await run_agent("Please clean up my account.", fault="hallucinated_tool")
    validation = next(s for s in tracer.spans if s.name == "validate.tool_call")
    assert validation.attributes["tool"] == "delete_member_records"
    assert validation.attributes["outcome"] == "unknown_tool"
    assert "tool.delete_member_records" not in span_names(tracer)


async def test_invalid_output_marks_parse_step_with_the_raw_output() -> None:
    _, tracer = await run_agent("What packages do you offer?", fault="malformed_json")
    parse = next(s for s in tracer.spans if s.name == "parse.plan")
    assert parse.status == "error"
    assert parse.error is not None and parse.error["type"] == "InvalidModelOutputError"
    assert parse.attributes["raw_output"].startswith('{"intent"')
    plan = next(s for s in tracer.spans if s.name == "plan")
    assert plan.status == "error"  # the failure propagates up the step tree
    assert tracer.root().status == "ok"  # ...and is handled at the agent level


async def test_fenced_json_is_accepted() -> None:
    response, _ = await run_agent("What is your cancellation policy?", fault="fenced_json")
    assert response.action == "answer"
    assert response.data["policy"]["topic"] == "cancellation"


async def test_timeout_is_retried_and_recovers() -> None:
    response, tracer = await run_agent("What packages do you offer?", fault="timeout_once")
    assert response.action == "answer"
    plan_attempts = [
        s for s in tracer.spans if s.name == "llm.generate" and s.attributes["purpose"] == "plan"
    ]
    assert [(s.attributes["attempt"], s.status) for s in plan_attempts] == [(1, "error"), (2, "ok")]
    assert (
        plan_attempts[0].error is not None and plan_attempts[0].error["type"] == "ModelTimeoutError"
    )


async def test_persistent_timeout_uses_all_attempts() -> None:
    _, tracer = await run_agent("What packages do you offer?", fault="timeout")
    attempts = [s for s in tracer.spans if s.name == "llm.generate"]
    assert [s.status for s in attempts] == ["error", "error"]


async def test_unavailable_model_is_not_retried() -> None:
    _, tracer = await run_agent("What packages do you offer?", fault="unavailable")
    assert len([s for s in tracer.spans if s.name == "llm.generate"]) == 1


async def test_tool_failure_is_reported_without_fabricated_data() -> None:
    response, tracer = await run_agent(
        "When is my appointment APT-1001?",
        MEMBER,
        tool_fault=ToolFault(tool="lookup_appointment"),
    )
    assert response.action == "error"
    assert response.error is not None and response.error.code == "tool_unavailable"
    assert "2026-11-03" not in response.answer
    assert response.data == {}
    tool_span = next(s for s in tracer.spans if s.kind == "tool")
    assert tool_span.status == "error"
    assert tool_span.error is not None and tool_span.error["code"] == "tool_unavailable"


class _ComposeFails:
    """Real plan, but the compose call fails: the model went away mid-request."""

    provider = "test"
    model = "compose-fails"

    def __init__(self) -> None:
        self.inner = MockAdapter(CAREDESK_MOCK_SCRIPT)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if request.purpose == "compose":
            raise ModelUnavailableError("went away")
        return await self.inner.generate(request)


async def test_completed_booking_is_not_lost_when_compose_fails() -> None:
    response, _ = await run_agent(
        "Book the Basic Wellness Screening on 2026-11-20", MEMBER, adapter=_ComposeFails()
    )
    assert response.action == "answer"
    assert response.warnings == ["compose_failed:model_unavailable"]
    assert response.data["booking"]["booking_id"] == "BK-BASIC-20261120"
    assert "BK-BASIC-20261120" in response.answer


async def test_compose_failure_without_tool_results_is_an_error() -> None:
    response, _ = await run_agent("Tell me a joke", adapter=_ComposeFails())
    assert response.action == "error"
    assert response.error is not None and response.error.code == "model_unavailable"


class _FixedPlan:
    provider = "test"
    model = "fixed-plan"

    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan

    async def generate(self, request: ModelRequest) -> ModelResponse:
        text = json.dumps(self.plan) if request.purpose == "plan" else "done"
        return ModelResponse(text=text, provider=self.provider, model=self.model)


async def test_unexpected_tool_arguments_are_rejected() -> None:
    plan = {
        "intent": "x",
        "tool_calls": [{"name": "get_policy", "arguments": {"topic": "payment", "sql": "drop"}}],
    }
    response, tracer = await run_agent("x", adapter=_FixedPlan(plan))
    assert response.error is not None and response.error.code == "invalid_tool_arguments"
    assert tracer.tool_calls() == []


async def test_unexpected_exceptions_propagate_and_are_traced() -> None:
    def broken() -> dict[str, Any]:
        raise RuntimeError("bug in tool")

    tools = ToolRegistry(specs=(ToolSpec("list_packages", "broken", (), broken),))
    with pytest.raises(RuntimeError, match="bug in tool"):
        await run_agent("What packages do you offer?", tools=tools)


async def test_unknown_prompt_version_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown prompt version"):
        CareDeskAgent(MockAdapter(CAREDESK_MOCK_SCRIPT), prompt_version="v9")


async def test_trace_is_complete_and_ordered() -> None:
    _, tracer = await run_agent("Book the Basic Wellness Screening on 2026-11-20", MEMBER)
    assert span_names(tracer) == [
        "agent.handle",
        "policy.screen_input",
        "plan",
        "llm.generate",
        "parse.plan",
        "validate.tool_call",
        "tool.book_appointment",
        "compose",
        "llm.generate",
        "parse.answer",
    ]
    ids = {s.span_id for s in tracer.spans}
    for span in tracer.spans[1:]:
        assert span.parent_span_id in ids
    assert all(s.status == "ok" for s in tracer.spans)
    root = tracer.root()
    assert root is not None and root.attributes["action"] == "answer"
    assert root.attributes["tools_used"] == ["book_appointment"]


def test_parse_plan_variants() -> None:
    assert parse_plan('{"intent": "x", "tool_calls": []}').intent == "x"
    assert parse_plan('```json\n{"intent": "x"}\n```').tool_calls == []
    assert parse_plan('{"intent": "x", "reasoning": "extra keys are ignored"}').intent == "x"
    for bad in [
        "",
        "   ",
        '{"intent": ',
        '{"tool_calls": []}',
        '["x"]',
        '{"intent": "x", "tool_calls": "t"}',
    ]:
        with pytest.raises(InvalidModelOutputError):
            parse_plan(bad)


def test_agent_response_contract_is_enforced() -> None:
    with pytest.raises(ValueError, match="missing_fields"):
        AgentResponse(action="request_info", answer="?")
    with pytest.raises(ValueError, match="error details"):
        AgentResponse(action="error", answer="oops")
    with pytest.raises(ValueError):
        AgentResponse(action="answer", answer="")
