"""Deterministic evaluators, business rules and the LLM judge."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import tracelab.demo_app.rules  # noqa: F401  (registers business rules)
from tracelab.adapters.base import ModelRequest, ModelResponse, ModelUnavailableError
from tracelab.adapters.mock import MockAdapter, MockScript
from tracelab.datasets.schema import EvalCase
from tracelab.demo_app.schemas import AgentResponse
from tracelab.evaluation.deterministic import (
    MISSING,
    evaluators_for,
    get_path,
    registered_evaluator_ids,
)
from tracelab.evaluation.judge import MOCK_JUDGE_SCRIPT, LLMJudgeEvaluator
from tracelab.evaluation.rules import business_rule, get_rule
from tracelab.evaluation.types import CaseExecution, EvalResult
from tracelab.tracing.tracer import Tracer


def make_case(expected: dict[str, Any], **extra: Any) -> EvalCase:
    return EvalCase.model_validate(
        {
            "schema_version": "1.0",
            "case_id": "test-case-001",
            "description": "unit test case",
            "input": extra.pop("input", {"message": "Please book the Heart Health Check"}),
            "expected": expected,
            **extra,
        }
    )


def tool_spans(*calls: tuple[str, dict[str, Any], bool, str]) -> list[dict[str, Any]]:
    tracer = Tracer()
    for name, arguments, idempotent, status in calls:
        try:
            with tracer.span(
                f"tool.{name}", "tool", tool_name=name, arguments=arguments, idempotent=idempotent
            ):
                if status == "error":
                    raise RuntimeError("tool failed")
        except RuntimeError:
            pass
    return [s.to_dict() for s in tracer.spans]


def run(
    case: EvalCase, output: dict[str, Any] | None, spans: list[dict[str, Any]] | None = None
) -> dict[str, EvalResult]:
    execution = CaseExecution(case, output, spans or [])
    return {
        r.evaluator_id: r
        for r in (ev.evaluate(execution) for ev in evaluators_for(case, AgentResponse))
    }


ANSWER = {
    "action": "answer",
    "answer": "Your Heart Health Check is booked for 2026-11-20.",
    "missing_fields": [],
    "data": {"booking": {"status": "confirmed", "date": "2026-11-20"}, "items": [{"id": "a"}]},
    "error": None,
    "tools_used": ["book_appointment"],
    "warnings": [],
}


def test_evaluators_are_selected_from_expected_fields() -> None:
    case = make_case(
        {"action": "answer", "forbidden_tools": ["x"], "business_rules": ["refusal_runs_no_tools"]}
    )
    ids = [ev.result_id for ev in evaluators_for(case, AgentResponse)]
    assert ids == [
        "output_schema",
        "expected_action",
        "forbidden_tools",
        "business_rule:refusal_runs_no_tools",
    ]
    assert "output_schema" not in [e.result_id for e in evaluators_for(case, None)]
    assert "forbidden_tools" not in [
        e.result_id for e in evaluators_for(case, AgentResponse, disabled=["forbidden_tools"])
    ]
    assert set(registered_evaluator_ids()) >= {"expected_action", "missing_fields", "error_code"}


def test_valid_json_alone_does_not_pass_a_case() -> None:
    case = make_case({"action": "request_info", "missing_fields": ["preferred_date"]})
    results = run(case, ANSWER)
    assert results["output_schema"].passed is True
    assert results["expected_action"].passed is False
    assert results["missing_fields"].passed is False
    assert results["missing_fields"].explanation == "did not request ['preferred_date']"


def test_output_schema_reports_violations() -> None:
    case = make_case({"action": "answer"})
    results = run(case, {"action": "maybe", "answer": ""})
    assert results["output_schema"].passed is False
    assert results["output_schema"].details["errors"]
    assert run(case, None)["output_schema"].explanation == "no output was produced"


def test_tool_evaluators_read_the_trace() -> None:
    case = make_case(
        {
            "action": "answer",
            "required_tools": ["book_appointment"],
            "forbidden_tools": ["lookup_appointment"],
        }
    )
    ok = run(case, ANSWER, tool_spans(("book_appointment", {}, False, "ok")))
    assert ok["required_tools"].passed and ok["forbidden_tools"].passed

    failed_call = run(case, ANSWER, tool_spans(("book_appointment", {}, False, "error")))
    assert (
        failed_call["required_tools"].explanation
        == "book_appointment was called but did not succeed"
    )

    missing = run(case, ANSWER, tool_spans(("lookup_appointment", {}, True, "error")))
    assert missing["required_tools"].explanation == "book_appointment was not called"
    # Even a failed attempt of a forbidden tool counts: the call should never have happened.
    assert missing["forbidden_tools"].passed is False


def test_field_values_and_answer_checks() -> None:
    case = make_case(
        {
            "action": "answer",
            "fields": {"data.booking.status": "confirmed", "data.items.0.id": "a", "data.nope": 1},
            "answer_contains": ["booked", "2026-11-21"],
            "answer_not_contains": ["HEART"],
        }
    )
    results = run(case, ANSWER)
    assert results["field_values"].passed is False
    assert results["field_values"].details["mismatches"] == {
        "data.nope": {"expected": 1, "actual": "<missing>"}
    }
    assert results["answer_contains"].explanation == "answer is missing ['2026-11-21']"
    assert results["answer_not_contains"].explanation == "answer contains forbidden text ['HEART']"


def test_error_code_evaluator() -> None:
    case = make_case({"action": "error", "error_code": "tool_unavailable"})
    output = ANSWER | {"action": "error", "error": {"code": "not_found", "message": "x"}}
    result = run(case, output)["error_code"]
    assert result.passed is False
    assert result.explanation == "expected error code 'tool_unavailable', got 'not_found'"


def test_get_path() -> None:
    data = {"a": {"b": [{"c": 1}]}}
    assert get_path(data, "a.b.0.c") == 1
    assert get_path(data, "a.b.1.c") is MISSING
    assert get_path(data, "a.x") is MISSING
    assert get_path({"a": None}, "a") is None


def test_booking_rule_rejects_missing_or_invented_dates() -> None:
    rule = get_rule("booking_requires_explicit_date")
    assert rule is not None
    case = make_case({"action": "answer"}, input={"message": "Book the heart check on 2026-11-20"})

    def check(date: Any) -> tuple[bool, str]:
        spans = tool_spans(("book_appointment", {"preferred_date": date}, False, "ok"))
        return rule(CaseExecution(case, ANSWER, spans))

    assert check("2026-11-20")[0] is True
    assert check(None) == (False, "book_appointment was executed without a preferred_date")
    assert check("2026-11-05") == (
        False,
        "book_appointment used date '2026-11-05', which the user never gave",
    )
    assert rule(CaseExecution(case, ANSWER, []))[0] is True  # no booking, nothing to check


def test_side_effect_and_refusal_rules() -> None:
    case = make_case({"action": "request_info"})
    side_effects = get_rule("request_info_has_no_side_effects")
    refusal = get_rule("refusal_runs_no_tools")
    assert side_effects is not None and refusal is not None
    asking = ANSWER | {"action": "request_info", "missing_fields": ["x"]}
    writes = tool_spans(("book_appointment", {}, False, "ok"))
    reads = tool_spans(("list_packages", {}, True, "ok"))
    assert side_effects(CaseExecution(case, asking, writes))[0] is False
    assert side_effects(CaseExecution(case, asking, reads))[0] is True
    refusing = ANSWER | {"action": "refuse"}
    assert refusal(CaseExecution(case, refusing, reads))[0] is False
    assert refusal(CaseExecution(case, refusing, []))[0] is True


def test_unregistered_rule_fails_loudly() -> None:
    case = make_case({"action": "answer", "business_rules": ["no_such_rule"]})
    result = run(case, ANSWER)["business_rule:no_such_rule"]
    assert result.passed is False and "not registered" in result.explanation


def test_rule_names_cannot_be_registered_twice() -> None:
    with pytest.raises(ValueError, match="already registered"):
        business_rule("refusal_runs_no_tools")(lambda execution: (True, ""))


# ------------------------------------------------------------------- judge

JUDGED = make_case(
    {"action": "answer"},
    judge={"rubric": "The answer is polite and clear.", "min_score": 4},
)


def judge_with(
    text: str | None = None, error: Exception | None = None, hang: bool = False
) -> LLMJudgeEvaluator:
    class Adapter:
        provider = "test"
        model = "judge-model"

        async def generate(self, request: ModelRequest) -> ModelResponse:
            if hang:
                await asyncio.sleep(10)
            if error:
                raise error
            return ModelResponse(text=text or "", provider="test", model="judge-model")

    return LLMJudgeEvaluator(Adapter(), timeout_s=0.05)


@pytest.mark.anyio
async def test_mock_judge_is_labelled_simulated() -> None:
    judge = LLMJudgeEvaluator(MockAdapter(MOCK_JUDGE_SCRIPT, model="mock-judge"))
    result = await judge.evaluate(CaseExecution(JUDGED, ANSWER, []))
    assert result.kind == "judge"
    assert result.score == 4.0 and result.max_score == 5.0
    assert result.passed is True
    assert result.details["simulated"] is True
    assert result.explanation.startswith("[simulated]")


@pytest.mark.anyio
async def test_judge_threshold_and_reasoning() -> None:
    result = await judge_with('{"score": 3, "reasoning": "Too terse."}').evaluate(
        CaseExecution(JUDGED, ANSWER, [])
    )
    assert result.passed is False
    assert result.score == 3.0
    assert "threshold 4" in result.explanation and "Too terse." in result.explanation
    assert result.details["simulated"] is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    "text",
    ['{"score": 9, "reasoning": "x"}', '{"score": "high"}', "Great answer! 5/5", ""],
)
async def test_invalid_judge_output_gives_no_verdict(text: str) -> None:
    result = await judge_with(text).evaluate(CaseExecution(JUDGED, ANSWER, []))
    assert result.passed is None
    assert result.score is None
    assert result.details["judge_error"] is True


@pytest.mark.anyio
async def test_judge_errors_and_timeouts_give_no_verdict() -> None:
    unavailable = await judge_with(error=ModelUnavailableError("down")).evaluate(
        CaseExecution(JUDGED, ANSWER, [])
    )
    assert unavailable.passed is None and "judge unavailable" in unavailable.explanation
    timed_out = await judge_with(hang=True).evaluate(CaseExecution(JUDGED, ANSWER, []))
    assert timed_out.passed is None and "timed out" in timed_out.explanation


def test_judge_only_applies_to_cases_with_a_rubric_and_output() -> None:
    judge = LLMJudgeEvaluator(MockAdapter(MockScript(responder=lambda r: "{}")))
    assert judge.applies_to(CaseExecution(JUDGED, ANSWER, []))
    assert not judge.applies_to(CaseExecution(JUDGED, None, []))
    assert not judge.applies_to(CaseExecution(make_case({"action": "answer"}), ANSWER, []))
