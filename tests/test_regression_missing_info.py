"""Regression tests for the "booking without a user-supplied date" incident.

Incident: a user asked "Please book the Heart Health Check for me." without a
date. The model emitted `"preferred_date": null`. Tool-argument validation only
checked that each required key existed, so `book_appointment` ran and the
scheduling backend silently booked the next free slot (2026-11-05), a date the
user never chose.

Root cause: `validate.tool_call` treated null/placeholder values as present.
Fix: `find_missing_arguments` treats None, empty strings and placeholders such
as "unknown" or "YYYY-MM-DD" as missing, so the agent asks the user instead.

The defective check is kept as a controlled fault so the failure can be
reproduced without editing tracked code:

    TRACELAB_INJECT_FAULT=presence_only_arg_check pytest tests/test_regression_missing_info.py
        -> the tests in this file FAIL (the bug is back)
    pytest tests/test_regression_missing_info.py
        -> they pass

`test_suite_detects_the_injected_fault` always runs with the fault on purpose
and checks that the evaluation suite itself catches it.
"""

from __future__ import annotations

import json

import pytest

from helpers import with_fault
from tracelab.adapters.base import ModelRequest, ModelResponse
from tracelab.adapters.mock import MockAdapter
from tracelab.config import RunConfig, Settings
from tracelab.demo_app.agent import CareDeskAgent
from tracelab.demo_app.mock_script import CAREDESK_MOCK_SCRIPT
from tracelab.demo_app.schemas import AgentRequest
from tracelab.demo_app.target import CareDeskTarget
from tracelab.evaluation.runner import execute_case
from tracelab.service import load_dataset_for
from tracelab.tracing.tracer import Tracer

pytestmark = pytest.mark.anyio

MEMBER = {"member_id": "M-1001"}


@pytest.mark.parametrize(
    ("message", "context", "missing"),
    [
        ("Please book the Heart Health Check for me.", MEMBER, ["preferred_date"]),
        ("When is my appointment?", MEMBER, ["appointment_id"]),
        ("Book the Basic Wellness Screening on 2026-11-20.", {}, ["member_id"]),
    ],
)
async def test_agent_asks_for_missing_information_instead_of_proceeding(
    message: str, context: dict[str, str], missing: list[str], injected_fault: str | None
) -> None:
    agent = CareDeskAgent(MockAdapter(CAREDESK_MOCK_SCRIPT), inject_fault=injected_fault)
    tracer = Tracer()
    response = await agent.handle(AgentRequest(message=message, context=context), tracer)

    assert response.action == "request_info"
    assert response.missing_fields == missing
    assert tracer.tool_calls() == [], "no tool may run while information is missing"


class _PlanWithDate:
    """A model that fills preferred_date with a placeholder instead of null."""

    provider = "test"
    model = "placeholder-model"

    def __init__(self, value: object) -> None:
        self.value = value

    async def generate(self, request: ModelRequest) -> ModelResponse:
        plan = {
            "intent": "book_appointment",
            "tool_calls": [
                {
                    "name": "book_appointment",
                    "arguments": {
                        "member_id": "M-1001",
                        "package_id": "PKG-HEART",
                        "preferred_date": self.value,
                    },
                }
            ],
        }
        return ModelResponse(text=json.dumps(plan), provider=self.provider, model=self.model)


@pytest.mark.parametrize("placeholder", [None, "", "   ", "null", "unknown", "YYYY-MM-DD", "TBD"])
async def test_placeholder_dates_are_treated_as_missing(
    placeholder: object, injected_fault: str | None
) -> None:
    agent = CareDeskAgent(_PlanWithDate(placeholder), inject_fault=injected_fault)
    tracer = Tracer()
    response = await agent.handle(
        AgentRequest(message="Book the Heart Health Check", context=MEMBER), tracer
    )
    assert response.action == "request_info"
    assert response.missing_fields == ["preferred_date"]
    assert "book_appointment" not in [c["tool"] for c in tracer.tool_calls()]


async def test_dataset_case_passes(
    baseline_config: RunConfig, settings: Settings, injected_fault: str | None
) -> None:
    config = with_fault(baseline_config, injected_fault)
    target = CareDeskTarget(config.adapter, config.agent, settings)
    case = next(
        c
        for c in load_dataset_for(config, settings, target).cases
        if c.case_id == "book-missing-date-001"
    )
    outcome = await execute_case(case, 0, target, case_timeout_s=5)
    assert outcome.status == "passed", [
        (r.evaluator_id, r.explanation) for r in outcome.results if r.passed is False
    ]


async def test_suite_detects_the_injected_fault(
    baseline_config: RunConfig, settings: Settings
) -> None:
    """Always runs WITH the fault: proves the evaluation catches this bug class."""
    config = with_fault(baseline_config, "presence_only_arg_check")
    target = CareDeskTarget(config.adapter, config.agent, settings)
    cases = {c.case_id: c for c in load_dataset_for(config, settings, target).cases}

    outcome = await execute_case(cases["book-missing-date-001"], 0, target, case_timeout_s=5)
    assert outcome.status == "failed"
    assert set(outcome.failed_evaluators) == {
        "expected_action",
        "missing_fields",
        "forbidden_tools",
        "answer_contains",
        "business_rule:booking_requires_explicit_date",
    }
    assert outcome.output is not None
    assert outcome.output["data"]["booking"]["date"] == "2026-11-05"  # the silently chosen slot

    # The trace pinpoints the faulty step: validation says "valid" for a null date.
    validation = next(s for s in outcome.spans if s["name"] == "validate.tool_call")
    assert validation["attributes"]["outcome"] == "valid"
    assert validation["attributes"]["arguments"]["preferred_date"] is None
    assert any(s["name"] == "tool.book_appointment" for s in outcome.spans)

    # Unrelated behaviour is unaffected by this fault.
    unaffected = await execute_case(cases["book-success-001"], 1, target, case_timeout_s=5)
    assert unaffected.status == "passed"
