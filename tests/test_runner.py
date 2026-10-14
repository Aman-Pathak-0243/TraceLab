"""Suite execution, persistence, trace completeness and harness error handling."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from tracelab.adapters import factory
from tracelab.adapters.ollama import OllamaAdapter
from tracelab.config import RunConfig, Settings
from tracelab.datasets.loader import Dataset
from tracelab.datasets.schema import EvalCase
from tracelab.demo_app.schemas import AgentResponse
from tracelab.evaluation.runner import execute_case, run_suite, select_cases
from tracelab.service import build_judge, build_target, load_dataset_for, run_evaluation
from tracelab.storage.store import Store
from tracelab.tracing.tracer import Tracer

pytestmark = pytest.mark.anyio


async def test_baseline_suite_passes_and_is_persisted(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    report = await run_evaluation(baseline_config, settings, store)
    summary = report.summary

    assert (summary["total"], summary["passed"], summary["failed"], summary["error"]) == (
        20,
        20,
        0,
        0,
    )
    assert summary["failed_case_ids"] == []
    assert summary["judge"]["cases_scored"] == 2 and summary["judge"]["simulated"] is True
    assert summary["token_usage"]["model_calls_with_usage"] == 0
    assert summary["token_usage"]["prompt_tokens"] is None  # never invented

    run = store.get_run(report.run_id)
    assert run is not None
    assert run["status"] == "completed"
    assert run["summary"] == summary
    assert run["config"]["agent"]["inject_fault"] is None
    assert run["target"] == {
        "target": "caredesk",
        "provider": "mock",
        "model": "caredesk-mock-v1",
        "prompt_version": "v1",
        "inject_fault": None,
    }
    cases = store.list_case_results(report.run_id)
    assert [c["case_id"] for c in cases] == [o.case.case_id for o in report.outcomes]
    for stored, outcome in zip(cases, report.outcomes, strict=True):
        assert stored["status"] == outcome.status
        assert stored["output"] == outcome.output
        assert len(stored["evaluations"]) == len(outcome.results)


async def test_every_trace_is_complete(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    report = await run_evaluation(baseline_config, settings, store)
    for case in store.list_case_results(report.run_id):
        trace = store.get_trace(case["trace_id"])
        assert trace is not None, case["case_id"]
        spans = trace["spans"]
        ids = {s["span_id"] for s in spans}
        roots = [s for s in spans if s["parent_span_id"] is None]
        assert [r["name"] for r in roots] == ["agent.handle"]
        assert [s["seq"] for s in spans] == list(range(len(spans)))
        for span in spans:
            assert span["status"] in ("ok", "error"), (case["case_id"], span["name"])
            assert span["ended_at"] is not None and span["duration_ms"] >= 0
            assert span["parent_span_id"] is None or span["parent_span_id"] in ids
        assert roots[0]["attributes"]["action"] == case["output"]["action"]
        assert case["latency_ms"] == roots[0]["duration_ms"]


async def test_results_do_not_depend_on_concurrency(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    serial = await run_evaluation(
        baseline_config.model_copy(update={"concurrency": 1}), settings, store
    )
    parallel = await run_evaluation(
        baseline_config.model_copy(update={"concurrency": 8}), settings, store
    )
    assert [(o.case.case_id, o.status, o.output) for o in serial.outcomes] == [
        (o.case.case_id, o.status, o.output) for o in parallel.outcomes
    ]


class _CrashingTarget:
    """A target that fails in the ways the harness must survive."""

    name = "crashing"
    output_model = AgentResponse

    def describe(self) -> dict[str, Any]:
        return {"target": self.name}

    def validate_case(self, case: EvalCase) -> list[str]:
        return []

    def skip_reason(self, case: EvalCase) -> str | None:
        return "not supported here" if case.case_id == "out-of-scope-001" else None

    async def execute(self, case: EvalCase, tracer: Tracer) -> dict[str, Any]:
        with tracer.span("agent.handle", "agent"):
            if case.case_id == "packages-list-001":
                raise RuntimeError("unexpected crash")
            if case.case_id == "appt-lookup-001":
                await asyncio.sleep(5)
        return {"action": "answer", "answer": "ok"}


async def test_harness_errors_are_recorded_not_hidden(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    target = build_target(baseline_config, settings)
    dataset = load_dataset_for(baseline_config, settings, target)
    config = baseline_config.model_copy(update={"case_timeout_s": 0.2})
    report = await run_suite(
        config=config,
        target=_CrashingTarget(),
        dataset=dataset,
        store=store,
        case_ids=[
            "packages-list-001",
            "appt-lookup-001",
            "out-of-scope-001",
            "policy-preparation-001",
        ],
    )
    by_id = {o.case.case_id: o for o in report.outcomes}

    crashed = by_id["packages-list-001"]
    assert crashed.status == "error"
    assert crashed.harness_error == "RuntimeError: unexpected crash"
    assert crashed.spans[0]["status"] == "error"  # partial trace kept

    slow = by_id["appt-lookup-001"]
    assert slow.status == "error"
    assert slow.harness_error == "case exceeded case_timeout_s=0.2"
    assert slow.spans[0]["status"] == "cancelled"

    assert by_id["out-of-scope-001"].status == "skipped"
    assert by_id["out-of-scope-001"].skip_reason == "not supported here"
    assert by_id["policy-preparation-001"].status == "failed"  # wrong answer, but evaluated

    assert report.summary["error"] == 2 and report.summary["skipped"] == 1
    stored = store.get_case_result(report.run_id, "packages-list-001")
    assert stored is not None and stored["harness_error"] == "RuntimeError: unexpected crash"
    trace = store.get_trace(stored["trace_id"])
    assert trace is not None and trace["error"] == "RuntimeError: unexpected crash"


async def test_a_case_with_no_deterministic_checks_cannot_pass(
    baseline_config: RunConfig, settings: Settings
) -> None:
    target = build_target(baseline_config, settings)
    case = load_dataset_for(baseline_config, settings, target).cases[0]
    outcome = await execute_case(
        case,
        0,
        target,
        case_timeout_s=5,
        disabled_evaluators=[
            "output_schema",
            "expected_action",
            "required_tools",
            "forbidden_tools",
            "answer_contains",
            "field_values",
        ],
    )
    assert outcome.status == "error"
    assert outcome.harness_error == "no deterministic evaluator applied to this case"


async def test_judge_result_never_changes_case_status(
    baseline_config: RunConfig, settings: Settings
) -> None:
    target = build_target(baseline_config, settings)
    case = next(
        c for c in load_dataset_for(baseline_config, settings, target).cases if c.judge is not None
    )
    strict = case.model_copy(update={"judge": case.judge.model_copy(update={"min_score": 5})})
    judge = build_judge(baseline_config, settings)
    outcome = await execute_case(strict, 0, target, case_timeout_s=5, judge=judge)
    assert outcome.judge_results[0].passed is False  # mock judge gives 4 < 5
    assert outcome.status == "passed"


def test_select_cases_rejects_unknown_ids(baseline_config: RunConfig, settings: Settings) -> None:
    target = build_target(baseline_config, settings)
    dataset: Dataset = load_dataset_for(baseline_config, settings, target)
    assert [c.case_id for c in select_cases(dataset, ["appt-lookup-001"])] == ["appt-lookup-001"]
    with pytest.raises(ValueError, match="unknown case id"):
        select_cases(dataset, ["nope"])


async def test_unreachable_ollama_fails_cleanly_and_skips_mock_only_cases(
    baseline_config: RunConfig, settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What happens when Ollama is not running: no crash, clear errors, honest skips."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    def build_unreachable(config, settings, script, fault):  # type: ignore[no-untyped-def]
        return OllamaAdapter("llama3", transport=httpx.MockTransport(refuse))

    monkeypatch.setitem(factory._BUILDERS, "ollama", build_unreachable)
    config = baseline_config.model_copy(
        update={
            "adapter": baseline_config.adapter.model_copy(
                update={"provider": "ollama", "model": None}
            ),
            "judge": baseline_config.judge.model_copy(update={"enabled": False}),
        }
    )
    report = await run_evaluation(config, settings, store)
    by_status: dict[str, list[str]] = {}
    for outcome in report.outcomes:
        by_status.setdefault(outcome.status, []).append(outcome.case.case_id)

    model_fault_cases = [
        o.case.case_id for o in report.outcomes if o.case.simulate and o.case.simulate.model_fault
    ]
    assert by_status["skipped"] == model_fault_cases and len(model_fault_cases) == 7
    assert "error" not in by_status  # no harness crashes
    refused = next(o for o in report.outcomes if o.case.case_id == "packages-list-001")
    assert refused.output is not None and refused.output["error"]["code"] == "model_unavailable"
    # Cases that never reach the model still pass (deterministic guard).
    assert "medical-refusal-001" in by_status["passed"]
