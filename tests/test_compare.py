"""Run comparison: regression categories, config attribution and honest metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import DATASET_PATH, with_fault
from tracelab.config import RunConfig, Settings
from tracelab.evaluation.compare import RunNotFoundError, compare_runs
from tracelab.service import run_evaluation
from tracelab.storage.store import Store

pytestmark = pytest.mark.anyio

FAULT = "presence_only_arg_check"
BROKEN = {"appt-lookup-missing-id-001", "book-missing-date-001", "book-missing-member-001"}


def ids(changes: list) -> set[str]:
    return {c.case_id for c in changes}


async def test_baseline_to_faulty_to_fixed(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    baseline = await run_evaluation(baseline_config, settings, store)
    faulty = await run_evaluation(with_fault(baseline_config, FAULT), settings, store)
    fixed = await run_evaluation(baseline_config, settings, store)

    regression = compare_runs(store, baseline.run_id, faulty.run_id)
    groups = regression.by_category()
    assert regression.has_regressions
    assert ids(groups["newly_failed"]) == BROKEN
    assert len(groups["passed_both"]) == 17
    assert not groups["newly_passed"] and not groups["still_failing"]
    assert regression.dataset_changed is False
    assert {
        "key": "config.agent.inject_fault",
        "base": None,
        "head": FAULT,
    } in regression.config_changes
    assert {"key": "target.inject_fault", "base": None, "head": FAULT} in regression.config_changes

    critical = next(c for c in groups["newly_failed"] if c.case_id == "book-missing-date-001")
    assert critical.severity == "critical"
    assert {
        "evaluator_id": "expected_action",
        "base": True,
        "head": False,
    } in critical.evaluator_changes

    resolution = compare_runs(store, faulty.run_id, fixed.run_id)
    assert not resolution.has_regressions
    assert ids(resolution.by_category()["newly_passed"]) == BROKEN

    metrics = resolution.metrics
    assert metrics["status_counts"]["base"]["failed"] == 3
    assert metrics["status_counts"]["head"]["failed"] == 0
    assert metrics["latency_ms_mean_shared_cases"]["cases"] == 20
    assert metrics["judge_mean_score_shared_cases"] == {
        "cases": 2,
        "base": 4.0,
        "head": 4.0,
        "delta": 0.0,
    }
    # The mock reports no tokens, so the comparison must say "unavailable", not 0.
    assert metrics["token_usage"] == {"base": None, "head": None, "available": False}


async def test_added_and_removed_cases(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    first = await run_evaluation(
        baseline_config, settings, store, case_ids=["packages-list-001", "appt-lookup-001"]
    )
    second = await run_evaluation(
        baseline_config, settings, store, case_ids=["appt-lookup-001", "policy-preparation-001"]
    )
    groups = compare_runs(store, first.run_id, second.run_id).by_category()
    assert ids(groups["only_in_base"]) == {"packages-list-001"}
    assert ids(groups["only_in_head"]) == {"policy-preparation-001"}
    assert ids(groups["passed_both"]) == {"appt-lookup-001"}


async def test_dataset_change_is_flagged(
    baseline_config: RunConfig, settings: Settings, store: Store, tmp_path: Path
) -> None:
    lines = DATASET_PATH.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["description"] = "Changed wording; same behaviour."
    copy = tmp_path / "caredesk-v1-edited.jsonl"
    copy.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", encoding="utf-8")

    original = await run_evaluation(baseline_config, settings, store)
    edited = await run_evaluation(
        baseline_config.model_copy(update={"dataset": str(copy)}), settings, store
    )
    comparison = compare_runs(store, original.run_id, edited.run_id)
    assert comparison.dataset_changed is True
    assert {
        "key": "config.dataset",
        "base": "datasets/caredesk-v1.jsonl",
        "head": str(copy),
    } in comparison.config_changes


async def test_to_dict_is_json_serialisable(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    run = await run_evaluation(baseline_config, settings, store, case_ids=["packages-list-001"])
    data = compare_runs(store, run.run_id, run.run_id).to_dict()
    assert json.loads(json.dumps(data))["has_regressions"] is False
    assert data["categories"]["passed_both"][0]["judge_delta"] == 0.0


def test_unknown_run_raises(store: Store) -> None:
    with pytest.raises(RunNotFoundError):
        compare_runs(store, "run_missing", "run_missing_too")
