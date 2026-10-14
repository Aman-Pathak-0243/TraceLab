"""Regression tests for defects found in the Session 1 code review.

Each test reproduces a concrete failure scenario that the earlier code got wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from helpers import PROJECT_ROOT
from tracelab.adapters.base import ModelRequest, ModelResponse
from tracelab.api.app import create_app
from tracelab.cli import main
from tracelab.config import ConfigError, RunConfig, Settings
from tracelab.demo_app.agent import CareDeskAgent
from tracelab.demo_app.schemas import AgentRequest
from tracelab.evaluation.compare import compare_runs
from tracelab.evaluation.deterministic import strict_equal
from tracelab.service import run_evaluation
from tracelab.storage.store import Store
from tracelab.tracing.sanitize import REDACTED, sanitize
from tracelab.tracing.tracer import Tracer

pytestmark = pytest.mark.anyio


# 1. The regression gate must not pass when the head run is incomplete.


async def test_gate_fails_when_passing_cases_were_not_executed(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    full = await run_evaluation(baseline_config, settings, store)
    partial = await run_evaluation(
        baseline_config, settings, store, case_ids=["packages-list-001", "appt-lookup-001"]
    )
    comparison = compare_runs(store, full.run_id, partial.run_id)
    assert comparison.has_regressions is False
    assert len(comparison.lost_coverage) == 18
    assert comparison.gate_passed is False
    assert "18 case(s) that passed in base were not executed in head" in comparison.warnings


async def test_gate_fails_when_head_run_was_aborted(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    base = await run_evaluation(baseline_config, settings, store, case_ids=["appt-lookup-001"])
    head = await run_evaluation(baseline_config, settings, store, case_ids=["appt-lookup-001"])
    store.finish_run(head.run_id, status="aborted", summary=None, error="KeyboardInterrupt()")
    comparison = compare_runs(store, base.run_id, head.run_id)
    assert comparison.gate_passed is False
    assert f"head run {head.run_id} has status 'aborted'" in comparison.warnings


# 2 + 3. Persisted text and broader credential patterns are sanitized.


async def test_trace_input_and_output_are_sanitized_before_storage(settings: Settings) -> None:
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as http:
        body = (
            await http.post(
                "/api/caredesk/ask",
                json={
                    "message": "packages? my key is sk-abcdefghijklmnopqrstuvwx",
                    "context": {"api_key": "sk-zzzzzzzzzzzzzzzzzzzzzzzz"},
                },
            )
        ).json()
        trace = (await http.get(f"/api/traces/{body['trace_id']}")).json()
        page = (await http.get(f"/traces/{body['trace_id']}")).text
    assert trace["input"]["context"]["api_key"] == REDACTED
    assert trace["input"]["message"] == f"packages? my key is {REDACTED}"
    assert "sk-abcdefghijklmnopqrstuvwx" not in page and "sk-zzzz" not in page


def test_store_sanitizes_errors(tmp_path: Path) -> None:
    store = Store(tmp_path / "s.db")
    store.save_trace(
        trace_id="tr_x",
        source="api",
        input={},
        output=None,
        spans=[],
        error="ConnectError: http://admin:hunter2@ollama:11434 refused",
    )
    trace = store.get_trace("tr_x")
    assert trace is not None
    assert trace["error"] == f"ConnectError: http://{REDACTED}@ollama:11434 refused"


@pytest.mark.parametrize(
    "key",
    [
        "token",
        "auth_token",
        "id_token",
        "x-auth-token",
        "private_key",
        "pwd",
        "db_pwd",
        "passphrase",
    ],
)
def test_more_credential_keys_are_redacted(key: str) -> None:
    assert sanitize({key: "value"}) == {key: REDACTED}


@pytest.mark.parametrize("key", ["token_usage", "prompt_tokens", "max_tokens", "tokens"])
def test_token_metric_keys_stay_visible(key: str) -> None:
    assert sanitize({key: 5}) == {key: 5}


def test_value_patterns_cover_case_urls_and_reprs() -> None:
    class Secretive:
        def __repr__(self) -> str:
            return "Secretive(key='sk-abcdefghijklmnopqrstuv')"

    assert sanitize("auth: bearer abcdefghijklmnop") == f"auth: {REDACTED}"
    assert sanitize("http://user:pass@host/x") == f"http://{REDACTED}@host/x"
    assert sanitize(Secretive()) == f"Secretive(key='{REDACTED}')"
    assert REDACTED in sanitize(b"Bearer abcdefghijklmnop")


# 4. Field comparison is type-strict for booleans.


@pytest.mark.parametrize(
    ("actual", "expected", "equal"),
    [
        (1, True, False),
        (False, 0, False),
        (True, True, True),
        (3, 3.0, True),
        ({"a": [1, True]}, {"a": [1, True]}, True),
        ({"a": [1, 1]}, {"a": [1, True]}, False),
        ([1, 2], [1, 2, 3], False),
    ],
)
def test_strict_equal(actual: Any, expected: Any, equal: bool) -> None:
    assert strict_equal(actual, expected) is equal


# 5 + 7. Wrong-typed tool arguments are a controlled error; tools_used stays truthful.


class _FixedPlan:
    provider = "test"
    model = "fixed"

    def __init__(self, tool_calls: list[dict[str, Any]]) -> None:
        self.tool_calls = tool_calls

    async def generate(self, request: ModelRequest) -> ModelResponse:
        text = json.dumps({"intent": "x", "tool_calls": self.tool_calls})
        return ModelResponse(
            text=text if request.purpose == "plan" else "ok", provider="test", model="fixed"
        )


@pytest.mark.parametrize(
    "call",
    [
        {"name": "get_policy", "arguments": {"topic": 5}},
        {"name": "get_policy", "arguments": {"topic": ["cancellation"]}},
        {
            "name": "book_appointment",
            "arguments": {
                "member_id": "M-1001",
                "package_id": "PKG-HEART",
                "preferred_date": 20261115,
            },
        },
    ],
)
async def test_mistyped_arguments_become_invalid_tool_arguments(call: dict[str, Any]) -> None:
    tracer = Tracer()
    response = await CareDeskAgent(_FixedPlan([call])).handle(AgentRequest(message="x"), tracer)
    assert response.action == "error"
    assert response.error is not None and response.error.code == "invalid_tool_arguments"
    assert tracer.tool_calls() == []
    validation = next(s for s in tracer.spans if s.name == "validate.tool_call")
    assert validation.attributes["outcome"] == "mistyped_arguments"


async def test_error_after_a_successful_tool_reports_tools_used() -> None:
    plan = [{"name": "list_packages", "arguments": {}}, {"name": "nope", "arguments": {}}]
    tracer = Tracer()
    response = await CareDeskAgent(_FixedPlan(plan)).handle(AgentRequest(message="x"), tracer)
    assert response.error is not None and response.error.code == "unknown_tool"
    assert response.tools_used == ["list_packages"]
    assert "packages" in response.data


# 6. Missing files are configuration errors, not tracebacks.


async def test_missing_dataset_is_a_config_error(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    config = baseline_config.model_copy(update={"dataset": "datasets/does-not-exist.jsonl"})
    with pytest.raises(ConfigError, match="missing dataset file"):
        await run_evaluation(config, settings, store)
    assert store.list_runs() == []


async def test_api_missing_dataset_returns_400(tmp_path: Path, settings: Settings) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "gone.json").write_text(
        json.dumps({"name": "gone", "dataset": "datasets/does-not-exist.jsonl"}), encoding="utf-8"
    )
    app = create_app(settings.model_copy(update={"configs_dir": configs}))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as http:
        response = await http.post("/api/runs", json={"config": "gone"})
    assert response.status_code == 400
    assert "missing dataset file" in response.json()["detail"]


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRACELAB_PROJECT_ROOT", str(PROJECT_ROOT))
    monkeypatch.setenv("TRACELAB_DB_PATH", str(tmp_path / "cli.db"))
    return tmp_path


def test_cli_missing_files_and_bad_database(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = cli_env / "gone.json"
    config.write_text(
        json.dumps({"name": "gone", "dataset": "datasets/nope.jsonl"}), encoding="utf-8"
    )
    assert main(["run", "--config", str(config)]) == 2
    assert "missing dataset file" in capsys.readouterr().err

    assert main(["validate", "--schema-only", str(cli_env / "nope.jsonl")]) == 1
    assert "cannot read file" in capsys.readouterr().out

    future = cli_env / "future.db"
    import sqlite3

    conn = sqlite3.connect(future)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    assert main(["--db", str(future), "runs"]) == 2
    assert "newer than supported" in capsys.readouterr().err


# 8. Individual business rules can be disabled; typos are rejected.


async def test_disable_single_business_rule_and_reject_typos(
    baseline_config: RunConfig, settings: Settings, store: Store
) -> None:
    rule = "business_rule:booking_requires_explicit_date"
    config = baseline_config.model_copy(update={"disabled_evaluators": [rule]})
    report = await run_evaluation(config, settings, store, case_ids=["book-missing-date-001"])
    ids = [r.evaluator_id for r in report.outcomes[0].results]
    assert rule not in ids
    assert "business_rule:request_info_has_no_side_effects" in ids

    typo = baseline_config.model_copy(update={"disabled_evaluators": ["forbiden_tools"]})
    with pytest.raises(ConfigError, match="forbiden_tools"):
        await run_evaluation(typo, settings, store)


# 9. A run that executes nothing must not look like a success.


def test_all_skipped_run_exits_non_zero(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The ollama config skips model-fault cases before any network call is made.
    assert main(["run", "--config", "ollama", "--case", "model-timeout-001"]) == 1
    assert "No case was executed" in capsys.readouterr().out
