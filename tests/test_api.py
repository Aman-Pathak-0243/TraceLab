"""HTTP API and dashboard, exercised in-process through httpx's ASGI transport.

Also checks that the API, the database and the dashboard agree with each other.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from tracelab.api.app import create_app
from tracelab.config import Settings
from tracelab.storage.store import Store

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tracelab.test") as http:
        yield http


async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert {"baseline", "ollama"} <= set(body["configs"])


async def test_empty_state(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/runs")).json() == []
    page = await client.get("/")
    assert page.status_code == 200
    assert "No runs recorded yet" in page.text
    assert "No runs to compare yet" in (await client.get("/compare")).text


async def test_run_lifecycle_api_store_and_dashboard_agree(
    client: httpx.AsyncClient, settings: Settings
) -> None:
    created = await client.post("/api/runs", json={"config": "baseline"})
    assert created.status_code == 201
    run_id = created.json()["run_id"]
    summary = created.json()["summary"]
    assert (summary["passed"], summary["failed"]) == (20, 0)

    run = (await client.get(f"/api/runs/{run_id}")).json()
    stored = Store(settings.db_path).get_run(run_id)
    assert stored is not None
    assert run["summary"] == stored["summary"] == summary
    assert len(run["cases"]) == 20

    page = (await client.get(f"/runs/{run_id}")).text
    for case in run["cases"]:
        assert case["case_id"] in page
    assert "Simulated mock judge" in page

    case = (await client.get(f"/api/runs/{run_id}/cases/book-missing-date-001")).json()
    assert case["status"] == "passed"
    assert [s["name"] for s in case["trace"]["spans"]][:3] == [
        "agent.handle",
        "policy.screen_input",
        "plan",
    ]
    case_page = (await client.get(f"/runs/{run_id}/cases/book-missing-date-001")).text
    for evaluation in case["evaluations"]:
        assert evaluation["evaluator_id"] in case_page
    assert "validate.tool_call" in case_page


async def test_fault_injection_and_comparison(client: httpx.AsyncClient) -> None:
    base = (await client.post("/api/runs", json={"config": "baseline", "judge": False})).json()
    head = (
        await client.post(
            "/api/runs",
            json={"config": "baseline", "judge": False, "inject_fault": "presence_only_arg_check"},
        )
    ).json()
    assert head["summary"]["failed"] == 3

    comparison = (
        await client.get("/api/compare", params={"base": base["run_id"], "head": head["run_id"]})
    ).json()
    assert comparison["has_regressions"] is True
    assert {c["case_id"] for c in comparison["categories"]["newly_failed"]} == {
        "appt-lookup-missing-id-001",
        "book-missing-date-001",
        "book-missing-member-001",
    }

    page = await client.get("/compare", params={"base": base["run_id"], "head": head["run_id"]})
    assert "3 regression(s)" in page.text
    assert "config.agent.inject_fault" in page.text

    failing = await client.get(f"/runs/{head['run_id']}", params={"status": "failing"})
    assert "book-missing-date-001" in failing.text
    assert "packages-list-001" not in failing.text
    assert "Controlled fault injected" in failing.text


async def test_run_subset_and_bad_requests(client: httpx.AsyncClient) -> None:
    subset = await client.post(
        "/api/runs", json={"config": "baseline", "case_ids": ["appt-lookup-001"]}
    )
    assert subset.status_code == 201 and subset.json()["summary"]["total"] == 1

    assert (await client.post("/api/runs", json={"config": "nope"})).status_code == 404
    assert (
        await client.post("/api/runs", json={"config": "../configs/baseline"})
    ).status_code == 422
    assert (
        await client.post("/api/runs", json={"config": "baseline", "extra": 1})
    ).status_code == 422
    bad_fault = await client.post("/api/runs", json={"config": "baseline", "inject_fault": "nope"})
    assert bad_fault.status_code == 400 and "Unknown fault" in bad_fault.json()["detail"]
    bad_case = await client.post("/api/runs", json={"config": "baseline", "case_ids": ["nope"]})
    assert bad_case.status_code == 400 and "unknown case id" in bad_case.json()["detail"]


async def test_config_dataset_must_stay_inside_datasets_dir(
    tmp_path: Path, settings: Settings
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "escape.json").write_text(
        json.dumps({"name": "escape", "dataset": "../outside.jsonl"}), encoding="utf-8"
    )
    app = create_app(settings.model_copy(update={"configs_dir": configs}))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as http:
        response = await http.post("/api/runs", json={"config": "escape"})
    assert response.status_code == 400
    assert "inside the datasets directory" in response.json()["detail"]


async def test_invalid_dataset_is_rejected_before_running(
    tmp_path: Path, settings: Settings
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "broken.json").write_text(
        json.dumps({"name": "broken", "dataset": "datasets/examples/invalid-records.jsonl"}),
        encoding="utf-8",
    )
    app = create_app(settings.model_copy(update={"configs_dir": configs}))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as http:
        response = await http.post("/api/runs", json={"config": "broken"})
        runs = (await http.get("/api/runs")).json()
    assert response.status_code == 422
    assert any("malformed JSON" in issue for issue in response.json()["detail"]["issues"])
    assert runs == []  # no half-created run


async def test_ask_endpoint_traces_every_request(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/caredesk/ask",
        json={"message": "Please book the Heart Health Check", "context": {"member_id": "M-1001"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["response"]["action"] == "request_info"
    assert body["response"]["missing_fields"] == ["preferred_date"]

    trace = (await client.get(f"/api/traces/{body['trace_id']}")).json()
    assert trace["source"] == "api"
    assert trace["output"] == body["response"]
    assert (await client.get(f"/traces/{body['trace_id']}")).status_code == 200
    assert body["trace_id"] in (await client.get("/")).text


async def test_ask_validation(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/caredesk/ask", json={"message": ""})).status_code == 422
    assert (
        await client.post("/api/caredesk/ask", json={"message": "hi", "config": "nope"})
    ).status_code == 404


async def test_user_content_is_escaped_in_the_dashboard(client: httpx.AsyncClient) -> None:
    body = (
        await client.post(
            "/api/caredesk/ask", json={"message": "<script>alert(1)</script> packages?"}
        )
    ).json()
    page = (await client.get(f"/traces/{body['trace_id']}")).text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


async def test_not_found_responses(client: httpx.AsyncClient) -> None:
    for url in ["/api/runs/nope", "/api/runs/nope/cases/x", "/api/traces/nope"]:
        assert (await client.get(url)).status_code == 404, url
    assert (await client.get("/api/compare", params={"base": "a", "head": "b"})).status_code == 404
    for url in ["/runs/nope", "/runs/nope/cases/x", "/traces/nope", "/compare?base=a&head=b"]:
        page = await client.get(url)
        assert page.status_code == 404 and "Not found" in page.text, url
