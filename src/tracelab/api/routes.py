"""JSON API.

Read-only endpoints are plain `def`: FastAPI runs them in a threadpool, so the
blocking SQLite calls never stall the event loop. Endpoints that await model
calls are `async def` and push their own blocking work to threads.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from tracelab import __version__
from tracelab.api.deps import get_settings, get_store
from tracelab.config import ConfigError, RunConfig, Settings, load_run_config
from tracelab.datasets.loader import DatasetValidationError
from tracelab.demo_app.schemas import AgentRequest, AgentResponse
from tracelab.demo_app.target import CareDeskTarget
from tracelab.evaluation.compare import RunNotFoundError, compare_runs
from tracelab.service import resolve_dataset_path, run_evaluation
from tracelab.storage.store import Store
from tracelab.tracing.tracer import Tracer

router = APIRouter(prefix="/api", tags=["api"])

StoreDep = Annotated[Store, Depends(get_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ConfigName = Field(default="baseline", pattern=r"^[A-Za-z0-9_-]{1,64}$")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str = ConfigName
    case_ids: list[str] | None = None
    inject_fault: str | None = None
    judge: bool | None = Field(default=None, description="override judge.enabled")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    context: dict[str, str] = Field(default_factory=dict)
    config: str = ConfigName


class AskResponse(BaseModel):
    trace_id: str
    response: AgentResponse


def available_configs(settings: Settings) -> list[str]:
    return sorted(p.stem for p in settings.configs_dir.glob("*.json"))


def load_named_config(settings: Settings, name: str) -> RunConfig:
    """Load configs/<name>.json. Only names are accepted, never client paths."""
    path = settings.configs_dir / f"{name}.json"
    if not path.is_file():
        raise HTTPException(
            404, f"unknown config {name!r}; available: {available_configs(settings)}"
        )
    try:
        config = load_run_config(path)
    except ConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    dataset = resolve_dataset_path(config, settings).resolve()
    if not dataset.is_relative_to(Path(settings.datasets_dir).resolve()):
        raise HTTPException(400, "config dataset must live inside the datasets directory")
    return config


@router.get("/health")
def health(settings: SettingsDep) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "configs": available_configs(settings),
        "database": settings.db_path.name,
    }


@router.get("/runs")
def list_runs(
    store: StoreDep, limit: int = Query(default=50, ge=1, le=500)
) -> list[dict[str, Any]]:
    return store.list_runs(limit=limit)


@router.post("/runs", status_code=201)
async def create_run(body: RunRequest, settings: SettingsDep, store: StoreDep) -> dict[str, Any]:
    config = await asyncio.to_thread(load_named_config, settings, body.config)
    data = config.model_dump()
    if body.inject_fault is not None:
        data["agent"]["inject_fault"] = body.inject_fault
    if body.judge is not None:
        data["judge"]["enabled"] = body.judge
    config = RunConfig.model_validate(data)
    try:
        report = await run_evaluation(config, settings, store, case_ids=body.case_ids)
    except DatasetValidationError as exc:
        raise HTTPException(
            422, {"message": "dataset is invalid", "issues": [str(i) for i in exc.issues]}
        ) from exc
    except (ConfigError, ValueError) as exc:
        # missing dataset file, unknown fault/provider/prompt version/case id/evaluator
        raise HTTPException(400, str(exc)) from exc
    return {
        "run_id": report.run_id,
        "summary": report.summary,
        "links": {"api": f"/api/runs/{report.run_id}", "dashboard": f"/runs/{report.run_id}"},
    }


@router.get("/runs/{run_id}")
def get_run(run_id: str, store: StoreDep) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id!r} not found")
    return run | {"cases": store.list_case_results(run_id)}


@router.get("/runs/{run_id}/cases/{case_id}")
def get_case(run_id: str, case_id: str, store: StoreDep) -> dict[str, Any]:
    case = store.get_case_result(run_id, case_id)
    if case is None:
        raise HTTPException(404, f"case {case_id!r} not found in run {run_id!r}")
    trace = store.get_trace(case["trace_id"]) if case["trace_id"] else None
    return case | {"trace": trace}


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str, store: StoreDep) -> dict[str, Any]:
    trace = store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(404, f"trace {trace_id!r} not found")
    return trace


@router.get("/compare")
def compare(base: str, head: str, store: StoreDep) -> dict[str, Any]:
    try:
        return compare_runs(store, base, head).to_dict()
    except RunNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/caredesk/ask")
async def ask(body: AskRequest, settings: SettingsDep, store: StoreDep) -> AskResponse:
    """Call the instrumented demo app directly. Every request gets a persisted trace.

    Controlled application outcomes (including action="error") return HTTP 200:
    the request was handled and the structured response explains the outcome.
    """
    config = await asyncio.to_thread(load_named_config, settings, body.config)
    try:
        agent = CareDeskTarget(config.adapter, config.agent, settings).build_agent()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    request = AgentRequest(message=body.message, context=body.context)
    tracer = Tracer()
    response: AgentResponse | None = None
    error: str | None = None
    try:
        response = await agent.handle(request, tracer)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise  # FastAPI turns this into a 500; the partial trace is still saved below
    finally:
        await asyncio.to_thread(
            store.save_trace,
            trace_id=tracer.trace_id,
            source="api",
            input=request.model_dump(),
            output=response.model_dump(mode="json") if response else None,
            spans=[s.to_dict() for s in tracer.spans],
            error=error,
        )
    return AskResponse(trace_id=tracer.trace_id, response=response)
