"""Server-rendered dashboard. Every page reads the same SQLite store as the API.

Jinja2 autoescaping is on for .html templates, so user messages and model
output stored in traces are rendered as text, never as HTML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from tracelab.api.routes import StoreDep
from tracelab.evaluation.compare import RunNotFoundError, compare_runs
from tracelab.tracing.render import span_depths

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["pretty"] = lambda value: json.dumps(
    value, indent=2, sort_keys=True, default=str
)


def _page(request: Request, name: str, status_code: int = 200, **context: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, name, context, status_code=status_code)


def _not_found(request: Request, message: str) -> HTMLResponse:
    return _page(request, "error.html", status_code=404, message=message)


def _trace_rows(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    depths = span_depths(spans)
    return [
        span | {"depth": depths[span["span_id"]]} for span in sorted(spans, key=lambda s: s["seq"])
    ]


@router.get("/", response_class=HTMLResponse)
def index(request: Request, store: StoreDep) -> HTMLResponse:
    runs = store.list_runs(limit=100)
    traces = store.list_traces(source="api", limit=10) + store.list_traces(source="cli", limit=10)
    traces.sort(key=lambda t: t["created_at"], reverse=True)
    return _page(request, "index.html", runs=runs, traces=traces[:10])


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(
    request: Request, run_id: str, store: StoreDep, status: str | None = None
) -> HTMLResponse:
    run = store.get_run(run_id)
    if run is None:
        return _not_found(request, f"Run {run_id} was not found.")
    cases = store.list_case_results(run_id)
    if status == "failing":
        cases = [c for c in cases if c["status"] in ("failed", "error")]
    elif status:
        cases = [c for c in cases if c["status"] == status]
    return _page(
        request,
        "run.html",
        run=run,
        cases=cases,
        status_filter=status,
        dataset_name=Path(run["dataset_path"]).name,
    )


@router.get("/runs/{run_id}/cases/{case_id}", response_class=HTMLResponse)
def case_detail(request: Request, run_id: str, case_id: str, store: StoreDep) -> HTMLResponse:
    run = store.get_run(run_id)
    case = store.get_case_result(run_id, case_id)
    if run is None or case is None:
        return _not_found(request, f"Case {case_id} was not found in run {run_id}.")
    trace = store.get_trace(case["trace_id"]) if case["trace_id"] else None
    return _page(
        request,
        "case.html",
        run=run,
        case=case,
        trace=trace,
        spans=_trace_rows(trace["spans"]) if trace else [],
    )


@router.get("/traces/{trace_id}", response_class=HTMLResponse)
def trace_detail(request: Request, trace_id: str, store: StoreDep) -> HTMLResponse:
    trace = store.get_trace(trace_id)
    if trace is None:
        return _not_found(request, f"Trace {trace_id} was not found.")
    return _page(request, "trace.html", trace=trace, spans=_trace_rows(trace["spans"]))


@router.get("/compare", response_class=HTMLResponse)
def compare_page(
    request: Request, store: StoreDep, base: str | None = None, head: str | None = None
) -> HTMLResponse:
    runs = store.list_runs(limit=100)
    if not base or not head:
        return _page(request, "compare.html", runs=runs, comparison=None, base=base, head=head)
    try:
        comparison = compare_runs(store, base, head)
    except RunNotFoundError as exc:
        return _not_found(request, str(exc))
    return _page(
        request,
        "compare.html",
        runs=runs,
        comparison=comparison,
        groups=comparison.by_category(),
        base=base,
        head=head,
    )
