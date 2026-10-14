"""Evaluation runner: executes a dataset against a target and persists results.

Cases run concurrently (bounded by `RunConfig.concurrency`), which matters when
each case waits on a slow local model. Each case is persisted as soon as it
finishes, so a run that dies half-way still leaves inspectable traces behind.
"""

from __future__ import annotations

import asyncio
import inspect
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from tracelab.config import RunConfig
from tracelab.datasets.loader import Dataset
from tracelab.datasets.schema import EvalCase
from tracelab.evaluation.deterministic import evaluators_for
from tracelab.evaluation.judge import LLMJudgeEvaluator
from tracelab.evaluation.types import CaseExecution, CaseOutcome, EvalResult, Target
from tracelab.storage.store import Store
from tracelab.tracing.tracer import Tracer, new_id


@dataclass(frozen=True)
class RunReport:
    run_id: str
    summary: dict[str, Any]
    outcomes: list[CaseOutcome]


async def execute_case(
    case: EvalCase,
    position: int,
    target: Target,
    *,
    case_timeout_s: float,
    disabled_evaluators: Sequence[str] = (),
    judge: LLMJudgeEvaluator | None = None,
) -> CaseOutcome:
    skip = target.skip_reason(case)
    if skip is not None:
        return CaseOutcome(case, position, "skipped", None, None, [], [], None, skip_reason=skip)

    tracer = Tracer()
    output: dict[str, Any] | None = None
    harness_error: str | None = None
    started = time.perf_counter()
    try:
        async with asyncio.timeout(case_timeout_s) as deadline:
            output = await target.execute(case, tracer)
    except TimeoutError as exc:
        if deadline.expired():
            harness_error = f"case exceeded case_timeout_s={case_timeout_s}"
        else:
            harness_error = f"TimeoutError: {exc}"
    except Exception as exc:
        # Harness boundary: one crashing case must not abort the suite, but the
        # failure is recorded as status "error" with the exception text and the
        # partial trace - it is never hidden or counted as a pass.
        harness_error = f"{type(exc).__name__}: {exc}"
    wall_ms = round((time.perf_counter() - started) * 1000, 3)

    spans = [span.to_dict() for span in tracer.spans]
    root = tracer.root()
    latency_ms = root.duration_ms if root is not None and root.duration_ms is not None else wall_ms
    execution = CaseExecution(case, output, spans, harness_error)

    results: list[EvalResult] = []
    if harness_error is None:
        evaluators: list[Any] = list(evaluators_for(case, target.output_model, disabled_evaluators))
        if judge is not None and judge.applies_to(execution):
            evaluators.append(judge)
        for evaluator in evaluators:
            try:
                result = evaluator.evaluate(execution)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                harness_error = (
                    f"evaluator {evaluator.evaluator_id} raised {type(exc).__name__}: {exc}"
                )
                break
            results.append(result)
        if harness_error is None and not any(r.kind == "deterministic" for r in results):
            # Never report a vacuous pass for a case that nothing actually checked.
            harness_error = "no deterministic evaluator applied to this case"
    if harness_error is not None:
        results.append(EvalResult("execution", "deterministic", False, harness_error))

    if harness_error is not None:
        status = "error"
    elif all(r.passed for r in results if r.kind == "deterministic"):
        status = "passed"
    else:
        status = "failed"
    return CaseOutcome(
        case=case,
        position=position,
        status=status,
        trace_id=tracer.trace_id,
        output=output,
        spans=spans,
        results=results,
        latency_ms=latency_ms,
        harness_error=harness_error,
    )


def select_cases(dataset: Dataset, case_ids: Sequence[str] | None) -> list[EvalCase]:
    if not case_ids:
        return list(dataset.cases)
    known = {case.case_id for case in dataset.cases}
    unknown = sorted(set(case_ids) - known)
    if unknown:
        raise ValueError(f"unknown case id(s): {unknown}")
    wanted = set(case_ids)
    return [case for case in dataset.cases if case.case_id in wanted]


async def run_suite(
    *,
    config: RunConfig,
    target: Target,
    dataset: Dataset,
    store: Store,
    judge: LLMJudgeEvaluator | None = None,
    case_ids: Sequence[str] | None = None,
    on_case_done: Callable[[CaseOutcome], None] | None = None,
) -> RunReport:
    cases = select_cases(dataset, case_ids)
    run_id = new_id("run")
    await asyncio.to_thread(
        store.create_run,
        run_id=run_id,
        config_name=config.name,
        config=config.model_dump(mode="json"),
        target=target.describe(),
        dataset_path=config.dataset,  # as configured (relative); sha256 pins the content
        dataset_sha256=dataset.sha256,
        dataset_schema_versions=dataset.schema_versions,
        case_count=len(cases),
    )

    semaphore = asyncio.Semaphore(config.concurrency)
    write_lock = asyncio.Lock()  # SQLite has one writer; serialise our writes

    async def run_one(position: int, case: EvalCase) -> CaseOutcome:
        async with semaphore:
            outcome = await execute_case(
                case,
                position,
                target,
                case_timeout_s=config.case_timeout_s,
                disabled_evaluators=config.disabled_evaluators,
                judge=judge,
            )
        async with write_lock:
            await asyncio.to_thread(store.save_case_outcome, run_id, outcome)
        if on_case_done is not None:
            on_case_done(outcome)
        return outcome

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run_one(i, case)) for i, case in enumerate(cases)]
    except BaseException as exc:
        # Keep the partial run visible as "aborted" rather than stuck in "running".
        await asyncio.to_thread(
            store.finish_run, run_id, status="aborted", summary=None, error=repr(exc)[:1000]
        )
        raise

    outcomes = sorted((task.result() for task in tasks), key=lambda o: o.position)
    summary = summarize(outcomes, judge_enabled=judge is not None)
    await asyncio.to_thread(store.finish_run, run_id, status="completed", summary=summary)
    return RunReport(run_id=run_id, summary=summary, outcomes=outcomes)


def summarize(outcomes: Sequence[CaseOutcome], *, judge_enabled: bool) -> dict[str, Any]:
    """Aggregate a run without collapsing everything into one accuracy number.

    Deterministic outcomes, judge scores, latency and token usage are reported
    side by side, each with the number of observations behind it.
    """
    statuses = Counter(o.status for o in outcomes)

    by_evaluator: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        for result in outcome.deterministic_results:
            entry = by_evaluator.setdefault(result.evaluator_id, {"passed": 0, "failed": 0})
            entry["passed" if result.passed else "failed"] += 1

    failing = [o for o in outcomes if o.status in ("failed", "error")]
    judge_results = [r for o in outcomes for r in o.judge_results]
    scores = [r.score for r in judge_results if r.score is not None]

    latencies = [
        o.latency_ms for o in outcomes if o.status != "skipped" and o.latency_ms is not None
    ]
    model_spans = [s for o in outcomes for s in o.spans if s["kind"] == "model"]
    reported = [s["token_usage"] for s in model_spans if s.get("token_usage")]

    def token_sum(key: str) -> int | None:
        values = [u[key] for u in reported if isinstance(u.get(key), int)]
        return sum(values) if values else None

    return {
        "total": len(outcomes),
        "passed": statuses.get("passed", 0),
        "failed": statuses.get("failed", 0),
        "error": statuses.get("error", 0),
        "skipped": statuses.get("skipped", 0),
        "failed_case_ids": [o.case.case_id for o in failing],
        "failed_by_severity": dict(Counter(o.case.severity for o in failing)),
        "by_evaluator": by_evaluator,
        "judge": {
            "enabled": judge_enabled,
            "cases_scored": len(scores),
            "no_verdict": sum(1 for r in judge_results if r.passed is None),
            "below_threshold": sum(1 for r in judge_results if r.passed is False),
            "mean_score": round(statistics.fmean(scores), 3) if scores else None,
            "scale_max": 5,
            "simulated": any(r.details.get("simulated") for r in judge_results),
        },
        "latency_ms": {
            "count": len(latencies),
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p50": round(statistics.median(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "token_usage": {
            "model_calls": len(model_spans),
            "model_calls_with_usage": len(reported),
            "prompt_tokens": token_sum("prompt_tokens"),
            "completion_tokens": token_sum("completion_tokens"),
        },
    }
