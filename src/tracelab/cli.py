"""Command-line interface: `tracelab <command>` (or `python -m tracelab <command>`).

Exit codes: 0 = success, 1 = evaluation failures / regressions / invalid
dataset, 2 = usage or configuration error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from tracelab.config import ConfigError, RunConfig, Settings, load_run_config, resolve_config_path
from tracelab.datasets.loader import DatasetValidationError, validate_dataset
from tracelab.datasets.schema import schema_json
from tracelab.demo_app.faults import FAULTS
from tracelab.demo_app.schemas import AgentRequest
from tracelab.demo_app.target import CareDeskTarget
from tracelab.evaluation.compare import Comparison, RunNotFoundError, compare_runs
from tracelab.evaluation.runner import RunReport
from tracelab.service import build_target, resolve_dataset_path, run_evaluation
from tracelab.storage.store import Store, StoreError
from tracelab.tracing.render import format_trace
from tracelab.tracing.tracer import Tracer

STATUS_LABEL = {"passed": "PASS", "failed": "FAIL", "error": "ERROR", "skipped": "SKIP"}
DEMO_CASE = "book-missing-date-001"
DEMO_FAULT = "presence_only_arg_check"


class CLIError(Exception):
    """A user-facing error; printed without a traceback, exit code 2."""


# --------------------------------------------------------------------- helpers


def _store(settings: Settings) -> Store:
    return Store(settings.db_path)


def _load_config(args: argparse.Namespace, settings: Settings) -> RunConfig:
    config = load_run_config(resolve_config_path(settings, args.config))
    data = config.model_dump()
    if getattr(args, "model", None):
        data["adapter"]["model"] = args.model
    if getattr(args, "prompt_version", None):
        data["agent"]["prompt_version"] = args.prompt_version
    if getattr(args, "inject_fault", None):
        data["agent"]["inject_fault"] = args.inject_fault
    if getattr(args, "no_judge", False):
        data["judge"]["enabled"] = False
    return RunConfig.model_validate(data)


def _resolve_run(store: Store, ref: str) -> str:
    if ref in ("latest", "previous"):
        runs = store.list_runs(limit=2)
        index = 0 if ref == "latest" else 1
        if len(runs) <= index:
            raise CLIError(f"there is no {ref!r} run in {store.path}")
        return runs[index]["run_id"]
    return ref


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _signed(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}" if isinstance(value, float) else f"{value:+d}"


def _print_case_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'STATUS':<7} {'CASE':<30} {'SEVERITY':<9} FAILED CHECKS / NOTE")
    for row in rows:
        note = ", ".join(row["failed"]) or row.get("note") or ""
        print(f"{STATUS_LABEL[row['status']]:<7} {row['case_id']:<30} {row['severity']:<9} {note}")


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        f"\nDeterministic: {summary['passed']} passed, {summary['failed']} failed, "
        f"{summary['error']} error, {summary['skipped']} skipped (of {summary['total']} cases)"
    )
    if summary["failed_by_severity"]:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(summary["failed_by_severity"].items()))
        print(f"Failing cases by severity: {parts}")
    judge = summary["judge"]
    if judge["enabled"]:
        label = " (SIMULATED mock judge - not a quality signal)" if judge["simulated"] else ""
        print(
            f"Judge{label}: {judge['cases_scored']} scored, mean "
            f"{_fmt(judge['mean_score'], 2)}/{judge['scale_max']}, "
            f"{judge['below_threshold']} below threshold, {judge['no_verdict']} no verdict"
        )
    else:
        print("Judge: disabled")
    lat = summary["latency_ms"]
    print(
        f"Latency (ms): mean {_fmt(lat['mean'])}, p50 {_fmt(lat['p50'])}, "
        f"max {_fmt(lat['max'])} over {lat['count']} executed cases"
    )
    tokens = summary["token_usage"]
    if tokens["model_calls_with_usage"]:
        print(
            f"Token usage (provider-reported, {tokens['model_calls_with_usage']} of "
            f"{tokens['model_calls']} model calls): prompt {tokens['prompt_tokens']}, "
            f"completion {tokens['completion_tokens']}"
        )
    else:
        print(
            f"Token usage: not reported by the provider (0 of {tokens['model_calls']} model calls)"
        )


def _print_report(report: RunReport, config: RunConfig, store: Store) -> None:
    run = store.get_run(report.run_id) or {}
    target = run.get("target", {})
    print(
        f"Run {report.run_id}  config={config.name}  provider={target.get('provider')}  "
        f"model={target.get('model')}  prompt={target.get('prompt_version')}  "
        f"fault={target.get('inject_fault') or '-'}"
    )
    print(f"Dataset {run.get('dataset_path')} (sha256 {str(run.get('dataset_sha256'))[:12]})\n")
    _print_case_table(
        [
            {
                "status": o.status,
                "case_id": o.case.case_id,
                "severity": o.case.severity,
                "failed": o.failed_evaluators,
                "note": o.skip_reason or o.harness_error,
            }
            for o in report.outcomes
        ]
    )
    _print_summary(report.summary)
    print(
        f"\nInspect: tracelab show {report.run_id} --failed | "
        f"tracelab trace {report.run_id} <case_id> | tracelab serve"
    )


def _print_comparison(comparison: Comparison) -> None:
    base, head = comparison.base_run, comparison.head_run
    print(
        f"Compare base {base['run_id']} ({base['config_name']}) -> head {head['run_id']} ({head['config_name']})"
    )
    if comparison.config_changes:
        print("Configuration changes:")
        for change in comparison.config_changes:
            print(f"  {change['key']}: {change['base']!r} -> {change['head']!r}")
    else:
        print("Configuration changes: none")
    if comparison.dataset_changed:
        print("Dataset: CHANGED between runs (compare only overlapping cases with care)")
    else:
        print(f"Dataset: unchanged (sha256 {head['dataset_sha256'][:12]})")
    for warning in comparison.warnings:
        print(f"WARNING: {warning}")

    groups = comparison.by_category()
    titles = {
        "newly_failed": "Newly failed (REGRESSIONS)",
        "newly_passed": "Newly passed (fixed)",
        "still_failing": "Still failing",
    }
    for key, title in titles.items():
        print(f"\n{title}: {len(groups[key])}")
        for change in groups[key]:
            evaluators = ", ".join(
                f"{e['evaluator_id']}: {e['base']}->{e['head']}" for e in change.evaluator_changes
            )
            print(
                f"  {change.case_id} [{change.severity}] {change.base_status} -> "
                f"{change.head_status}" + (f"  ({evaluators})" if evaluators else "")
            )
    print(
        f"\nPassed in both: {len(groups['passed_both'])} | skipped: {len(groups['skipped'])} | "
        f"only in base: {len(groups['only_in_base'])} | only in head: {len(groups['only_in_head'])}"
    )

    m = comparison.metrics
    sb, sh = m["status_counts"]["base"], m["status_counts"]["head"]
    print("\nMetrics (base | head | delta):")
    for key in ("passed", "failed", "error", "skipped"):
        print(f"  {key:<8} {sb[key]:>4} | {sh[key]:>4} | {_signed(sh[key] - sb[key])}")
    lat = m["latency_ms_mean_shared_cases"]
    print(
        f"  mean latency ms over {lat['cases']} shared cases: {_fmt(lat['base'])} | "
        f"{_fmt(lat['head'])} | {_signed(lat['delta'])}"
    )
    judge = m["judge_mean_score_shared_cases"]
    print(
        f"  mean judge score over {judge['cases']} shared cases: {_fmt(judge['base'], 2)} | "
        f"{_fmt(judge['head'], 2)} | {_signed(judge['delta'])}"
    )
    if m["token_usage"]["available"]:
        tb, th = m["token_usage"]["base"], m["token_usage"]["head"]
        print(
            f"  completion tokens: {tb['completion_tokens']} | {th['completion_tokens']} | "
            f"{_signed((th['completion_tokens'] or 0) - (tb['completion_tokens'] or 0))}"
        )
    else:
        print("  token usage: unavailable (not reported by the provider in both runs)")
    lost = ", ".join(c.case_id for c in comparison.lost_coverage)
    print(
        f"\nRegression gate: {'PASSED' if comparison.gate_passed else 'FAILED'}"
        + (f" (passed in base, not executed in head: {lost})" if lost else "")
    )


# -------------------------------------------------------------------- commands


def cmd_validate(args: argparse.Namespace, settings: Settings) -> int:
    validators = []
    if not args.schema_only:
        config = load_run_config(resolve_config_path(settings, args.config))
        validators = [build_target(config, settings).validate_case]
        paths = [Path(p) for p in args.datasets] or [resolve_dataset_path(config, settings)]
    else:
        paths = [Path(p) for p in args.datasets]
        if not paths:
            raise CLIError("--schema-only needs at least one dataset path")
    exit_code = 0
    for path in paths:
        cases, issues, digest = validate_dataset(path, validators)
        if issues:
            exit_code = 1
            print(f"INVALID {path}: {len(issues)} issue(s), {len(cases)} valid record(s)")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"OK      {path}: {len(cases)} cases, sha256 {digest[:12]}")
    return exit_code


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    config = _load_config(args, settings)
    store = _store(settings)
    report = asyncio.run(run_evaluation(config, settings, store, case_ids=args.case or None))
    _print_report(report, config, store)
    summary = report.summary
    executed = summary["passed"] + summary["failed"] + summary["error"]
    if executed == 0:
        print("\nNo case was executed (all skipped); nothing was verified.")
        return 1
    return 0 if summary["failed"] == 0 and summary["error"] == 0 else 1


def cmd_runs(args: argparse.Namespace, settings: Settings) -> int:
    runs = _store(settings).list_runs(limit=args.limit)
    if not runs:
        print("No runs yet. Start one with: tracelab run")
        return 0
    print(f"{'RUN ID':<22} {'CREATED (UTC)':<20} {'CONFIG':<10} {'MODEL':<18} {'FAULT':<24} RESULT")
    for run in runs:
        s = run["summary"] or {}
        result = (
            f"{s.get('passed', 0)}/{s.get('total', 0)} passed, {s.get('failed', 0)} failed, "
            f"{s.get('error', 0)} error, {s.get('skipped', 0)} skipped"
            if s
            else run["status"]
        )
        print(
            f"{run['run_id']:<22} {run['created_at'][:19]:<20} {run['config_name']:<10} "
            f"{str(run['target'].get('model'))[:18]:<18} "
            f"{str(run['target'].get('inject_fault') or '-'):<24} {result}"
        )
    return 0


def cmd_show(args: argparse.Namespace, settings: Settings) -> int:
    store = _store(settings)
    run_id = _resolve_run(store, args.run_id)
    run = store.get_run(run_id)
    if run is None:
        raise CLIError(f"run {run_id!r} not found")
    cases = store.list_case_results(run_id)
    if args.failed:
        cases = [c for c in cases if c["status"] in ("failed", "error")]
    print(
        f"Run {run_id}  status={run['status']}  config={run['config_name']}  "
        f"model={run['target'].get('model')}  fault={run['target'].get('inject_fault') or '-'}\n"
    )
    _print_case_table(
        [
            {
                "status": c["status"],
                "case_id": c["case_id"],
                "severity": c["severity"],
                "failed": [
                    e["evaluator_id"]
                    for e in c["evaluations"]
                    if e["kind"] == "deterministic" and e["passed"] is False
                ],
                "note": c["skip_reason"] or c["harness_error"],
            }
            for c in cases
        ]
    )
    for case in cases:
        failing = [e for e in case["evaluations"] if e["passed"] is False]
        if failing:
            print(f"\n{case['case_id']}  (trace {case['trace_id']})")
            for e in failing:
                print(f"  x {e['evaluator_id']}: {e['explanation']}")
    if run["summary"]:
        _print_summary(run["summary"])
    return 0


def cmd_trace(args: argparse.Namespace, settings: Settings) -> int:
    store = _store(settings)
    if args.trace_id:
        trace_id = args.trace_id
    else:
        if not args.run_id or not args.case_id:
            raise CLIError("give RUN_ID CASE_ID, or --trace-id")
        run_id = _resolve_run(store, args.run_id)
        case = store.get_case_result(run_id, args.case_id)
        if case is None:
            raise CLIError(f"case {args.case_id!r} not found in run {run_id}")
        if case["trace_id"] is None:
            raise CLIError(f"case {args.case_id!r} has no trace (status {case['status']})")
        trace_id = case["trace_id"]
        print(f"Case {case['case_id']} in run {run_id}: {case['status'].upper()}")
        for e in case["evaluations"]:
            mark = {True: "ok", False: "x ", None: "? "}[e["passed"]]
            print(f"  {mark} [{e['kind']}] {e['evaluator_id']}: {e['explanation']}")
        print()
    trace = store.get_trace(trace_id)
    if trace is None:
        raise CLIError(f"trace {trace_id!r} not found")
    print(f"Trace {trace_id}  source={trace['source']}")
    print(f"Input:  {json.dumps(trace['input'])}")
    print(f"Output: {json.dumps(trace['output'])}")
    if trace["error"]:
        print(f"Harness error: {trace['error']}")
    print()
    print(format_trace(trace["spans"], show_attributes=not args.no_attrs))
    return 0


def cmd_compare(args: argparse.Namespace, settings: Settings) -> int:
    store = _store(settings)
    comparison = compare_runs(store, _resolve_run(store, args.base), _resolve_run(store, args.head))
    if args.json:
        print(json.dumps(comparison.to_dict(), indent=2))
    else:
        _print_comparison(comparison)
    return 0 if comparison.gate_passed else 1


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    config = _load_config(args, settings)
    context = {}
    for item in args.context or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise CLIError(f"--context expects key=value, got {item!r}")
        context[key] = value
    agent = CareDeskTarget(config.adapter, config.agent, settings).build_agent()
    request = AgentRequest(message=args.message, context=context)
    tracer = Tracer()
    response = asyncio.run(agent.handle(request, tracer))
    spans = [s.to_dict() for s in tracer.spans]
    _store(settings).save_trace(
        trace_id=tracer.trace_id,
        source="cli",
        input=request.model_dump(),
        output=response.model_dump(mode="json"),
        spans=spans,
    )
    print(json.dumps(response.model_dump(mode="json"), indent=2))
    print(
        f"\nTrace {tracer.trace_id} (saved; view with: tracelab trace --trace-id {tracer.trace_id})"
    )
    print(format_trace(spans, show_attributes=args.attrs))
    return 0


def cmd_schema(args: argparse.Namespace, settings: Settings) -> int:
    text = schema_json()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return 0


def cmd_faults(args: argparse.Namespace, settings: Settings) -> int:
    for name, description in FAULTS.items():
        print(f"{name}\n  {description}")
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    # The app factory reads settings from the environment in the server process.
    os.environ["TRACELAB_DB_PATH"] = str(settings.db_path)
    print(
        f"Dashboard: http://{args.host}:{args.port}/   API docs: http://{args.host}:{args.port}/docs"
    )
    uvicorn.run("tracelab.api.app:create_app", factory=True, host=args.host, port=args.port)
    return 0


def cmd_demo_regression(args: argparse.Namespace, settings: Settings) -> int:
    """Baseline -> faulty build -> fixed build, using fault injection instead of
    editing tracked code. Exits 0 only if every expected step is observed."""
    store = _store(settings)
    base = load_run_config(resolve_config_path(settings, args.config))
    faulty = base.model_copy(
        update={"agent": base.agent.model_copy(update={"inject_fault": DEMO_FAULT})}
    )

    def run(config: RunConfig) -> RunReport:
        return asyncio.run(run_evaluation(config, settings, store))

    def banner(text: str) -> None:
        print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")

    banner("STEP 1  Baseline run (correct build)")
    baseline = run(base)
    _print_report(baseline, base, store)

    banner(f"STEP 2  Same suite with controlled fault '{DEMO_FAULT}' (simulated bad change)")
    print(FAULTS[DEMO_FAULT])
    broken = run(faulty)
    _print_report(broken, faulty, store)

    banner("STEP 3  Regression gate: compare baseline -> faulty build")
    regression = compare_runs(store, baseline.run_id, broken.run_id)
    _print_comparison(regression)

    banner(f"STEP 4  Inspect the trace of the critical failure ({DEMO_CASE})")
    case = store.get_case_result(broken.run_id, DEMO_CASE)
    if case is None or case["trace_id"] is None:
        print(f"Case {DEMO_CASE} missing from run {broken.run_id}")
        return 1
    trace = store.get_trace(case["trace_id"]) or {"spans": []}
    print(format_trace(trace["spans"]))
    print(
        "\nDiagnosis: validate.tool_call reports outcome 'valid' although "
        "arguments.preferred_date is null, so tool.book_appointment ran and the backend "
        "picked a date the user never gave. The faulty step is argument validation."
    )

    banner("STEP 5  Fixed build (fault removed) and comparison faulty -> fixed")
    fixed = run(base)
    _print_report(fixed, base, store)
    print()
    resolution = compare_runs(store, broken.run_id, fixed.run_id)
    _print_comparison(resolution)

    newly_failed = {c.case_id for c in regression.by_category()["newly_failed"]}
    newly_passed = {c.case_id for c in resolution.by_category()["newly_passed"]}
    checks = {
        "baseline has no failures": baseline.summary["failed"] + baseline.summary["error"] == 0,
        f"faulty build fails {DEMO_CASE}": DEMO_CASE in newly_failed,
        "fixed build has no failures": fixed.summary["failed"] + fixed.summary["error"] == 0,
        f"fix resolves {DEMO_CASE}": DEMO_CASE in newly_passed,
        "fix introduces no regressions": resolution.gate_passed,
    }
    banner("RESULT")
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAILED'}] {name}")
    print(
        "\nReproduce with pytest (the regression test fails with the fault, passes without):\n"
        f"  TRACELAB_INJECT_FAULT={DEMO_FAULT} pytest tests/test_regression_missing_info.py\n"
        "  pytest tests/test_regression_missing_info.py"
    )
    return 0 if all(checks.values()) else 1


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracelab", description="Evaluate, trace and regression-test LLM applications."
    )
    parser.add_argument("--db", help="SQLite database path (default: .tracelab/tracelab.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    def config_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default="baseline", help="config name in configs/ or a path")

    p = sub.add_parser("validate", help="validate dataset files")
    p.add_argument("datasets", nargs="*", help="dataset paths (default: the config's dataset)")
    config_arg(p)
    p.add_argument("--schema-only", action="store_true", help="skip application-specific checks")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="run an evaluation suite")
    config_arg(p)
    p.add_argument("--case", action="append", help="run only this case id (repeatable)")
    p.add_argument("--model", help="override the adapter model name")
    p.add_argument("--prompt-version", help="override agent.prompt_version")
    p.add_argument("--inject-fault", choices=sorted(FAULTS), help="enable a controlled fault")
    p.add_argument("--no-judge", action="store_true", help="disable the LLM judge")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("runs", help="list recent runs")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("show", help="show case results of a run")
    p.add_argument("run_id", help="run id, 'latest' or 'previous'")
    p.add_argument("--failed", action="store_true", help="only failed or errored cases")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("trace", help="print the execution trace of a case")
    p.add_argument("run_id", nargs="?")
    p.add_argument("case_id", nargs="?")
    p.add_argument("--trace-id", help="print a trace by id instead")
    p.add_argument("--no-attrs", action="store_true", help="hide span attributes")
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("compare", help="compare two runs (exit 1 if the regression gate fails)")
    p.add_argument("base", help="base run id, 'latest' or 'previous'")
    p.add_argument("head", help="head run id, 'latest' or 'previous'")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("ask", help="send one message to the CareDesk demo app and trace it")
    p.add_argument("message")
    p.add_argument("--context", action="append", help="context entry key=value (repeatable)")
    config_arg(p)
    p.add_argument("--inject-fault", choices=sorted(FAULTS))
    p.add_argument("--attrs", action="store_true", help="show span attributes")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("schema", help="print the dataset record JSON schema")
    p.add_argument("--output", help="write to this file instead of stdout")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("faults", help="list controlled faults")
    p.set_defaults(func=cmd_faults)

    p = sub.add_parser("serve", help="start the API and dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("demo-regression", help="baseline -> injected bug -> fix walkthrough")
    config_arg(p)
    p.set_defaults(func=cmd_demo_regression)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.db:
        settings = settings.model_copy(update={"db_path": Path(args.db).resolve()})
    try:
        return args.func(args, settings)
    except DatasetValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (CLIError, ConfigError, RunNotFoundError, StoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # Raised for invalid configuration values (unknown provider, fault, prompt
        # version or case id) before any run is created.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
