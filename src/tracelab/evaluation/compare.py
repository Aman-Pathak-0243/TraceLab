"""Compare two persisted runs (base -> head).

Absolute values for each run are kept separate from deltas, and a metric that
is missing in either run is reported as unavailable instead of being treated
as zero.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from tracelab.storage.store import Store

FAILING = ("failed", "error")

CATEGORIES = (
    "newly_failed",  # passed in base, failing in head  -> regression
    "newly_passed",  # failing in base, passed in head  -> fix
    "still_failing",
    "passed_both",
    "skipped",  # skipped in either run
    "only_in_base",  # removed from the dataset (or not selected)
    "only_in_head",  # added to the dataset (or not selected)
)


@dataclass
class CaseChange:
    case_id: str
    category: str
    severity: str | None
    base_status: str | None
    head_status: str | None
    evaluator_changes: list[dict[str, Any]] = field(default_factory=list)
    judge_base: float | None = None
    judge_head: float | None = None
    latency_base_ms: float | None = None
    latency_head_ms: float | None = None

    @property
    def judge_delta(self) -> float | None:
        if self.judge_base is None or self.judge_head is None:
            return None
        return round(self.judge_head - self.judge_base, 3)


@dataclass
class Comparison:
    base_run: dict[str, Any]
    head_run: dict[str, Any]
    cases: list[CaseChange]
    config_changes: list[dict[str, Any]]
    dataset_changed: bool
    metrics: dict[str, Any]

    def by_category(self) -> dict[str, list[CaseChange]]:
        groups: dict[str, list[CaseChange]] = {name: [] for name in CATEGORIES}
        for change in self.cases:
            groups[change.category].append(change)
        return groups

    @property
    def has_regressions(self) -> bool:
        return any(c.category == "newly_failed" for c in self.cases)

    @property
    def lost_coverage(self) -> list[CaseChange]:
        """Cases that passed in base but were not executed in head (skipped or missing)."""
        return [
            c
            for c in self.cases
            if c.base_status == "passed" and c.head_status in (None, "skipped")
        ]

    @property
    def warnings(self) -> list[str]:
        notes = []
        for label, run in (("base", self.base_run), ("head", self.head_run)):
            if run["status"] != "completed":
                notes.append(f"{label} run {run['run_id']} has status {run['status']!r}")
        if self.lost_coverage:
            notes.append(
                f"{len(self.lost_coverage)} case(s) that passed in base were not executed in head"
            )
        if self.dataset_changed:
            notes.append("the dataset changed between the runs (different sha256)")
        return notes

    @property
    def gate_passed(self) -> bool:
        """Regression gate: no new failures, a completed head run, and no lost coverage.

        A head run that never executed a previously passing case cannot prove that
        the case still passes, so it does not pass the gate.
        """
        return (
            not self.has_regressions
            and self.head_run["status"] == "completed"
            and not self.lost_coverage
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_run_id": self.base_run["run_id"],
            "head_run_id": self.head_run["run_id"],
            "has_regressions": self.has_regressions,
            "gate_passed": self.gate_passed,
            "warnings": self.warnings,
            "lost_coverage": [c.case_id for c in self.lost_coverage],
            "dataset_changed": self.dataset_changed,
            "config_changes": self.config_changes,
            "metrics": self.metrics,
            "categories": {
                name: [asdict(c) | {"judge_delta": c.judge_delta} for c in changes]
                for name, changes in self.by_category().items()
            },
        }


class RunNotFoundError(LookupError):
    pass


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat |= _flatten(item, f"{prefix}.{key}" if prefix else str(key))
        return flat
    return {prefix: value}


def _config_changes(base: dict[str, Any], head: dict[str, Any]) -> list[dict[str, Any]]:
    flat_base = _flatten({"config": base["config"], "target": base["target"]})
    flat_head = _flatten({"config": head["config"], "target": head["target"]})
    return [
        {"key": key, "base": flat_base.get(key), "head": flat_head.get(key)}
        for key in sorted(set(flat_base) | set(flat_head))
        if flat_base.get(key) != flat_head.get(key)
    ]


def _judge_score(case: dict[str, Any]) -> float | None:
    scores = [
        e["score"] for e in case["evaluations"] if e["kind"] == "judge" and e["score"] is not None
    ]
    return scores[0] if scores else None


def _categorize(base: dict[str, Any] | None, head: dict[str, Any] | None) -> str:
    if base is None:
        return "only_in_head"
    if head is None:
        return "only_in_base"
    if "skipped" in (base["status"], head["status"]):
        return "skipped"
    base_fail, head_fail = base["status"] in FAILING, head["status"] in FAILING
    if not base_fail and head_fail:
        return "newly_failed"
    if base_fail and not head_fail:
        return "newly_passed"
    return "still_failing" if head_fail else "passed_both"


def _evaluator_changes(base: dict[str, Any], head: dict[str, Any]) -> list[dict[str, Any]]:
    def verdicts(case: dict[str, Any]) -> dict[str, bool | None]:
        return {
            e["evaluator_id"]: e["passed"]
            for e in case["evaluations"]
            if e["kind"] == "deterministic"
        }

    b, h = verdicts(base), verdicts(head)
    return [
        {"evaluator_id": key, "base": b.get(key), "head": h.get(key)}
        for key in sorted(set(b) | set(h))
        if b.get(key) != h.get(key)
    ]


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def _delta(base: float | None, head: float | None) -> float | None:
    return None if base is None or head is None else round(head - base, 3)


def compare_runs(store: Store, base_run_id: str, head_run_id: str) -> Comparison:
    base_run, head_run = store.get_run(base_run_id), store.get_run(head_run_id)
    if base_run is None:
        raise RunNotFoundError(f"run {base_run_id!r} not found")
    if head_run is None:
        raise RunNotFoundError(f"run {head_run_id!r} not found")

    base_cases = {c["case_id"]: c for c in store.list_case_results(base_run_id)}
    head_cases = {c["case_id"]: c for c in store.list_case_results(head_run_id)}
    order = list(head_cases) + [cid for cid in base_cases if cid not in head_cases]

    changes = []
    for case_id in order:
        base, head = base_cases.get(case_id), head_cases.get(case_id)
        either = head or base
        assert either is not None
        changes.append(
            CaseChange(
                case_id=case_id,
                category=_categorize(base, head),
                severity=either["severity"],
                base_status=base["status"] if base else None,
                head_status=head["status"] if head else None,
                evaluator_changes=_evaluator_changes(base, head) if base and head else [],
                judge_base=_judge_score(base) if base else None,
                judge_head=_judge_score(head) if head else None,
                latency_base_ms=base["latency_ms"] if base else None,
                latency_head_ms=head["latency_ms"] if head else None,
            )
        )

    # Latency is compared only over cases executed in both runs, so adding or
    # removing slow cases does not masquerade as a speed change.
    shared = [
        c
        for c in changes
        if c.latency_base_ms is not None
        and c.latency_head_ms is not None
        and c.category != "skipped"
    ]
    judged = [c for c in changes if c.judge_delta is not None]

    def tokens(run: dict[str, Any]) -> dict[str, Any] | None:
        usage = (run.get("summary") or {}).get("token_usage") or {}
        return usage if usage.get("model_calls_with_usage") else None

    base_tokens, head_tokens = tokens(base_run), tokens(head_run)
    lat_base = _mean([c.latency_base_ms for c in shared])
    lat_head = _mean([c.latency_head_ms for c in shared])
    judge_base = _mean([c.judge_base for c in judged])
    judge_head = _mean([c.judge_head for c in judged])
    metrics = {
        "status_counts": {
            "base": _status_counts(base_cases.values()),
            "head": _status_counts(head_cases.values()),
        },
        "latency_ms_mean_shared_cases": {
            "cases": len(shared),
            "base": lat_base,
            "head": lat_head,
            "delta": _delta(lat_base, lat_head),
        },
        "judge_mean_score_shared_cases": {
            "cases": len(judged),
            "base": judge_base,
            "head": judge_head,
            "delta": _delta(judge_base, judge_head),
        },
        "token_usage": {
            "base": base_tokens,
            "head": head_tokens,
            "available": base_tokens is not None and head_tokens is not None,
        },
    }
    return Comparison(
        base_run=base_run,
        head_run=head_run,
        cases=changes,
        config_changes=_config_changes(base_run, head_run),
        dataset_changed=base_run["dataset_sha256"] != head_run["dataset_sha256"],
        metrics=metrics,
    )


def _status_counts(cases: Any) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for case in cases:
        counts[case["status"]] = counts.get(case["status"], 0) + 1
    return counts
