"""Core types of the evaluation engine.

The engine knows nothing about CareDesk. It evaluates any `Target` that turns
an `EvalCase` into a JSON object while recording spans on a `Tracer`.

Output contract used by the built-in evaluators (all keys optional):
    action: str, answer: str, missing_fields: list[str], error: {"code": str}, plus
    any nested data addressed by dotted paths in `expected.fields`.
Tool calls are read from the trace (spans of kind "tool"), not from the output,
so an application cannot claim it did not call a tool that it actually called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from tracelab.datasets.schema import EvalCase
from tracelab.tracing.tracer import Tracer, tool_calls_from_spans

EvaluatorKind = Literal["deterministic", "judge"]


@dataclass(frozen=True)
class CaseExecution:
    case: EvalCase
    output: dict[str, Any] | None
    spans: list[dict[str, Any]]
    harness_error: str | None = None

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return tool_calls_from_spans(self.spans)

    @property
    def tools_called(self) -> list[str]:
        return [call["tool"] for call in self.tool_calls]


@dataclass(frozen=True)
class EvalResult:
    evaluator_id: str
    kind: EvaluatorKind
    # None means the evaluator could not reach a verdict (e.g. the judge model was
    # unavailable). It is reported separately and never counted as a pass.
    passed: bool | None
    explanation: str
    score: float | None = None
    max_score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "kind": self.kind,
            "passed": self.passed,
            "explanation": self.explanation,
            "score": self.score,
            "max_score": self.max_score,
            "details": self.details,
        }


CaseStatus = Literal["passed", "failed", "error", "skipped"]


@dataclass(frozen=True)
class CaseOutcome:
    """Everything recorded for one case in one run.

    status: "passed"/"failed" come from deterministic evaluators only; "error"
    means the harness could not execute or evaluate the case; "skipped" means
    the case cannot run in this configuration (see `skip_reason`).
    """

    case: EvalCase
    position: int
    status: CaseStatus
    trace_id: str | None
    output: dict[str, Any] | None
    spans: list[dict[str, Any]]
    results: list[EvalResult]
    latency_ms: float | None
    harness_error: str | None = None
    skip_reason: str | None = None

    @property
    def deterministic_results(self) -> list[EvalResult]:
        return [r for r in self.results if r.kind == "deterministic"]

    @property
    def judge_results(self) -> list[EvalResult]:
        return [r for r in self.results if r.kind == "judge"]

    @property
    def failed_evaluators(self) -> list[str]:
        return [r.evaluator_id for r in self.deterministic_results if r.passed is False]


class Target(Protocol):
    """An application under test."""

    name: str
    output_model: type[BaseModel] | None

    def describe(self) -> dict[str, Any]:
        """Configuration facts stored with each run (provider, model, prompt version...)."""
        ...

    def validate_case(self, case: EvalCase) -> list[str]:
        """Application-specific dataset checks; returns error messages."""
        ...

    def skip_reason(self, case: EvalCase) -> str | None:
        """Why this case cannot run in the current configuration, or None."""
        ...

    async def execute(self, case: EvalCase, tracer: Tracer) -> dict[str, Any]: ...
