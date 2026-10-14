"""Deterministic evaluators.

Each evaluator checks one property and explains its verdict. The checks a case
gets are derived from its `expected` block: every populated field enables the
matching evaluator (`from_case`). A case passes only if all of its
deterministic evaluators pass - valid JSON on its own is never enough.

To add an evaluator: subclass `DeterministicEvaluator`, implement `from_case`
and `evaluate`, and decorate the class with `@register_evaluator`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from tracelab.datasets.schema import EvalCase
from tracelab.evaluation.rules import get_rule, registered_rules
from tracelab.evaluation.types import CaseExecution, EvalResult

MISSING = object()  # sentinel: path not present (distinct from a None value)


def get_path(obj: Any, path: str) -> Any:
    """Read a dotted path such as "data.booking.status" or "data.packages.0.name"."""
    current = obj
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return MISSING
    return current


def strict_equal(actual: Any, expected: Any) -> bool:
    """JSON-style equality: unlike Python's ==, True != 1 and False != 0."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            strict_equal(actual[k], expected[k]) for k in expected
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            strict_equal(a, e) for a, e in zip(actual, expected, strict=True)
        )
    return actual == expected


class DeterministicEvaluator:
    evaluator_id: ClassVar[str]
    kind: ClassVar[str] = "deterministic"

    @classmethod
    def from_case(
        cls, case: EvalCase, output_model: type[BaseModel] | None
    ) -> Sequence[DeterministicEvaluator]:
        raise NotImplementedError

    @property
    def result_id(self) -> str:
        return self.evaluator_id

    def evaluate(self, execution: CaseExecution) -> EvalResult:
        raise NotImplementedError

    def _result(self, passed: bool, explanation: str, **details: Any) -> EvalResult:
        return EvalResult(self.result_id, "deterministic", passed, explanation, details=details)


_REGISTRY: list[type[DeterministicEvaluator]] = []


def register_evaluator(cls: type[DeterministicEvaluator]) -> type[DeterministicEvaluator]:
    if cls not in _REGISTRY:
        _REGISTRY.append(cls)
    return cls


def registered_evaluator_ids() -> list[str]:
    return [cls.evaluator_id for cls in _REGISTRY]


def evaluators_for(
    case: EvalCase,
    output_model: type[BaseModel] | None = None,
    disabled: Sequence[str] = (),
) -> list[DeterministicEvaluator]:
    """Evaluators for one case. `disabled` may name a family ("business_rule") or a
    single result id ("business_rule:booking_requires_explicit_date")."""
    evaluators = [ev for cls in _REGISTRY for ev in cls.from_case(case, output_model)]
    return [
        ev for ev in evaluators if ev.evaluator_id not in disabled and ev.result_id not in disabled
    ]


def unknown_evaluator_names(names: Sequence[str]) -> list[str]:
    """Names in `disabled_evaluators` that match no evaluator or registered rule (likely typos)."""
    known = set(registered_evaluator_ids())
    known |= {f"business_rule:{rule}" for rule in registered_rules()}
    return sorted(set(names) - known)


@register_evaluator
class OutputSchemaEvaluator(DeterministicEvaluator):
    evaluator_id = "output_schema"

    def __init__(self, model: type[BaseModel]) -> None:
        self.model = model

    @classmethod
    def from_case(cls, case, output_model):
        return [cls(output_model)] if output_model is not None else []

    def evaluate(self, execution):
        if execution.output is None:
            return self._result(False, "no output was produced")
        try:
            self.model.model_validate(execution.output)
        except ValidationError as exc:
            errors = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
            return self._result(
                False, f"output does not match {self.model.__name__}", errors=errors
            )
        return self._result(True, f"output matches {self.model.__name__}")


@register_evaluator
class ExpectedActionEvaluator(DeterministicEvaluator):
    evaluator_id = "expected_action"

    def __init__(self, expected: str) -> None:
        self.expected = expected

    @classmethod
    def from_case(cls, case, output_model):
        return [cls(case.expected.action)]

    def evaluate(self, execution):
        actual = (execution.output or {}).get("action")
        return self._result(
            actual == self.expected,
            f"expected action {self.expected!r}, got {actual!r}",
            expected=self.expected,
            actual=actual,
        )


@register_evaluator
class MissingFieldsEvaluator(DeterministicEvaluator):
    evaluator_id = "missing_fields"

    def __init__(self, expected: list[str]) -> None:
        self.expected = sorted(expected)

    @classmethod
    def from_case(cls, case, output_model):
        fields = case.expected.missing_fields
        return [cls(fields)] if fields is not None else []

    def evaluate(self, execution):
        actual = sorted((execution.output or {}).get("missing_fields") or [])
        not_requested = sorted(set(self.expected) - set(actual))
        unexpected = sorted(set(actual) - set(self.expected))
        if not not_requested and not unexpected:
            return self._result(True, f"requested exactly {self.expected}", actual=actual)
        parts = []
        if not_requested:
            parts.append(f"did not request {not_requested}")
        if unexpected:
            parts.append(f"unexpectedly requested {unexpected}")
        return self._result(False, "; ".join(parts), expected=self.expected, actual=actual)


@register_evaluator
class RequiredToolsEvaluator(DeterministicEvaluator):
    evaluator_id = "required_tools"

    def __init__(self, tools: list[str]) -> None:
        self.tools = tools

    @classmethod
    def from_case(cls, case, output_model):
        return [cls(case.expected.required_tools)] if case.expected.required_tools else []

    def evaluate(self, execution):
        calls = execution.tool_calls
        problems = []
        for tool in self.tools:
            attempts = [c for c in calls if c["tool"] == tool]
            if not attempts:
                problems.append(f"{tool} was not called")
            elif not any(c["status"] == "ok" for c in attempts):
                problems.append(f"{tool} was called but did not succeed")
        if problems:
            return self._result(False, "; ".join(problems), called=execution.tools_called)
        return self._result(True, f"called {self.tools}", called=execution.tools_called)


@register_evaluator
class ForbiddenToolsEvaluator(DeterministicEvaluator):
    evaluator_id = "forbidden_tools"

    def __init__(self, tools: list[str]) -> None:
        self.tools = tools

    @classmethod
    def from_case(cls, case, output_model):
        return [cls(case.expected.forbidden_tools)] if case.expected.forbidden_tools else []

    def evaluate(self, execution):
        called = execution.tools_called
        violations = sorted(set(self.tools) & set(called))
        if violations:
            return self._result(
                False, f"forbidden tool(s) were executed: {violations}", called=called
            )
        return self._result(True, f"none of {self.tools} were executed", called=called)


@register_evaluator
class FieldValuesEvaluator(DeterministicEvaluator):
    evaluator_id = "field_values"

    def __init__(self, fields: dict[str, Any]) -> None:
        self.fields = fields

    @classmethod
    def from_case(cls, case, output_model):
        return [cls(case.expected.fields)] if case.expected.fields else []

    def evaluate(self, execution):
        mismatches = {}
        for path, expected in self.fields.items():
            actual = get_path(execution.output or {}, path)
            if actual is MISSING:
                mismatches[path] = {"expected": expected, "actual": "<missing>"}
            elif not strict_equal(actual, expected):
                mismatches[path] = {"expected": expected, "actual": actual}
        if mismatches:
            summary = ", ".join(
                f"{p}: expected {m['expected']!r}, got {m['actual']!r}"
                for p, m in mismatches.items()
            )
            return self._result(False, summary, mismatches=mismatches)
        return self._result(True, f"all {len(self.fields)} field(s) match")


@register_evaluator
class AnswerContainsEvaluator(DeterministicEvaluator):
    evaluator_id = "answer_contains"

    def __init__(self, phrases: list[str]) -> None:
        self.phrases = phrases

    @classmethod
    def from_case(cls, case, output_model):
        return [cls(case.expected.answer_contains)] if case.expected.answer_contains else []

    def evaluate(self, execution):
        answer = str((execution.output or {}).get("answer") or "").lower()
        absent = [p for p in self.phrases if p.lower() not in answer]
        if absent:
            return self._result(False, f"answer is missing {absent}")
        return self._result(True, f"answer contains all of {self.phrases}")


@register_evaluator
class AnswerNotContainsEvaluator(DeterministicEvaluator):
    evaluator_id = "answer_not_contains"

    def __init__(self, phrases: list[str]) -> None:
        self.phrases = phrases

    @classmethod
    def from_case(cls, case, output_model):
        phrases = case.expected.answer_not_contains
        return [cls(phrases)] if phrases else []

    def evaluate(self, execution):
        answer = str((execution.output or {}).get("answer") or "").lower()
        present = [p for p in self.phrases if p.lower() in answer]
        if present:
            return self._result(False, f"answer contains forbidden text {present}")
        return self._result(True, f"answer avoids all of {self.phrases}")


@register_evaluator
class ErrorCodeEvaluator(DeterministicEvaluator):
    evaluator_id = "error_code"

    def __init__(self, expected: str) -> None:
        self.expected = expected

    @classmethod
    def from_case(cls, case, output_model):
        code = case.expected.error_code
        return [cls(code)] if code is not None else []

    def evaluate(self, execution):
        error = (execution.output or {}).get("error") or {}
        actual = error.get("code") if isinstance(error, dict) else None
        return self._result(
            actual == self.expected,
            f"expected error code {self.expected!r}, got {actual!r}",
            expected=self.expected,
            actual=actual,
        )


@register_evaluator
class BusinessRuleEvaluator(DeterministicEvaluator):
    evaluator_id = "business_rule"

    def __init__(self, rule_name: str) -> None:
        self.rule_name = rule_name

    @property
    def result_id(self) -> str:
        return f"business_rule:{self.rule_name}"

    @classmethod
    def from_case(cls, case, output_model):
        return [cls(name) for name in case.expected.business_rules]

    def evaluate(self, execution):
        rule = get_rule(self.rule_name)
        if rule is None:
            return self._result(False, f"business rule {self.rule_name!r} is not registered")
        passed, explanation = rule(execution)
        return self._result(passed, explanation)
