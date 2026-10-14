"""Registry for named business rules.

Applications register invariants that must hold for every relevant execution,
and dataset cases opt in by name via `expected.business_rules`. A rule returns
(passed, explanation).

    @business_rule("request_info_has_no_side_effects")
    def _rule(execution: CaseExecution) -> tuple[bool, str]: ...
"""

from __future__ import annotations

from collections.abc import Callable

from tracelab.evaluation.types import CaseExecution

RuleFn = Callable[[CaseExecution], tuple[bool, str]]

_RULES: dict[str, RuleFn] = {}


def business_rule(name: str) -> Callable[[RuleFn], RuleFn]:
    def register(fn: RuleFn) -> RuleFn:
        if name in _RULES and _RULES[name] is not fn:
            raise ValueError(f"business rule {name!r} is already registered")
        _RULES[name] = fn
        return fn

    return register


def get_rule(name: str) -> RuleFn | None:
    return _RULES.get(name)


def registered_rules() -> list[str]:
    return sorted(_RULES)
