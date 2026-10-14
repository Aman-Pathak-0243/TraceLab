"""Controlled faults used to demonstrate the regression workflow safely.

Each fault reproduces a defect that plausibly occurs in LLM applications. They
are only active when a run explicitly sets `agent.inject_fault` (CLI:
`--inject-fault`), and the setting is stored with the run so a faulty run can
never be mistaken for a normal one. No committed config enables a fault.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tracelab.demo_app.guards import find_missing_arguments
from tracelab.demo_app.tools import ToolSpec

MissingArgsCheck = Callable[[ToolSpec, dict[str, Any]], list[str]]


def presence_only_missing_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    """BUGGY variant of `find_missing_arguments`.

    It only checks that each required key exists. Models frequently emit
    `"preferred_date": null` when the user did not give a date, so this check
    lets the call through and the booking backend silently picks a date the
    user never asked for.
    """
    return [name for name in spec.required_args if name not in arguments]


FAULTS: dict[str, str] = {
    "presence_only_arg_check": (
        "Tool-argument validation only checks that required keys exist, so null or "
        "placeholder values pass and the app proceeds instead of asking for missing data."
    ),
}


def missing_args_check(fault: str | None) -> MissingArgsCheck:
    if fault is None:
        return find_missing_arguments
    if fault == "presence_only_arg_check":
        return presence_only_missing_arguments
    raise ValueError(f"Unknown fault {fault!r}; available: {sorted(FAULTS)}")
