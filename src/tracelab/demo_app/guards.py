"""Deterministic guards that run around the model.

These checks deliberately do not rely on the model: whether a request asks for
medical advice, or whether a tool call has all of its required arguments, must
not depend on the model behaving well.
"""

from __future__ import annotations

import re
from typing import Any

from tracelab.demo_app.tools import ToolSpec

# Requests for diagnosis or treatment are out of scope for a booking assistant.
_MEDICAL_ADVICE = re.compile(
    r"\b(diagnos\w*|symptoms?|medications?|medicines?|dosage|dose|prescri\w*|"
    r"treat(ment)?|pain|should i take|is it serious)\b",
    re.IGNORECASE,
)

# Strings models commonly emit instead of admitting a value is unknown.
_PLACEHOLDERS = frozenset(
    {"", "null", "none", "unknown", "n/a", "na", "tbd", "?", "yyyy-mm-dd", "<date>", "date"}
)


def screen_input(message: str) -> str | None:
    """Return the matched phrase if the message asks for medical advice, else None."""
    match = _MEDICAL_ADVICE.search(message)
    return match.group(0) if match else None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS


def find_missing_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    """Required arguments that are absent, null, empty or a placeholder string."""
    return [name for name in spec.required_args if is_missing(arguments.get(name))]


def find_unexpected_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    return sorted(set(arguments) - spec.allowed_args)


def find_mistyped_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    """Arguments that are present but not strings (every CareDesk tool argument is a
    string). None is left to the missing-argument check."""
    return sorted(
        name
        for name in spec.allowed_args
        if arguments.get(name) is not None and not isinstance(arguments[name], str)
    )
