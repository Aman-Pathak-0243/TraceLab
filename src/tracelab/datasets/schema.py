"""Schema for evaluation dataset records (one JSON object per JSONL line).

The schema is intentionally strict: unknown keys are rejected (`extra="forbid"`)
so that a typo such as "forbiden_tools" fails validation instead of silently
disabling a check.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tracelab.adapters.mock import MockFault

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

Severity = Literal["critical", "high", "medium", "low"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolFault(_Strict):
    """Make a named tool fail. Works with every model adapter (tools are local)."""

    tool: str
    kind: Literal["unavailable", "internal_error"] = "unavailable"


class Simulation(_Strict):
    """Controlled failure injection for a case.

    `model_fault` is only honoured by the mock adapter; cases that use it are
    skipped (not passed) when a real model is configured.
    """

    model_fault: MockFault | None = None
    tool_fault: ToolFault | None = None


class JudgeSpec(_Strict):
    """Rubric for the optional LLM-as-judge evaluator (integer scale 1-5)."""

    rubric: str = Field(min_length=10)
    min_score: int = Field(default=3, ge=1, le=5)


class Expected(_Strict):
    """Expected behaviour. Each populated field enables one deterministic check."""

    action: str = Field(min_length=1)
    missing_fields: list[str] | None = None
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    fields: dict[str, Any] = Field(default_factory=dict)
    answer_contains: list[str] = Field(default_factory=list)
    answer_not_contains: list[str] = Field(default_factory=list)
    error_code: str | None = None
    business_rules: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _tools_do_not_overlap(self) -> Expected:
        overlap = set(self.required_tools) & set(self.forbidden_tools)
        if overlap:
            raise ValueError(f"tools cannot be both required and forbidden: {sorted(overlap)}")
        return self


class EvalCase(_Strict):
    schema_version: str
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    description: str = Field(min_length=5)
    input: dict[str, Any]
    expected: Expected
    tags: list[str] = Field(default_factory=list)
    severity: Severity = "medium"
    simulate: Simulation | None = None
    judge: JudgeSpec | None = None

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version {value!r}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        return value

    @field_validator("tags")
    @classmethod
    def _unique_tags(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("tags must be unique")
        return value


def schema_json() -> str:
    """JSON Schema for one dataset record, as committed in datasets/schema/."""
    return json.dumps(EvalCase.model_json_schema(), indent=2, sort_keys=True) + "\n"
