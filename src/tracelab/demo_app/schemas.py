"""Request/response contracts for CareDesk and the model's structured plan."""

from __future__ import annotations

from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Action = Literal["answer", "request_info", "refuse", "error"]
ACTIONS: tuple[str, ...] = ("answer", "request_info", "refuse", "error")


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    # Facts the client already knows, e.g. {"member_id": "M-1001"} for a signed-in user.
    context: dict[str, str] = Field(default_factory=dict)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """The structured output the model must produce in the planning step.

    Extra keys (e.g. a "reasoning" field some models add) are ignored, but the
    required structure is validated strictly.
    """

    model_config = ConfigDict(extra="ignore")

    intent: str = Field(min_length=1)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=3)


class ErrorInfo(BaseModel):
    code: str
    message: str


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    answer: str = Field(min_length=1)
    missing_fields: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    error: ErrorInfo | None = None
    tools_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> AgentResponse:
        if self.action == "request_info" and not self.missing_fields:
            raise ValueError("request_info responses must list missing_fields")
        if self.action != "request_info" and self.missing_fields:
            raise ValueError("missing_fields is only allowed for request_info responses")
        if (self.action == "error") != (self.error is not None):
            raise ValueError("error details are required for, and only for, action='error'")
        return self


@cache
def plan_json_schema() -> dict[str, Any]:
    """JSON schema sent to providers that support constrained decoding (cached; do not mutate)."""
    return Plan.model_json_schema()
