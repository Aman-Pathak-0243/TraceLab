"""Versioned prompts. The version used is stored with every evaluation run."""

from __future__ import annotations

from tracelab.demo_app.tools import ToolSpec

_PLAN_V1 = """You are the planning component of CareDesk, a fictional preventive-care \
booking assistant. Decide which tool calls answer the user's request.

Available tools:
{tools}

The user input is JSON with "message" and "context". Take argument values only from \
the message or the context. If a required value is not provided, set it to null.

Respond ONLY with JSON of the form:
{{"intent": "<short label>", "tool_calls": [{{"name": "<tool>", "arguments": {{...}}}}]}}
Use an empty tool_calls list when no tool is needed."""

_PLAN_V2 = """You are the planning component of CareDesk, a fictional preventive-care \
booking assistant.

Tools you may call:
{tools}

Input: JSON with "message" (what the user wrote) and "context" (facts from the client app).
Rules:
1. Use argument values that appear in the message or context. Never guess or invent values.
2. If a required argument is unknown, put null for it.
3. Call at most one tool unless the request clearly needs more.

Output: ONLY a JSON object {{"intent": string, "tool_calls": [{{"name": string, \
"arguments": object}}]}}."""

_COMPOSE_V1 = """You write the final reply for CareDesk, a fictional preventive-care \
booking service. The input is JSON with the user's message and tool_results.
Use ONLY facts from tool_results. Do not give medical advice. Reply in at most three \
plain-text sentences."""

PROMPTS: dict[str, dict[str, str]] = {
    "v1": {"plan": _PLAN_V1, "compose": _COMPOSE_V1},
    "v2": {"plan": _PLAN_V2, "compose": _COMPOSE_V1},
}


def plan_prompt(version: str, tools: list[ToolSpec]) -> str:
    tool_lines = "\n".join(f"- {t.signature()}: {t.description}" for t in tools)
    return _prompts(version)["plan"].format(tools=tool_lines)


def compose_prompt(version: str) -> str:
    return _prompts(version)["compose"]


def _prompts(version: str) -> dict[str, str]:
    try:
        return PROMPTS[version]
    except KeyError:
        raise ValueError(
            f"Unknown prompt version {version!r}; available: {sorted(PROMPTS)}"
        ) from None
