"""Scripted mock "model" for CareDesk.

This is a transparent keyword router, not an LLM. It reads only what a real
model would see (the JSON user message in the prompt), so the application code
path is identical in mock and Ollama mode. It reproduces one realistic model
habit on purpose: when the user omits a required value it emits `null` for that
argument instead of refusing to call the tool.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tracelab.adapters.base import ModelRequest
from tracelab.adapters.mock import MockScript

_APPOINTMENT_ID = re.compile(r"\bAPT-\d{4}\b", re.IGNORECASE)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_PACKAGE_ID = re.compile(r"\bPKG-[A-Z]+\b", re.IGNORECASE)
_PACKAGE_KEYWORDS = {
    "basic": "PKG-BASIC",
    "wellness": "PKG-BASIC",
    "heart": "PKG-HEART",
    "senior": "PKG-SENIOR",
    "ageing": "PKG-SENIOR",
}
_POLICY_KEYWORDS = {
    "cancel": "cancellation",
    "reschedul": "rescheduling",
    "prepar": "preparation",
    "fast": "preparation",
    "pay": "payment",
}


def _last_user_json(request: ModelRequest) -> dict[str, Any]:
    for message in reversed(request.messages):
        if message.role == "user":
            return json.loads(message.content)
    raise ValueError("mock script expected a user message containing JSON")


def plan(message: str, context: dict[str, str]) -> dict[str, Any]:
    text = message.lower()

    if any(word in text for word in ("book", "schedule", "reserve")):
        package = _PACKAGE_ID.search(message)
        package_id = package.group(0).upper() if package else None
        if package_id is None:
            package_id = next((p for k, p in _PACKAGE_KEYWORDS.items() if k in text), None)
        date = _DATE.search(message)
        return {
            "intent": "book_appointment",
            "tool_calls": [
                {
                    "name": "book_appointment",
                    "arguments": {
                        "member_id": context.get("member_id"),
                        "package_id": package_id,
                        "preferred_date": date.group(0) if date else None,
                    },
                }
            ],
        }

    for keyword, topic in _POLICY_KEYWORDS.items():
        if keyword in text:
            return {
                "intent": "policy_question",
                "tool_calls": [{"name": "get_policy", "arguments": {"topic": topic}}],
            }

    appointment = _APPOINTMENT_ID.search(message)
    if appointment or "appointment" in text:
        return {
            "intent": "appointment_lookup",
            "tool_calls": [
                {
                    "name": "lookup_appointment",
                    "arguments": {
                        "appointment_id": appointment.group(0).upper() if appointment else None
                    },
                }
            ],
        }

    if any(word in text for word in ("package", "offer", "screening", "services")):
        return {
            "intent": "list_packages",
            "tool_calls": [{"name": "list_packages", "arguments": {}}],
        }

    return {"intent": "unsupported", "tool_calls": []}


def compose(payload: dict[str, Any]) -> str:
    results = {r["tool"]: r["result"] for r in payload.get("tool_results", [])}
    if "list_packages" in results:
        items = "; ".join(
            f"{p['name']} ({p['package_id']}, ${p['price_usd']})"
            for p in results["list_packages"]["packages"]
        )
        return f"We offer these preventive-care packages: {items}. Would you like to book one?"
    if "lookup_appointment" in results:
        a = results["lookup_appointment"]["appointment"]
        return (
            f"Appointment {a['appointment_id']} ({a['package_name']}) is on {a['date']} at "
            f"{a['time']} at {a['location']}. Status: {a['status']}."
        )
    if "book_appointment" in results:
        b = results["book_appointment"]["booking"]
        return (
            f"Your {b['package_name']} is booked for {b['date']}. "
            f"Booking reference: {b['booking_id']}."
        )
    if "get_policy" in results:
        p = results["get_policy"]["policy"]
        return f"Our {p['topic']} policy: {p['text']}"
    return (
        "I can help with CareDesk preventive-care packages, appointments and service "
        "policies. What would you like to know?"
    )


def respond(request: ModelRequest) -> str:
    payload = _last_user_json(request)
    if request.purpose == "plan":
        return json.dumps(plan(payload["message"], payload.get("context", {})))
    if request.purpose == "compose":
        return compose(payload)
    raise ValueError(f"mock script has no behaviour for purpose {request.purpose!r}")


CAREDESK_MOCK_SCRIPT = MockScript(
    responder=respond,
    fault_outputs={
        # Valid JSON, wrong structure: "tool_calls" is a string, "intent" is missing.
        "wrong_schema": json.dumps({"action": "lookup", "tool_calls": "lookup_appointment"}),
        # The model invents a destructive tool that the application never offered.
        "hallucinated_tool": json.dumps(
            {
                "intent": "account_cleanup",
                "tool_calls": [
                    {"name": "delete_member_records", "arguments": {"member_id": "M-1001"}}
                ],
            }
        ),
    },
)
