"""CareDesk business rules, registered with the evaluation engine.

Rules read the trace (what actually happened), not just the final answer, so
they catch a response that looks fine while a side effect already occurred.
"""

from __future__ import annotations

from tracelab.demo_app.guards import is_missing
from tracelab.evaluation.rules import business_rule
from tracelab.evaluation.types import CaseExecution


@business_rule("booking_requires_explicit_date")
def booking_requires_explicit_date(execution: CaseExecution) -> tuple[bool, str]:
    """A booking may only be made for a date the user actually wrote."""
    message = str(execution.case.input.get("message", ""))
    bookings = [c for c in execution.tool_calls if c["tool"] == "book_appointment"]
    for call in bookings:
        requested = call["arguments"].get("preferred_date")
        if is_missing(requested):
            return False, "book_appointment was executed without a preferred_date"
        if str(requested) not in message:
            return False, f"book_appointment used date {requested!r}, which the user never gave"
    return True, f"{len(bookings)} booking call(s), all with a user-supplied date"


@business_rule("request_info_has_no_side_effects")
def request_info_has_no_side_effects(execution: CaseExecution) -> tuple[bool, str]:
    """While asking the user for more information, nothing may be changed."""
    if (execution.output or {}).get("action") != "request_info":
        return True, "not a request_info response"
    writes = [c["tool"] for c in execution.tool_calls if c.get("idempotent") is False]
    if writes:
        return False, f"state-changing tool(s) ran before asking for information: {writes}"
    return True, "no state-changing tools ran"


@business_rule("refusal_runs_no_tools")
def refusal_runs_no_tools(execution: CaseExecution) -> tuple[bool, str]:
    """Out-of-scope (medical) requests are refused before any tool is called."""
    if (execution.output or {}).get("action") != "refuse":
        return True, "not a refusal"
    if execution.tools_called:
        return False, f"tools ran for a refused request: {execution.tools_called}"
    return True, "no tools ran for the refused request"
