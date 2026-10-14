"""CareDesk tools: the functions the model may ask the application to run.

Tools are plain synchronous functions over in-memory fixtures, so calling them
from async code does not block the event loop in any meaningful way. A tool that
performed real I/O would need `asyncio.to_thread` or an async client.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from tracelab.datasets.schema import ToolFault
from tracelab.demo_app.data import (
    APPOINTMENTS,
    MEMBERS,
    NEXT_AVAILABLE_DATE,
    PACKAGES,
    POLICIES,
    REFERENCE_DATE,
)


class ToolError(Exception):
    """A tool failed in an expected, reportable way. `code` is shown to evaluators."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    required_args: tuple[str, ...]
    func: Callable[..., dict[str, Any]]
    optional_args: tuple[str, ...] = ()
    # Non-idempotent tools change state and must never be retried blindly.
    idempotent: bool = True

    @property
    def allowed_args(self) -> set[str]:
        return set(self.required_args) | set(self.optional_args)

    def signature(self) -> str:
        return f"{self.name}({', '.join(self.required_args + self.optional_args)})"


def list_packages() -> dict[str, Any]:
    return {"packages": [dict(p) for p in PACKAGES.values()]}


def get_policy(topic: str) -> dict[str, Any]:
    key = topic.strip().lower()
    if key not in POLICIES:
        raise ToolError("unknown_policy_topic", f"No policy for topic {topic!r}")
    return {"policy": {"topic": key, "text": POLICIES[key]}}


def lookup_appointment(appointment_id: str) -> dict[str, Any]:
    appointment = APPOINTMENTS.get(str(appointment_id).upper())
    if appointment is None:
        raise ToolError("not_found", f"Appointment {appointment_id!r} does not exist")
    package = PACKAGES[appointment["package_id"]]
    return {"appointment": dict(appointment) | {"package_name": package["name"]}}


def book_appointment(
    member_id: str, package_id: str, preferred_date: str | None = None
) -> dict[str, Any]:
    if member_id not in MEMBERS:
        raise ToolError("unknown_member", f"Unknown member {member_id!r}")
    if package_id not in PACKAGES:
        raise ToolError("unknown_package", f"Unknown package {package_id!r}")
    if preferred_date is None:
        # Mirrors a real scheduling API: with no date it books the next free slot.
        # This is exactly why the application must not call it without a date.
        booked_date = NEXT_AVAILABLE_DATE
    else:
        try:
            parsed = date.fromisoformat(preferred_date)
        except ValueError as exc:
            raise ToolError("invalid_date", f"Date {preferred_date!r} is not YYYY-MM-DD") from exc
        if parsed < REFERENCE_DATE:
            raise ToolError("date_in_past", f"Date {preferred_date} is in the past")
        booked_date = parsed.isoformat()
    booking_id = f"BK-{package_id.removeprefix('PKG-')}-{booked_date.replace('-', '')}"
    return {
        "booking": {
            "booking_id": booking_id,
            "member_id": member_id,
            "package_id": package_id,
            "package_name": PACKAGES[package_id]["name"],
            "date": booked_date,
            "status": "confirmed",
        }
    }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("list_packages", "List available preventive-care packages.", (), list_packages),
    ToolSpec(
        "get_policy",
        "Get a service policy. topic is one of: " + ", ".join(sorted(POLICIES)) + ".",
        ("topic",),
        get_policy,
    ),
    ToolSpec(
        "lookup_appointment",
        "Look up an existing appointment by ID, e.g. APT-1001.",
        ("appointment_id",),
        lookup_appointment,
    ),
    ToolSpec(
        "book_appointment",
        "Book a package for a member. package_id is one of "
        + ", ".join(PACKAGES)
        + "; preferred_date is YYYY-MM-DD.",
        ("member_id", "package_id", "preferred_date"),
        book_appointment,
        idempotent=False,
    ),
)


class ToolRegistry:
    def __init__(
        self, specs: tuple[ToolSpec, ...] = TOOL_SPECS, fault: ToolFault | None = None
    ) -> None:
        self._specs = {spec.name: spec for spec in specs}
        self._fault = fault

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def names(self) -> list[str]:
        return list(self._specs)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = self._specs[name]
        if self._fault is not None and self._fault.tool == name:
            if self._fault.kind == "unavailable":
                raise ToolError("tool_unavailable", f"Simulated: {name} backend is unavailable")
            raise ToolError("tool_internal_error", f"Simulated: {name} raised an internal error")
        return spec.func(**arguments)
