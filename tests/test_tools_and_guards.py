"""CareDesk tools and the deterministic guards around the model."""

from __future__ import annotations

import pytest

from tracelab.datasets.schema import ToolFault
from tracelab.demo_app.data import NEXT_AVAILABLE_DATE
from tracelab.demo_app.faults import missing_args_check, presence_only_missing_arguments
from tracelab.demo_app.guards import (
    find_missing_arguments,
    find_unexpected_arguments,
    is_missing,
    screen_input,
)
from tracelab.demo_app.tools import (
    ToolError,
    ToolRegistry,
    book_appointment,
    get_policy,
    list_packages,
    lookup_appointment,
)

BOOK = ToolRegistry().get("book_appointment")


def test_list_packages_returns_all_fixtures() -> None:
    ids = [p["package_id"] for p in list_packages()["packages"]]
    assert ids == ["PKG-BASIC", "PKG-HEART", "PKG-SENIOR"]


def test_get_policy_known_and_unknown_topics() -> None:
    assert "24 hours" in get_policy("Cancellation")["policy"]["text"]
    with pytest.raises(ToolError) as excinfo:
        get_policy("refunds")
    assert excinfo.value.code == "unknown_policy_topic"


def test_lookup_appointment_found_and_not_found() -> None:
    assert (
        lookup_appointment("apt-1001")["appointment"]["package_name"] == "Basic Wellness Screening"
    )
    with pytest.raises(ToolError) as excinfo:
        lookup_appointment("APT-9999")
    assert excinfo.value.code == "not_found"


def test_booking_with_a_date() -> None:
    booking = book_appointment("M-1001", "PKG-BASIC", "2026-11-20")["booking"]
    assert booking == {
        "booking_id": "BK-BASIC-20261120",
        "member_id": "M-1001",
        "package_id": "PKG-BASIC",
        "package_name": "Basic Wellness Screening",
        "date": "2026-11-20",
        "status": "confirmed",
    }


def test_backend_books_next_free_slot_when_date_is_none() -> None:
    """Documents why the app must never call this tool without a date."""
    assert book_appointment("M-1001", "PKG-HEART", None)["booking"]["date"] == NEXT_AVAILABLE_DATE


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (("M-9999", "PKG-BASIC", "2026-11-20"), "unknown_member"),
        (("M-1001", "PKG-GOLD", "2026-11-20"), "unknown_package"),
        (("M-1001", "PKG-BASIC", "20/11/2026"), "invalid_date"),
        (("M-1001", "PKG-BASIC", "2026-10-01"), "date_in_past"),
    ],
)
def test_booking_errors(args: tuple[str, str, str], code: str) -> None:
    with pytest.raises(ToolError) as excinfo:
        book_appointment(*args)
    assert excinfo.value.code == code


@pytest.mark.parametrize(
    ("kind", "code"),
    [("unavailable", "tool_unavailable"), ("internal_error", "tool_internal_error")],
)
def test_registry_fault_injection(kind: str, code: str) -> None:
    registry = ToolRegistry(fault=ToolFault(tool="list_packages", kind=kind))
    with pytest.raises(ToolError) as excinfo:
        registry.execute("list_packages", {})
    assert excinfo.value.code == code
    assert registry.execute("get_policy", {"topic": "payment"})  # other tools unaffected


@pytest.mark.parametrize(
    ("message", "blocked"),
    [
        ("Which medication should I take for chest pain?", True),
        ("What dose of ibuprofen is safe?", True),
        ("Can you diagnose my rash?", True),
        ("What packages do you offer?", False),
        ("Do I need to fast before my screening?", False),
    ],
)
def test_screen_input(message: str, blocked: bool) -> None:
    assert (screen_input(message) is not None) is blocked


@pytest.mark.parametrize(
    ("value", "missing"),
    [
        (None, True),
        ("", True),
        ("   ", True),
        ("null", True),
        ("Unknown", True),
        ("YYYY-MM-DD", True),
        ("TBD", True),
        ("2026-11-20", False),
        ("M-1001", False),
        (0, False),
    ],
)
def test_is_missing(value: object, missing: bool) -> None:
    assert is_missing(value) is missing


def test_find_missing_and_unexpected_arguments() -> None:
    args = {"member_id": "M-1001", "package_id": "PKG-HEART", "preferred_date": None, "notes": "x"}
    assert find_missing_arguments(BOOK, args) == ["preferred_date"]
    assert find_unexpected_arguments(BOOK, args) == ["notes"]


def test_buggy_presence_only_check_misses_null_values() -> None:
    args = {"member_id": "M-1001", "package_id": "PKG-HEART", "preferred_date": None}
    assert presence_only_missing_arguments(BOOK, args) == []  # the bug
    assert presence_only_missing_arguments(BOOK, {"member_id": "M-1001"}) == [
        "package_id",
        "preferred_date",
    ]


def test_fault_selection() -> None:
    assert missing_args_check(None) is find_missing_arguments
    assert missing_args_check("presence_only_arg_check") is presence_only_missing_arguments
    with pytest.raises(ValueError, match="Unknown fault"):
        missing_args_check("nope")
