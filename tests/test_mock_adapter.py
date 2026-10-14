"""The mock adapter: deterministic output and controlled failure modes."""

from __future__ import annotations

import asyncio
import json

import pytest

from tracelab.adapters.base import Message, ModelRequest, ModelUnavailableError
from tracelab.adapters.mock import MOCK_FAULTS, MockAdapter, MockScript
from tracelab.demo_app.mock_script import CAREDESK_MOCK_SCRIPT, plan

pytestmark = pytest.mark.anyio


def plan_request(message: str, context: dict[str, str] | None = None) -> ModelRequest:
    payload = json.dumps({"message": message, "context": context or {}})
    return ModelRequest(
        messages=[Message("system", "plan"), Message("user", payload)],
        purpose="plan",
        response_format="json",
    )


async def test_same_request_gives_same_output() -> None:
    adapter = MockAdapter(CAREDESK_MOCK_SCRIPT)
    request = plan_request("When is my appointment APT-1001?")
    first = await adapter.generate(request)
    second = await adapter.generate(request)
    assert first.text == second.text
    assert json.loads(first.text)["tool_calls"][0]["arguments"] == {"appointment_id": "APT-1001"}


async def test_mock_never_reports_token_usage_and_is_labelled_simulated() -> None:
    response = await MockAdapter(CAREDESK_MOCK_SCRIPT).generate(plan_request("What packages?"))
    assert response.token_usage is None
    assert response.metadata["simulated"] is True
    assert response.provider == "mock"


def test_mock_plan_emits_null_for_missing_arguments_like_a_real_model() -> None:
    result = plan("Please book the Heart Health Check", {"member_id": "M-1001"})
    assert result["tool_calls"] == [
        {
            "name": "book_appointment",
            "arguments": {"member_id": "M-1001", "package_id": "PKG-HEART", "preferred_date": None},
        }
    ]


async def test_malformed_json_fault_returns_invalid_json() -> None:
    response = await MockAdapter(CAREDESK_MOCK_SCRIPT, fault="malformed_json").generate(
        plan_request("What packages do you offer?")
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(response.text)


async def test_fenced_json_fault_wraps_valid_json_in_a_code_fence() -> None:
    response = await MockAdapter(CAREDESK_MOCK_SCRIPT, fault="fenced_json").generate(
        plan_request("What packages do you offer?")
    )
    assert response.text.startswith("```json\n") and response.text.endswith("\n```")


@pytest.mark.parametrize("fault", ["wrong_schema", "hallucinated_tool"])
async def test_content_faults_use_script_payloads(fault: str) -> None:
    response = await MockAdapter(CAREDESK_MOCK_SCRIPT, fault=fault).generate(plan_request("hi"))
    assert response.text == CAREDESK_MOCK_SCRIPT.fault_outputs[fault]


async def test_unavailable_fault_raises() -> None:
    with pytest.raises(ModelUnavailableError):
        await MockAdapter(CAREDESK_MOCK_SCRIPT, fault="unavailable").generate(plan_request("hi"))


async def test_timeout_fault_hangs_until_the_caller_times_out() -> None:
    adapter = MockAdapter(CAREDESK_MOCK_SCRIPT, fault="timeout")
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(adapter.generate(plan_request("What packages?")), timeout=0.05)


async def test_timeout_once_fault_hangs_only_on_the_first_call() -> None:
    adapter = MockAdapter(CAREDESK_MOCK_SCRIPT, fault="timeout_once")
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(adapter.generate(plan_request("What packages?")), timeout=0.05)
    response = await asyncio.wait_for(adapter.generate(plan_request("What packages?")), timeout=1)
    assert json.loads(response.text)["intent"] == "list_packages"


async def test_faults_only_apply_to_configured_purposes() -> None:
    adapter = MockAdapter(CAREDESK_MOCK_SCRIPT, fault="unavailable")  # default: plan only
    compose = ModelRequest(
        messages=[Message("user", json.dumps({"message": "x", "tool_results": []}))],
        purpose="compose",
    )
    response = await adapter.generate(compose)
    assert "I can help with CareDesk" in response.text


def test_unknown_fault_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown mock fault"):
        MockAdapter(CAREDESK_MOCK_SCRIPT, fault="explode")  # type: ignore[arg-type]


def test_content_fault_requires_a_script_payload() -> None:
    script = MockScript(responder=lambda request: "{}")
    with pytest.raises(ValueError, match="does not define an output"):
        MockAdapter(script, fault="wrong_schema")


def test_every_fault_is_documented_in_the_type() -> None:
    assert set(MOCK_FAULTS) == {
        "malformed_json",
        "fenced_json",
        "wrong_schema",
        "hallucinated_tool",
        "empty_response",
        "timeout",
        "timeout_once",
        "unavailable",
    }
