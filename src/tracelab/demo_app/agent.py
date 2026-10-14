"""CareDesk agent: the instrumented application under test.

Execution path for one request (every step is a traced span):

    agent.handle
      policy.screen_input      deterministic medical-advice guard
      plan                     model proposes tool calls as structured JSON
        llm.generate             one span per attempt (timeouts are retried)
        parse.plan               JSON + schema validation of the model output
      validate.tool_call       tool exists, required arguments present, no extras
      tool.<name>              tool execution
      compose                  model writes the reply from tool results
        llm.generate
        parse.answer

Every outcome is returned as a validated `AgentResponse`. Expected failures
(model down, invalid output, tool errors) become `action="error"` responses
with a stable error code; unexpected exceptions propagate to the caller.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import ValidationError

from tracelab.adapters.base import (
    Message,
    ModelAdapter,
    ModelError,
    ModelRequest,
    ModelResponse,
    ModelTimeoutError,
)
from tracelab.demo_app.faults import missing_args_check
from tracelab.demo_app.guards import (
    find_mistyped_arguments,
    find_unexpected_arguments,
    screen_input,
)
from tracelab.demo_app.prompts import compose_prompt, plan_prompt
from tracelab.demo_app.schemas import AgentRequest, AgentResponse, ErrorInfo, Plan, plan_json_schema
from tracelab.demo_app.tools import ToolError, ToolRegistry, ToolSpec
from tracelab.tracing.tracer import Tracer


class InvalidModelOutputError(Exception):
    code = "invalid_model_output"


REFUSAL_ANSWER = (
    "I can't give medical advice. Please contact a qualified healthcare professional, or "
    "emergency services if it is urgent. I can help with CareDesk packages, appointments "
    "and policies."
)

_FIELD_PROMPTS = {
    "member_id": "your member ID (please sign in)",
    "package_id": "which package you would like",
    "preferred_date": "your preferred date (YYYY-MM-DD)",
    "appointment_id": "your appointment ID (for example APT-1001)",
    "topic": "which policy you are asking about",
}

_ERROR_ANSWERS = {
    "model_timeout": "Sorry, I could not process your request in time. Please try again later.",
    "model_unavailable": "Sorry, the assistant is unavailable right now. Please try again later.",
    "model_response_error": "Sorry, the assistant is unavailable right now. Please try again later.",
    "invalid_model_output": "Sorry, something went wrong while handling your request. Please try again.",
    "unknown_tool": "Sorry, I can't help with that request.",
    "invalid_tool_arguments": "Sorry, something went wrong while handling your request. Please try again.",
    "tool_unavailable": "That system is temporarily unavailable. Please try again later.",
    "tool_internal_error": "That system is temporarily unavailable. Please try again later.",
    "not_found": "I couldn't find that appointment. Please check the ID (for example APT-1001).",
}

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_plan(text: str) -> Plan:
    """Parse and validate the model's plan. Tolerates a markdown code fence."""
    body = text.strip()
    fence = _FENCE.match(body)
    if fence:
        body = fence.group(1)
    if not body:
        raise InvalidModelOutputError("model returned an empty plan")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise InvalidModelOutputError(
            f"plan is not valid JSON: {exc.msg} (char {exc.pos})"
        ) from exc
    try:
        return Plan.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()[:3]
        )
        raise InvalidModelOutputError(f"plan does not match the schema: {problems}") from exc


class CareDeskAgent:
    def __init__(
        self,
        adapter: ModelAdapter,
        tools: ToolRegistry | None = None,
        *,
        prompt_version: str = "v1",
        model_timeout_s: float = 30.0,
        max_model_attempts: int = 2,
        inject_fault: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.tools = tools or ToolRegistry()
        self.prompt_version = prompt_version
        self.model_timeout_s = model_timeout_s
        self.max_model_attempts = max_model_attempts
        self.inject_fault = inject_fault
        self._find_missing = missing_args_check(inject_fault)
        plan_prompt(prompt_version, self.tools.specs)  # fail fast on an unknown prompt version

    async def handle(self, request: AgentRequest, tracer: Tracer) -> AgentResponse:
        with tracer.span(
            "agent.handle",
            "agent",
            message=request.message,
            context_keys=sorted(request.context),
            provider=self.adapter.provider,
            model=self.adapter.model,
            prompt_version=self.prompt_version,
        ) as root:
            response = await self._handle(request, tracer)
            root.set(
                action=response.action,
                error_code=response.error.code if response.error else None,
                missing_fields=response.missing_fields,
                tools_used=response.tools_used,
            )
            return response

    async def _handle(self, request: AgentRequest, tracer: Tracer) -> AgentResponse:
        with tracer.span("policy.screen_input", "policy") as span:
            matched = screen_input(request.message)
            span.set(outcome="blocked" if matched else "allowed", matched=matched)
        if matched:
            return AgentResponse(action="refuse", answer=REFUSAL_ANSWER)

        try:
            plan = await self._plan(request, tracer)
        except (ModelError, InvalidModelOutputError) as exc:
            return _error_response(exc.code, str(exc))

        tool_results: list[dict[str, Any]] = []
        data: dict[str, Any] = {}
        for call in plan.tool_calls:
            with tracer.span(
                "validate.tool_call", "validation", tool=call.name, arguments=call.arguments
            ) as span:
                spec = self.tools.get(call.name)
                if spec is None:
                    span.set(outcome="unknown_tool")
                    return _error_response(
                        "unknown_tool",
                        f"model requested unknown tool {call.name!r}",
                        data,
                        tool_results,
                    )
                missing = self._find_missing(spec, call.arguments)
                unexpected = find_unexpected_arguments(spec, call.arguments)
                mistyped = find_mistyped_arguments(spec, call.arguments)
                outcome = "valid"
                if unexpected:
                    outcome = "unexpected_arguments"
                elif mistyped:
                    outcome = "mistyped_arguments"
                elif missing:
                    outcome = "missing_arguments"
                span.set(outcome=outcome, missing=missing, unexpected=unexpected, mistyped=mistyped)
            if unexpected or mistyped:
                problem = (
                    f"unexpected arguments {unexpected}"
                    if unexpected
                    else f"arguments must be strings: {mistyped}"
                )
                return _error_response(
                    "invalid_tool_arguments",
                    f"{spec.name}: {problem}",
                    data,
                    tool_results,
                )
            if missing:
                return _request_info(missing, data, tool_results)

            try:
                result = self._run_tool(spec, call.arguments, tracer)
            except ToolError as exc:
                return _error_response(exc.code, str(exc), data, tool_results)
            tool_results.append({"tool": spec.name, "result": result})
            data.update(result)

        tools_used = [r["tool"] for r in tool_results]
        try:
            answer = await self._compose(request, plan, tool_results, tracer)
        except (ModelError, InvalidModelOutputError) as exc:
            if not tool_results:
                return _error_response(exc.code, str(exc))
            # The tools already ran (possibly a booking). Losing that because the
            # model failed to phrase a reply would be worse than a plain answer.
            return AgentResponse(
                action="answer",
                answer="Your request was processed. Details: "
                + json.dumps(data, sort_keys=True)[:400],
                data=data,
                tools_used=tools_used,
                warnings=[f"compose_failed:{exc.code}"],
            )
        return AgentResponse(action="answer", answer=answer, data=data, tools_used=tools_used)

    async def _plan(self, request: AgentRequest, tracer: Tracer) -> Plan:
        with tracer.span("plan", "step", prompt_version=self.prompt_version):
            model_request = ModelRequest(
                messages=[
                    Message("system", plan_prompt(self.prompt_version, self.tools.specs)),
                    Message(
                        "user", json.dumps({"message": request.message, "context": request.context})
                    ),
                ],
                purpose="plan",
                response_format="json",
                json_schema=plan_json_schema(),
            )
            response = await self._generate(model_request, tracer)
            with tracer.span("parse.plan", "validation", raw_output=response.text) as span:
                plan = parse_plan(response.text)
                span.set(intent=plan.intent, tool_calls=[c.model_dump() for c in plan.tool_calls])
            return plan

    def _run_tool(
        self, spec: ToolSpec, arguments: dict[str, Any], tracer: Tracer
    ) -> dict[str, Any]:
        with tracer.span(
            f"tool.{spec.name}",
            "tool",
            tool_name=spec.name,
            arguments=arguments,
            idempotent=spec.idempotent,
        ) as span:
            result = self.tools.execute(spec.name, arguments)
            span.set(output=result)
            return result

    async def _compose(
        self,
        request: AgentRequest,
        plan: Plan,
        tool_results: list[dict[str, Any]],
        tracer: Tracer,
    ) -> str:
        with tracer.span("compose", "step"):
            model_request = ModelRequest(
                messages=[
                    Message("system", compose_prompt(self.prompt_version)),
                    Message(
                        "user",
                        json.dumps(
                            {
                                "message": request.message,
                                "intent": plan.intent,
                                "tool_results": tool_results,
                            }
                        ),
                    ),
                ],
                purpose="compose",
            )
            response = await self._generate(model_request, tracer)
            with tracer.span("parse.answer", "validation") as span:
                answer = response.text.strip()
                if not answer:
                    raise InvalidModelOutputError("model returned an empty answer")
                span.set(answer=answer)
            return answer

    async def _generate(self, request: ModelRequest, tracer: Tracer) -> ModelResponse:
        """Call the model with a timeout. Only timeouts are retried: a refused
        connection or a missing model will not fix itself on an immediate retry."""
        for attempt in range(1, self.max_model_attempts + 1):
            try:
                with tracer.span(
                    "llm.generate",
                    "model",
                    purpose=request.purpose,
                    provider=self.adapter.provider,
                    model=self.adapter.model,
                    attempt=attempt,
                    max_attempts=self.max_model_attempts,
                    timeout_s=self.model_timeout_s,
                ) as span:
                    try:
                        response = await asyncio.wait_for(
                            self.adapter.generate(request), timeout=self.model_timeout_s
                        )
                    except TimeoutError as exc:
                        raise ModelTimeoutError(
                            f"no model response within {self.model_timeout_s}s"
                        ) from exc
                    span.set(output=response.text, response_model=response.model)
                    span.record_usage(response.token_usage)
                return response
            except ModelTimeoutError:
                if attempt == self.max_model_attempts:
                    raise
        raise AssertionError("unreachable: the loop either returns or raises")


def _error_response(
    code: str,
    message: str,
    data: dict[str, Any] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
) -> AgentResponse:
    return AgentResponse(
        action="error",
        answer=_ERROR_ANSWERS.get(code, f"Sorry, I couldn't complete that request: {message}"),
        error=ErrorInfo(code=code, message=message),
        data=data or {},
        tools_used=[r["tool"] for r in tool_results or []],
    )


def _request_info(
    missing: list[str], data: dict[str, Any], tool_results: list[dict[str, Any]]
) -> AgentResponse:
    needed = ", ".join(_FIELD_PROMPTS.get(name, name) for name in missing)
    return AgentResponse(
        action="request_info",
        answer=f"To continue, I need {needed}.",
        missing_fields=missing,
        data=data,
        tools_used=[r["tool"] for r in tool_results],
    )
