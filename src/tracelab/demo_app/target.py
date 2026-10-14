"""Connects the CareDesk application to the evaluation engine (`Target` protocol)."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

import tracelab.demo_app.rules  # noqa: F401  (registers CareDesk business rules)
from tracelab.adapters.base import ModelAdapter
from tracelab.adapters.factory import create_adapter
from tracelab.config import AdapterConfig, AgentConfig, Settings
from tracelab.datasets.schema import EvalCase, Simulation
from tracelab.demo_app.agent import CareDeskAgent
from tracelab.demo_app.faults import missing_args_check
from tracelab.demo_app.mock_script import CAREDESK_MOCK_SCRIPT
from tracelab.demo_app.prompts import PROMPTS
from tracelab.demo_app.schemas import ACTIONS, AgentRequest, AgentResponse
from tracelab.demo_app.tools import ToolRegistry
from tracelab.evaluation.rules import registered_rules
from tracelab.tracing.tracer import Tracer


class CareDeskTarget:
    name = "caredesk"
    output_model = AgentResponse

    def __init__(self, adapter: AdapterConfig, agent: AgentConfig, settings: Settings) -> None:
        missing_args_check(agent.inject_fault)  # raises ValueError for an unknown fault
        if agent.prompt_version not in PROMPTS:
            raise ValueError(
                f"Unknown prompt version {agent.prompt_version!r}; available: {sorted(PROMPTS)}"
            )
        self.adapter_config = adapter
        self.agent_config = agent
        self.settings = settings
        # Real providers are shared across cases (stateless HTTP calls). The mock
        # is rebuilt per case because each case may inject a different fault.
        self._shared_adapter: ModelAdapter | None = (
            None if adapter.provider == "mock" else create_adapter(adapter, settings)
        )

    @property
    def uses_mock(self) -> bool:
        return self.adapter_config.provider == "mock"

    def describe(self) -> dict[str, Any]:
        adapter = self._shared_adapter or self.build_adapter(None)
        return {
            "target": self.name,
            "provider": adapter.provider,
            "model": adapter.model,
            "prompt_version": self.agent_config.prompt_version,
            "inject_fault": self.agent_config.inject_fault,
        }

    def validate_case(self, case: EvalCase) -> list[str]:
        errors: list[str] = []
        try:
            AgentRequest.model_validate(case.input)
        except ValidationError as exc:
            errors.extend(f"input.{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        if case.expected.action not in ACTIONS:
            errors.append(f"expected.action {case.expected.action!r} is not one of {list(ACTIONS)}")
        tool_names = set(ToolRegistry().names())
        for tool in case.expected.required_tools:
            if tool not in tool_names:
                errors.append(f"expected.required_tools: unknown tool {tool!r}")
        tool_fault = case.simulate.tool_fault if case.simulate else None
        if tool_fault is not None and tool_fault.tool not in tool_names:
            errors.append(f"simulate.tool_fault.tool: unknown tool {tool_fault.tool!r}")
        rules = set(registered_rules())
        for rule in case.expected.business_rules:
            if rule not in rules:
                errors.append(f"expected.business_rules: unknown rule {rule!r}")
        return errors

    def skip_reason(self, case: EvalCase) -> str | None:
        if case.simulate and case.simulate.model_fault and not self.uses_mock:
            return (
                f"simulate.model_fault={case.simulate.model_fault!r} needs the mock adapter; "
                f"a real model cannot be made to fail on demand"
            )
        return None

    def build_adapter(self, simulation: Simulation | None) -> ModelAdapter:
        if self._shared_adapter is not None:
            return self._shared_adapter
        return create_adapter(
            self.adapter_config,
            self.settings,
            mock_script=CAREDESK_MOCK_SCRIPT,
            mock_fault=simulation.model_fault if simulation else None,
        )

    def build_agent(self, simulation: Simulation | None = None) -> CareDeskAgent:
        return CareDeskAgent(
            self.build_adapter(simulation),
            ToolRegistry(fault=simulation.tool_fault if simulation else None),
            prompt_version=self.agent_config.prompt_version,
            model_timeout_s=self.adapter_config.timeout_s,
            max_model_attempts=self.agent_config.max_model_attempts,
            inject_fault=self.agent_config.inject_fault,
        )

    async def execute(self, case: EvalCase, tracer: Tracer) -> dict[str, Any]:
        request = AgentRequest.model_validate(case.input)
        response = await self.build_agent(case.simulate).handle(request, tracer)
        return response.model_dump(mode="json")
