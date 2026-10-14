"""Builds model adapters from configuration.

Extension point: to add a provider, implement the `ModelAdapter` protocol and
register a builder in `_BUILDERS`. Application and evaluation code need no changes.
"""

from __future__ import annotations

from collections.abc import Callable

from tracelab.adapters.base import ModelAdapter
from tracelab.adapters.mock import MockAdapter, MockFault, MockScript
from tracelab.adapters.ollama import OllamaAdapter
from tracelab.config import AdapterConfig, Settings

AdapterBuilder = Callable[
    [AdapterConfig, Settings, MockScript | None, "MockFault | None"], ModelAdapter
]


def _build_mock(
    config: AdapterConfig, settings: Settings, script: MockScript | None, fault: MockFault | None
) -> ModelAdapter:
    if script is None:
        raise ValueError("The mock provider needs a MockScript describing its behaviour")
    return MockAdapter(script, model=config.model or "mock-model", fault=fault)


def _build_ollama(
    config: AdapterConfig, settings: Settings, script: MockScript | None, fault: MockFault | None
) -> ModelAdapter:
    return OllamaAdapter(
        model=config.model or settings.ollama_model,
        base_url=config.base_url or settings.ollama_base_url,
        timeout_s=config.timeout_s,
    )


_BUILDERS: dict[str, AdapterBuilder] = {
    "mock": _build_mock,
    "ollama": _build_ollama,
}


def available_providers() -> list[str]:
    return sorted(_BUILDERS)


def create_adapter(
    config: AdapterConfig,
    settings: Settings,
    *,
    mock_script: MockScript | None = None,
    mock_fault: MockFault | None = None,
) -> ModelAdapter:
    """Create the adapter named by `config.provider`.

    `mock_script` and `mock_fault` only affect the mock provider; real providers
    cannot be told to fail on demand, so callers must skip fault-injection cases
    for them (see CareDeskTarget.skip_reason).
    """
    try:
        builder = _BUILDERS[config.provider]
    except KeyError:
        raise ValueError(
            f"Unknown model provider {config.provider!r}; available: {available_providers()}"
        ) from None
    return builder(config, settings, mock_script, mock_fault)
