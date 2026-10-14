"""Shared fixtures. Tests never need a network, an API key or Ollama."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from helpers import BASELINE_CONFIG_PATH, INJECTED_FAULT, PROJECT_ROOT
from tracelab.config import RunConfig, Settings, load_run_config
from tracelab.storage.store import Store


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("TRACELAB_OLLAMA_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live Ollama test; set TRACELAB_OLLAMA_TESTS=1 to run")
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's TRACELAB_* settings from leaking into tests."""
    for key in list(os.environ):
        if key.startswith("TRACELAB_"):
            monkeypatch.delenv(key)


@pytest.fixture
def injected_fault() -> str | None:
    return INJECTED_FAULT


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=PROJECT_ROOT,
        db_path=tmp_path / "tracelab.db",
        configs_dir=PROJECT_ROOT / "configs",
        datasets_dir=PROJECT_ROOT / "datasets",
    )


@pytest.fixture
def store(settings: Settings) -> Store:
    return Store(settings.db_path)


@pytest.fixture
def baseline_config() -> RunConfig:
    return load_run_config(BASELINE_CONFIG_PATH)
