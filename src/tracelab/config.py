"""Configuration: environment settings and versioned run configuration files.

* `Settings` comes from environment variables (where things live, which local
  Ollama to use). It never contains secrets because TraceLab needs none.
* `RunConfig` comes from a JSON file under `configs/`. Its full contents are
  stored with every run, so a change in prompt version, model or injected fault
  can always be traced back to the run that used it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tracelab.adapters.ollama import DEFAULT_OLLAMA_URL

DEFAULT_OLLAMA_MODEL = "llama3"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_root: Path
    db_path: Path
    configs_dir: Path
    datasets_dir: Path
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        root = Path(env.get("TRACELAB_PROJECT_ROOT", ".")).resolve()

        def path(var: str, default: str) -> Path:
            value = Path(env.get(var, default))
            return value if value.is_absolute() else root / value

        return cls(
            project_root=root,
            db_path=path("TRACELAB_DB_PATH", ".tracelab/tracelab.db"),
            configs_dir=path("TRACELAB_CONFIGS_DIR", "configs"),
            datasets_dir=path("TRACELAB_DATASETS_DIR", "datasets"),
            ollama_base_url=env.get("TRACELAB_OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
            ollama_model=env.get("TRACELAB_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        )


class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "mock"
    # None means "use the environment default" (TRACELAB_OLLAMA_MODEL for Ollama).
    model: str | None = None
    base_url: str | None = None
    timeout_s: float = Field(default=30.0, gt=0, le=600)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_version: str = "v1"
    # Total attempts per model call. Only timeouts are retried.
    max_model_attempts: int = Field(default=2, ge=1, le=5)
    # Name of a controlled fault from tracelab.demo_app.faults. Never set in
    # committed baseline configs; used to demonstrate the regression workflow.
    inject_fault: str | None = None


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    description: str = ""
    target: str = "caredesk"
    dataset: str
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    concurrency: int = Field(default=4, ge=1, le=32)
    case_timeout_s: float = Field(default=120.0, gt=0)
    disabled_evaluators: list[str] = Field(default_factory=list)


class ConfigError(Exception):
    """A run configuration file is missing or invalid."""


def load_run_config(path: Path) -> RunConfig:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid JSON: {exc}") from exc
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Config file {path} is invalid:\n{exc}") from exc


def resolve_config_path(settings: Settings, name_or_path: str) -> Path:
    """Accept either a config name ("baseline") or an explicit path."""
    candidate = Path(name_or_path)
    if candidate.suffix == ".json" or len(candidate.parts) > 1:
        return candidate if candidate.is_absolute() else settings.project_root / candidate
    return settings.configs_dir / f"{name_or_path}.json"
