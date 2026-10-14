"""Application services shared by the CLI and the HTTP API.

Wires a `RunConfig` to concrete objects (target, judge, dataset) so that the
CLI and the API run evaluations in exactly the same way.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

from tracelab.adapters.factory import create_adapter
from tracelab.config import ConfigError, RunConfig, Settings
from tracelab.datasets.loader import Dataset, load_dataset
from tracelab.demo_app.target import CareDeskTarget
from tracelab.evaluation.deterministic import unknown_evaluator_names
from tracelab.evaluation.judge import MOCK_JUDGE_SCRIPT, LLMJudgeEvaluator
from tracelab.evaluation.runner import RunReport, run_suite
from tracelab.evaluation.types import CaseOutcome, Target
from tracelab.storage.store import Store


def build_target(config: RunConfig, settings: Settings) -> Target:
    if config.target == "caredesk":
        return CareDeskTarget(config.adapter, config.agent, settings)
    raise ValueError(f"Unknown target {config.target!r}; available: ['caredesk']")


def build_judge(config: RunConfig, settings: Settings) -> LLMJudgeEvaluator | None:
    if not config.judge.enabled:
        return None
    adapter = create_adapter(config.judge.adapter, settings, mock_script=MOCK_JUDGE_SCRIPT)
    return LLMJudgeEvaluator(adapter, timeout_s=config.judge.adapter.timeout_s)


def resolve_dataset_path(config: RunConfig, settings: Settings) -> Path:
    path = Path(config.dataset)
    return path if path.is_absolute() else settings.project_root / path


def load_dataset_for(config: RunConfig, settings: Settings, target: Target) -> Dataset:
    path = resolve_dataset_path(config, settings)
    if not path.is_file():
        raise ConfigError(f"Config {config.name!r} points to a missing dataset file: {path}")
    return load_dataset(path, validators=[target.validate_case])


async def run_evaluation(
    config: RunConfig,
    settings: Settings,
    store: Store,
    *,
    case_ids: Sequence[str] | None = None,
    on_case_done: Callable[[CaseOutcome], None] | None = None,
) -> RunReport:
    """Validate everything up front, then run. Raises before creating a run if
    the config, target or dataset is invalid."""
    target = build_target(config, settings)  # also registers the target's business rules
    unknown = unknown_evaluator_names(config.disabled_evaluators)
    if unknown:
        raise ConfigError(f"disabled_evaluators contains unknown names: {unknown}")
    # File I/O and validation are synchronous; keep them off the event loop.
    dataset = await asyncio.to_thread(load_dataset_for, config, settings, target)
    judge = build_judge(config, settings)
    return await run_suite(
        config=config,
        target=target,
        dataset=dataset,
        store=store,
        judge=judge,
        case_ids=case_ids,
        on_case_done=on_case_done,
    )
