"""Constants and helpers shared by test modules (imported as `helpers`)."""

from __future__ import annotations

import os
from pathlib import Path

from tracelab.config import RunConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "caredesk-v1.jsonl"
INVALID_DATASET_PATH = PROJECT_ROOT / "datasets" / "examples" / "invalid-records.jsonl"
BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.json"

# Read at import time, before the autouse `_isolated_env` fixture strips TRACELAB_*
# variables. Used only by the regression-demo tests (test_regression_missing_info.py).
INJECTED_FAULT = os.environ.get("TRACELAB_INJECT_FAULT") or None


def with_fault(config: RunConfig, fault: str | None) -> RunConfig:
    return config.model_copy(
        update={"agent": config.agent.model_copy(update={"inject_fault": fault})}
    )
