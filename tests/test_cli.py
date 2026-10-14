"""Command-line interface: commands, output and exit codes."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import INVALID_DATASET_PATH, PROJECT_ROOT
from tracelab.cli import main


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "cli.db"
    monkeypatch.setenv("TRACELAB_PROJECT_ROOT", str(PROJECT_ROOT))
    monkeypatch.setenv("TRACELAB_DB_PATH", str(db))
    return db


def test_validate_exit_codes(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate"]) == 0
    assert "20 cases" in capsys.readouterr().out
    assert main(["validate", str(INVALID_DATASET_PATH)]) == 1
    out = capsys.readouterr().out
    assert "INVALID" in out and "line 2: malformed JSON" in out


def test_run_show_trace_and_compare(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["runs"]) == 0
    assert "No runs yet" in capsys.readouterr().out

    assert main(["run", "--no-judge"]) == 0
    out = capsys.readouterr().out
    assert "20 passed, 0 failed" in out and "Judge: disabled" in out

    assert main(["run", "--no-judge", "--inject-fault", "presence_only_arg_check"]) == 1
    out = capsys.readouterr().out
    assert "17 passed, 3 failed" in out
    assert "FAIL    book-missing-date-001" in out

    assert main(["show", "latest", "--failed"]) == 0
    out = capsys.readouterr().out
    assert "x expected_action: expected action 'request_info', got 'answer'" in out
    assert "packages-list-001" not in out

    assert main(["trace", "latest", "book-missing-date-001"]) == 0
    out = capsys.readouterr().out
    assert "validate.tool_call (validation)" in out
    assert '"outcome": "valid"' in out
    assert "tool.book_appointment (tool)" in out

    assert main(["compare", "previous", "latest"]) == 1  # regression -> non-zero exit
    assert "Newly failed (REGRESSIONS): 3" in capsys.readouterr().out
    assert main(["compare", "latest", "previous"]) == 0
    assert "Newly passed (fixed): 3" in capsys.readouterr().out

    assert main(["runs"]) == 0
    assert "presence_only_arg_check" in capsys.readouterr().out


def test_ask_saves_a_trace(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "When is my appointment APT-1001?", "--context", "member_id=M-1001"]) == 0
    out = capsys.readouterr().out
    assert '"action": "answer"' in out
    trace_id = out.split("Trace ")[1].split()[0]
    assert main(["trace", "--trace-id", trace_id, "--no-attrs"]) == 0
    assert "tool.lookup_appointment (tool)" in capsys.readouterr().out


def test_usage_errors_exit_2(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", "does-not-exist"]) == 2
    assert "Config file not found" in capsys.readouterr().err
    assert main(["show", "latest"]) == 2
    assert main(["run", "--case", "nope", "--no-judge"]) == 2
    assert main(["ask", "hi", "--context", "novalue"]) == 2
    assert main(["trace"]) == 2


def test_demo_regression_walkthrough(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo-regression"]) == 0
    out = capsys.readouterr().out
    assert out.count("[ok]") == 5
    assert "[FAILED]" not in out
    assert "Diagnosis: validate.tool_call" in out


def test_schema_and_faults_commands(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema"]) == 0
    assert '"title": "EvalCase"' in capsys.readouterr().out
    assert main(["faults"]) == 0
    assert "presence_only_arg_check" in capsys.readouterr().out
