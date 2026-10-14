"""Dataset schema validation: invalid records are reported, never skipped."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from helpers import DATASET_PATH, INVALID_DATASET_PATH, PROJECT_ROOT
from tracelab.config import AdapterConfig, AgentConfig, Settings
from tracelab.datasets.loader import DatasetValidationError, load_dataset, validate_dataset
from tracelab.datasets.schema import SCHEMA_VERSION, schema_json
from tracelab.demo_app.target import CareDeskTarget


@pytest.fixture
def target(settings: Settings) -> CareDeskTarget:
    return CareDeskTarget(AdapterConfig(), AgentConfig(), settings)


def test_committed_dataset_is_valid(target: CareDeskTarget) -> None:
    dataset = load_dataset(DATASET_PATH, validators=[target.validate_case])
    assert len(dataset.cases) == 20
    assert dataset.schema_versions == [SCHEMA_VERSION]
    assert dataset.sha256 == hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()


def test_hard_cases_are_still_in_the_dataset() -> None:
    """Guards against a dataset edit quietly deleting the cases that catch real bugs."""
    dataset = load_dataset(DATASET_PATH)
    ids = {case.case_id for case in dataset.cases}
    assert {
        "book-missing-date-001",
        "appt-lookup-missing-id-001",
        "book-missing-member-001",
        "medical-refusal-001",
        "tool-failure-lookup-001",
        "model-malformed-json-001",
        "model-hallucinated-tool-001",
        "model-timeout-001",
    } <= ids
    tags = {tag for case in dataset.cases for tag in case.tags}
    assert {"missing-info", "tool-failure", "model-failure", "safety", "regression"} <= tags
    assert any(case.severity == "critical" for case in dataset.cases)


EXPECTED_INVALID = [
    (1, "case_id", "Field required"),
    (2, "", "malformed JSON"),
    (3, "schema_version", "unsupported schema_version '2.0'"),
    (4, "expected.forbiden_tools", "Extra inputs are not permitted"),
    (5, "severity", "Input should be"),
    (7, "case_id", "duplicate case_id (first defined on line 6)"),
    (8, "case_id", "String should match pattern"),
    (9, "expected", "both required and forbidden"),
    (10, "", "record must be a JSON object"),
    (11, "", "expected.action 'book' is not one of"),
    (12, "", "input.message: Field required"),
]


def test_every_invalid_example_is_detected_with_its_line(target: CareDeskTarget) -> None:
    cases, issues, _ = validate_dataset(INVALID_DATASET_PATH, [target.validate_case])
    found = [(i.line, i.location, i.message) for i in issues]
    for line, location, fragment in EXPECTED_INVALID:
        assert any(
            line == f_line and location == f_loc and fragment in f_msg
            for f_line, f_loc, f_msg in found
        ), f"line {line}: expected {location!r} / {fragment!r}; got {found}"
    # Only line 6 (the first definition of the duplicated id) is a valid record.
    assert [c.case_id for c in cases] == ["dup-case-001"]


def test_load_dataset_refuses_a_partially_invalid_file(target: CareDeskTarget) -> None:
    with pytest.raises(DatasetValidationError) as excinfo:
        load_dataset(INVALID_DATASET_PATH, validators=[target.validate_case])
    assert len(excinfo.value.issues) >= len(EXPECTED_INVALID)
    assert "line 2: malformed JSON" in str(excinfo.value)


def test_schema_only_validation_misses_application_rules() -> None:
    """Application checks (known actions, input shape) need the target's validator."""
    _, issues, _ = validate_dataset(INVALID_DATASET_PATH)
    assert not any("expected.action 'book'" in issue.message for issue in issues)


def test_blank_lines_are_ignored_but_an_empty_file_is_an_error(tmp_path: Path) -> None:
    record = DATASET_PATH.read_text(encoding="utf-8").splitlines()[0]
    padded = tmp_path / "padded.jsonl"
    padded.write_text(f"\n{record}\n\n", encoding="utf-8")
    assert len(load_dataset(padded).cases) == 1

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="no records"):
        load_dataset(empty)


def test_non_utf8_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "latin1.jsonl"
    path.write_bytes('{"description": "café"}'.encode("latin-1"))
    with pytest.raises(DatasetValidationError, match="not valid UTF-8"):
        load_dataset(path)


def test_hash_changes_when_a_case_changes(tmp_path: Path) -> None:
    lines = DATASET_PATH.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["description"] = "Edited description to change the dataset version."
    edited = tmp_path / "edited.jsonl"
    edited.write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n", encoding="utf-8")
    assert load_dataset(edited).sha256 != load_dataset(DATASET_PATH).sha256


def test_committed_json_schema_matches_the_model() -> None:
    committed = (PROJECT_ROOT / "datasets" / "schema" / "eval-case.v1.schema.json").read_text(
        encoding="utf-8"
    )
    assert committed == schema_json(), (
        "run: tracelab schema --output datasets/schema/eval-case.v1.schema.json"
    )
