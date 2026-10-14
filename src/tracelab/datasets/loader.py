"""Load and validate JSONL evaluation datasets.

Validation is all-or-nothing: every problem in the file is collected with its
line number, and if there is at least one, loading fails. Invalid rows are
never skipped silently, because a dataset that quietly loses its hardest cases
produces misleadingly good scores.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from tracelab.datasets.schema import EvalCase

# Extra, application-specific checks (e.g. "this tool name exists"). Each returns
# a list of error messages for the case.
CaseValidator = Callable[[EvalCase], list[str]]


@dataclass(frozen=True)
class ValidationIssue:
    line: int
    case_id: str | None
    location: str
    message: str

    def __str__(self) -> str:
        case = f" [{self.case_id}]" if self.case_id else ""
        where = f" {self.location}:" if self.location else ""
        return f"line {self.line}{case}:{where} {self.message}"


class DatasetValidationError(Exception):
    def __init__(self, path: Path, issues: Sequence[ValidationIssue]) -> None:
        self.path = path
        self.issues = list(issues)
        details = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(f"{path}: {len(self.issues)} validation issue(s)\n{details}")


@dataclass(frozen=True)
class Dataset:
    path: Path
    cases: list[EvalCase]
    sha256: str  # hash of the exact file bytes; identifies the dataset version used by a run

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def schema_versions(self) -> list[str]:
        return sorted({case.schema_version for case in self.cases})


def validate_dataset(
    path: Path, validators: Sequence[CaseValidator] = ()
) -> tuple[list[EvalCase], list[ValidationIssue], str]:
    """Return (valid cases, issues, sha256). Does not raise for invalid records."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [], [ValidationIssue(0, None, "", f"cannot read file: {exc.strerror or exc}")], ""
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [], [ValidationIssue(0, None, "", f"file is not valid UTF-8: {exc}")], digest

    cases: list[EvalCase] = []
    issues: list[ValidationIssue] = []
    seen: dict[str, int] = {}

    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue  # blank lines are formatting, not records
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue(line_no, None, "", f"malformed JSON: {exc.msg}"))
            continue
        if not isinstance(record, dict):
            issues.append(ValidationIssue(line_no, None, "", "record must be a JSON object"))
            continue
        case_id = record.get("case_id") if isinstance(record.get("case_id"), str) else None
        try:
            case = EvalCase.model_validate(record)
        except ValidationError as exc:
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"])
                issues.append(ValidationIssue(line_no, case_id, location, error["msg"]))
            continue

        if case.case_id in seen:
            issues.append(
                ValidationIssue(
                    line_no,
                    case.case_id,
                    "case_id",
                    f"duplicate case_id (first defined on line {seen[case.case_id]})",
                )
            )
            continue
        seen[case.case_id] = line_no

        extra = [message for validator in validators for message in validator(case)]
        if extra:
            issues.extend(ValidationIssue(line_no, case.case_id, "", m) for m in extra)
            continue
        cases.append(case)

    if not cases and not issues:
        issues.append(ValidationIssue(0, None, "", "dataset contains no records"))
    return cases, issues, digest


def load_dataset(path: Path, validators: Sequence[CaseValidator] = ()) -> Dataset:
    """Load a dataset, raising DatasetValidationError if any record is invalid."""
    path = Path(path)
    cases, issues, digest = validate_dataset(path, validators)
    if issues:
        raise DatasetValidationError(path, issues)
    return Dataset(path=path, cases=cases, sha256=digest)
