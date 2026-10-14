"""SQLite persistence for runs, traces, spans, case results and evaluations.

Design notes:
* stdlib `sqlite3`, one short-lived connection per operation. Connections are
  not shared between threads, so methods are safe to call via
  `asyncio.to_thread` from async code.
* Every method is synchronous and blocking. Async callers must not call them
  directly on the event loop (the runner and API use `asyncio.to_thread`, and
  read-only API endpoints are plain `def` so FastAPI runs them in a threadpool).
* WAL journal mode lets the dashboard read while a run is writing.
* Inputs, outputs, error texts and evaluation details are passed through the
  trace sanitizer before they are written (credentials redacted, long text cut).
* `PRAGMA user_version` stores the schema version; opening a database created
  by a newer schema fails loudly instead of misreading it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tracelab.evaluation.types import CaseOutcome
from tracelab.tracing.sanitize import redact_text, sanitize, truncate
from tracelab.tracing.tracer import utc_now

SCHEMA_VERSION = 1
# Longer than the 500-char span limit: stored outputs are shown in full in the dashboard.
PERSISTED_MAX_STRING = 4000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    config_name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    target_json TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    dataset_schema_versions TEXT NOT NULL,
    case_count INTEGER NOT NULL,
    summary_json TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(run_id),
    case_id TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES traces(trace_id),
    parent_span_id TEXT,
    seq INTEGER NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms REAL,
    status TEXT NOT NULL,
    attributes_json TEXT NOT NULL,
    error_json TEXT,
    token_usage_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id, seq);

CREATE TABLE IF NOT EXISTS case_results (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    case_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    trace_id TEXT REFERENCES traces(trace_id),
    status TEXT NOT NULL,
    severity TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    description TEXT NOT NULL,
    input_json TEXT NOT NULL,
    expected_json TEXT NOT NULL,
    output_json TEXT,
    latency_ms REAL,
    harness_error TEXT,
    skip_reason TEXT,
    PRIMARY KEY (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    evaluator_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    passed INTEGER,
    score REAL,
    max_score REAL,
    explanation TEXT NOT NULL,
    details_json TEXT NOT NULL,
    FOREIGN KEY (run_id, case_id) REFERENCES case_results(run_id, case_id)
);
CREATE INDEX IF NOT EXISTS idx_evaluations_case ON evaluations(run_id, case_id);
"""


class StoreError(Exception):
    pass


def _clean(value: Any) -> str:
    """Serialize user/model-derived data after redacting credentials (see tracing.sanitize)."""
    return _dumps(sanitize(value, max_string=PERSISTED_MAX_STRING))


def _clean_text(text: str | None) -> str | None:
    return None if text is None else truncate(redact_text(text), PERSISTED_MAX_STRING)


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


class Store:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # sqlite3's own context manager commits/rolls back but does NOT close;
        # closing explicitly matters on Windows, where open handles lock the file.
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise StoreError(
                    f"{self.path} uses schema version {version}, newer than supported "
                    f"version {SCHEMA_VERSION}"
                )
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ----------------------------------------------------------------- writes

    def create_run(
        self,
        *,
        run_id: str,
        config_name: str,
        config: dict[str, Any],
        target: dict[str, Any],
        dataset_path: str,
        dataset_sha256: str,
        dataset_schema_versions: list[str],
        case_count: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, created_at, status, config_name, config_json, "
                "target_json, dataset_path, dataset_sha256, dataset_schema_versions, case_count) "
                "VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    utc_now(),
                    config_name,
                    _dumps(config),
                    _dumps(target),
                    dataset_path,
                    dataset_sha256,
                    _dumps(dataset_schema_versions),
                    case_count,
                ),
            )

    def finish_run(
        self, run_id: str, *, status: str, summary: dict[str, Any] | None, error: str | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, summary_json = ?, error = ? "
                "WHERE run_id = ?",
                (
                    status,
                    utc_now(),
                    _dumps(summary) if summary is not None else None,
                    _clean_text(error),
                    run_id,
                ),
            )

    def save_trace(
        self,
        *,
        trace_id: str,
        source: str,
        input: dict[str, Any],
        output: dict[str, Any] | None,
        spans: list[dict[str, Any]],
        error: str | None = None,
        run_id: str | None = None,
        case_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            self._insert_trace(conn, trace_id, source, input, output, spans, error, run_id, case_id)

    def save_case_outcome(self, run_id: str, outcome: CaseOutcome) -> None:
        """Persist a case result, its trace and its evaluations in one transaction."""
        case = outcome.case
        with self._connect() as conn:
            if outcome.trace_id is not None:
                self._insert_trace(
                    conn,
                    outcome.trace_id,
                    "eval",
                    case.input,
                    outcome.output,
                    outcome.spans,
                    outcome.harness_error,
                    run_id,
                    case.case_id,
                )
            conn.execute(
                "INSERT INTO case_results (run_id, case_id, position, trace_id, status, severity, "
                "tags_json, description, input_json, expected_json, output_json, latency_ms, "
                "harness_error, skip_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    case.case_id,
                    outcome.position,
                    outcome.trace_id,
                    outcome.status,
                    case.severity,
                    _dumps(case.tags),
                    case.description,
                    _clean(case.input),
                    _dumps(case.expected.model_dump(mode="json")),
                    _clean(outcome.output) if outcome.output is not None else None,
                    outcome.latency_ms,
                    _clean_text(outcome.harness_error),
                    outcome.skip_reason,
                ),
            )
            conn.executemany(
                "INSERT INTO evaluations (run_id, case_id, evaluator_id, kind, passed, score, "
                "max_score, explanation, details_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        case.case_id,
                        r.evaluator_id,
                        r.kind,
                        None if r.passed is None else int(r.passed),
                        r.score,
                        r.max_score,
                        _clean_text(r.explanation),
                        _clean(r.details),
                    )
                    for r in outcome.results
                ],
            )

    @staticmethod
    def _insert_trace(
        conn: sqlite3.Connection,
        trace_id: str,
        source: str,
        input: dict[str, Any],
        output: dict[str, Any] | None,
        spans: list[dict[str, Any]],
        error: str | None,
        run_id: str | None,
        case_id: str | None,
    ) -> None:
        conn.execute(
            "INSERT INTO traces (trace_id, run_id, case_id, source, created_at, input_json, "
            "output_json, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace_id,
                run_id,
                case_id,
                source,
                utc_now(),
                _clean(input),
                _clean(output) if output is not None else None,
                _clean_text(error),
            ),
        )
        conn.executemany(
            "INSERT INTO spans (span_id, trace_id, parent_span_id, seq, name, kind, started_at, "
            "ended_at, duration_ms, status, attributes_json, error_json, token_usage_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    s["span_id"],
                    trace_id,
                    s["parent_span_id"],
                    s["seq"],
                    s["name"],
                    s["kind"],
                    s["started_at"],
                    s["ended_at"],
                    s["duration_ms"],
                    s["status"],
                    _dumps(s["attributes"]),
                    _dumps(s["error"]) if s["error"] is not None else None,
                    _dumps(s["token_usage"]) if s["token_usage"] is not None else None,
                )
                for s in spans
            ],
        )

    # ------------------------------------------------------------------ reads

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_run_row(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _run_row(row) if row else None

    def list_case_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM case_results WHERE run_id = ? ORDER BY position", (run_id,)
            ).fetchall()
            evals = conn.execute(
                "SELECT * FROM evaluations WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in evals:
            by_case.setdefault(row["case_id"], []).append(_evaluation_row(row))
        return [_case_row(row) | {"evaluations": by_case.get(row["case_id"], [])} for row in rows]

    def get_case_result(self, run_id: str, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM case_results WHERE run_id = ? AND case_id = ?", (run_id, case_id)
            ).fetchone()
            if row is None:
                return None
            evals = conn.execute(
                "SELECT * FROM evaluations WHERE run_id = ? AND case_id = ? ORDER BY id",
                (run_id, case_id),
            ).fetchall()
        return _case_row(row) | {"evaluations": [_evaluation_row(e) for e in evals]}

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
            if row is None:
                return None
            spans = conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY seq", (trace_id,)
            ).fetchall()
        return {
            "trace_id": row["trace_id"],
            "run_id": row["run_id"],
            "case_id": row["case_id"],
            "source": row["source"],
            "created_at": row["created_at"],
            "input": _loads(row["input_json"]),
            "output": _loads(row["output_json"]),
            "error": row["error"],
            "spans": [_span_row(s) for s in spans],
        }

    def list_traces(self, source: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT trace_id, run_id, case_id, source, created_at, output_json FROM traces"
        params: tuple[Any, ...] = ()
        if source is not None:
            query += " WHERE source = ?"
            params = (source,)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(query, (*params, limit)).fetchall()
        return [
            {
                "trace_id": r["trace_id"],
                "run_id": r["run_id"],
                "case_id": r["case_id"],
                "source": r["source"],
                "created_at": r["created_at"],
                "output": _loads(r["output_json"]),
            }
            for r in rows
        ]


def _run_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "created_at": row["created_at"],
        "finished_at": row["finished_at"],
        "status": row["status"],
        "config_name": row["config_name"],
        "config": _loads(row["config_json"]),
        "target": _loads(row["target_json"]),
        "dataset_path": row["dataset_path"],
        "dataset_sha256": row["dataset_sha256"],
        "dataset_schema_versions": _loads(row["dataset_schema_versions"]),
        "case_count": row["case_count"],
        "summary": _loads(row["summary_json"]),
        "error": row["error"],
    }


def _case_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "case_id": row["case_id"],
        "position": row["position"],
        "trace_id": row["trace_id"],
        "status": row["status"],
        "severity": row["severity"],
        "tags": _loads(row["tags_json"]),
        "description": row["description"],
        "input": _loads(row["input_json"]),
        "expected": _loads(row["expected_json"]),
        "output": _loads(row["output_json"]),
        "latency_ms": row["latency_ms"],
        "harness_error": row["harness_error"],
        "skip_reason": row["skip_reason"],
    }


def _evaluation_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "evaluator_id": row["evaluator_id"],
        "kind": row["kind"],
        "passed": None if row["passed"] is None else bool(row["passed"]),
        "score": row["score"],
        "max_score": row["max_score"],
        "explanation": row["explanation"],
        "details": _loads(row["details_json"]),
    }


def _span_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "span_id": row["span_id"],
        "trace_id": row["trace_id"],
        "parent_span_id": row["parent_span_id"],
        "seq": row["seq"],
        "name": row["name"],
        "kind": row["kind"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "duration_ms": row["duration_ms"],
        "status": row["status"],
        "attributes": _loads(row["attributes_json"]),
        "error": _loads(row["error_json"]),
        "token_usage": _loads(row["token_usage_json"]),
    }
