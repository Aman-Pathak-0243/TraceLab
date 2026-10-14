"""SQLite persistence details."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tracelab.storage.store import SCHEMA_VERSION, Store, StoreError
from tracelab.tracing.tracer import Tracer


def test_schema_version_is_recorded(tmp_path: Path) -> None:
    store = Store(tmp_path / "a.db")
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    Store(store.path)  # reopening is fine


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(StoreError, match="newer than supported"):
        Store(path)


def test_trace_round_trip_preserves_order_and_nesting(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    tracer = Tracer()
    with tracer.span("root", "agent", message="hi"), tracer.span("child", "tool", tool_name="x"):
        pass
    store.save_trace(
        trace_id=tracer.trace_id,
        source="api",
        input={"message": "hi"},
        output={"action": "answer"},
        spans=[s.to_dict() for s in tracer.spans],
    )
    trace = store.get_trace(tracer.trace_id)
    assert trace is not None
    assert [s["name"] for s in trace["spans"]] == ["root", "child"]
    assert trace["spans"][1]["parent_span_id"] == trace["spans"][0]["span_id"]
    assert trace["spans"][0]["attributes"] == {"message": "hi"}
    assert store.list_traces(source="api")[0]["trace_id"] == tracer.trace_id
    assert store.list_traces(source="eval") == []


def test_missing_records_return_none(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    assert store.get_run("nope") is None
    assert store.get_case_result("nope", "x") is None
    assert store.get_trace("nope") is None
    assert store.list_runs() == []


def test_failed_write_is_rolled_back(tmp_path: Path) -> None:
    store = Store(tmp_path / "r.db")
    tracer = Tracer()
    with tracer.span("root", "agent"):
        pass
    spans = [s.to_dict() for s in tracer.spans]
    store.save_trace(trace_id="tr_1", source="api", input={}, output=None, spans=spans)
    # Same span ids again -> primary-key violation -> the whole transaction rolls back.
    with pytest.raises(sqlite3.IntegrityError):
        store.save_trace(trace_id="tr_2", source="api", input={}, output=None, spans=spans)
    assert store.get_trace("tr_2") is None
