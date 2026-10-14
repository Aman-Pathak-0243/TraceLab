"""Plain-text rendering of a trace for the CLI."""

from __future__ import annotations

import json
from typing import Any


def span_depths(spans: list[dict[str, Any]]) -> dict[str, int]:
    by_id = {s["span_id"]: s for s in spans}
    depths: dict[str, int] = {}

    def depth(span_id: str) -> int:
        if span_id not in depths:
            parent = by_id[span_id].get("parent_span_id")
            depths[span_id] = 0 if parent is None or parent not in by_id else depth(parent) + 1
        return depths[span_id]

    for span in spans:
        depth(span["span_id"])
    return depths


def format_trace(spans: list[dict[str, Any]], *, show_attributes: bool = True) -> str:
    ordered = sorted(spans, key=lambda s: s["seq"])
    depths = span_depths(ordered)
    lines = []
    for span in ordered:
        indent = "  " * depths[span["span_id"]]
        duration = span.get("duration_ms")
        duration_text = f"{duration:.2f} ms" if isinstance(duration, int | float) else "-"
        marker = {"ok": "ok ", "error": "ERR", "cancelled": "CXL", "running": "..."}.get(
            span["status"], span["status"]
        )
        lines.append(f"[{marker}] {indent}{span['name']} ({span['kind']}) {duration_text}")
        if show_attributes and span.get("attributes"):
            attrs = json.dumps(span["attributes"], sort_keys=True, default=str)
            lines.append(f"      {indent}attrs: {attrs}")
        if span.get("error"):
            lines.append(f"      {indent}error: {json.dumps(span['error'], default=str)}")
        if span.get("token_usage"):
            lines.append(f"      {indent}tokens: {json.dumps(span['token_usage'])}")
    return "\n".join(lines)
