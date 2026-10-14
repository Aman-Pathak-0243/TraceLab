"""Redaction and truncation applied to everything written into a trace.

Traces are persisted to disk and shown in the dashboard, so they must never
contain credentials and should not store unbounded payloads.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"
MAX_STRING_LENGTH = 500
MAX_DEPTH = 6
MAX_ITEMS = 50

# Keys whose values are always redacted, whatever they contain.
SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|secret|passw(or)?d|passphrase|authorization|private[_-]?key"
    r"|bearer|credential|cookie|session[_-]?id"
    # "token" / "pwd" only as the whole key or its last segment (access_token,
    # x-auth-token, db_pwd), so that e.g. "token_usage" or "prompt_tokens" stay visible.
    r"|(^|[_.-])(token|pwd)$)",
    re.IGNORECASE,
)
# Values that look like credentials even under an innocent key.
SENSITIVE_VALUE = re.compile(
    r"((?i:bearer)\s+[A-Za-z0-9._~+/=-]{8,}|sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,})"
)
# user:password@ inside a URL, e.g. http://user:pass@host:11434
URL_CREDENTIALS = re.compile(r"(?<=://)[^\s/@:]+:[^\s/@]+@")


def truncate(text: str, limit: int = MAX_STRING_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def redact_text(text: str) -> str:
    """Redact credential-looking substrings in free text (messages, errors, reprs)."""
    return URL_CREDENTIALS.sub(f"{REDACTED}@", SENSITIVE_VALUE.sub(REDACTED, text))


def sanitize(value: Any, *, max_string: int = MAX_STRING_LENGTH, _depth: int = 0) -> Any:
    """Return a JSON-safe copy of `value` with secrets redacted and strings truncated."""
    if _depth > MAX_DEPTH:
        return "[max depth reached]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return truncate(redact_text(value), max_string)
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_ITEMS:
                clean["..."] = f"[{len(value) - MAX_ITEMS} more keys omitted]"
                break
            key_text = str(key)
            if SENSITIVE_KEY.search(key_text):
                clean[key_text] = REDACTED
            else:
                clean[key_text] = sanitize(item, max_string=max_string, _depth=_depth + 1)
        return clean
    if isinstance(value, list | tuple | set):
        items = list(value)
        clean_items = [
            sanitize(item, max_string=max_string, _depth=_depth + 1) for item in items[:MAX_ITEMS]
        ]
        if len(items) > MAX_ITEMS:
            clean_items.append(f"[{len(items) - MAX_ITEMS} more items omitted]")
        return clean_items
    return truncate(redact_text(repr(value)), max_string)
