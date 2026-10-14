"""FastAPI dependencies (injected with `Depends`), so tests can supply their own."""

from __future__ import annotations

from fastapi import Request

from tracelab.config import Settings
from tracelab.storage.store import Store


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> Store:
    return request.app.state.store
