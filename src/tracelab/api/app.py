"""FastAPI application factory: JSON API under /api and the HTML dashboard under /."""

from __future__ import annotations

from fastapi import FastAPI

from tracelab import __version__
from tracelab.api.dashboard import router as dashboard_router
from tracelab.api.routes import router as api_router
from tracelab.config import Settings
from tracelab.storage.store import Store


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="TraceLab",
        version=__version__,
        description="Evaluation, tracing and regression testing for LLM applications.",
    )
    app.state.settings = settings
    app.state.store = Store(settings.db_path)
    app.include_router(api_router)
    app.include_router(dashboard_router)
    return app
