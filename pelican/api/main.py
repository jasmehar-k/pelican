"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

import pelican.factors  # noqa: F401 - register classic and EDGAR factors
from pelican.backtest.signals import load_dynamic_signals
from pelican.data.store import DataStore
from pelican.utils.config import get_settings
from pelican.utils.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared settings, logging, and the DuckDB store."""
    configure_logging(dev=False)
    settings = get_settings()
    store = DataStore(settings.duckdb_path)
    store.init_schema()
    load_dynamic_signals(store)

    app.state.settings = settings
    app.state.store = store
    try:
        yield
    finally:
        store.close()


def create_app() -> FastAPI:
    """Create the API application."""
    app = FastAPI(title="Pelican", version="0.1.0", lifespan=lifespan)

    from pelican.api.routers.agents import router as agents_router
    from pelican.api.routers.factors import router as factors_router
    from pelican.api.routers.portfolio import router as portfolio_router
    from pelican.api.routers.signals import router as signals_router

    app.include_router(agents_router)
    app.include_router(signals_router)
    app.include_router(factors_router)
    app.include_router(portfolio_router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
