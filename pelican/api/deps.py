"""FastAPI dependency helpers for stage-1 data access."""

from __future__ import annotations

from fastapi import Request

from pelican.data.store import DataStore
from pelican.utils.config import Settings, get_settings


def get_app_settings() -> Settings:
    """Return cached application settings."""
    return get_settings()


def get_store(request: Request) -> DataStore:
    """Return the application-scoped DuckDB store."""
    return request.app.state.store
