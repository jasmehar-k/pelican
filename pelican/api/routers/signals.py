"""Signals API router."""

from __future__ import annotations

from datetime import date
from typing import Any

import pelican.factors  # noqa: F401 - register factor signals
from fastapi import APIRouter, HTTPException, Request

from pelican.api.models import SignalSummary
from pelican.api.services import signal_names, signal_summary_payload
from pelican.backtest.signals import get_signal

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("")
async def list_signals(request: Request, start: date | None = None, end: date | None = None) -> list[SignalSummary]:
    """List all registered signals with their metadata and backtest stats."""
    settings = request.app.state.settings
    store = request.app.state.store
    return [
        SignalSummary.model_validate(signal_summary_payload(settings, store, name, start, end))
        for name in signal_names()
    ]


@router.get("/{signal_name}")
async def get_signal_summary(request: Request, signal_name: str, start: date | None = None, end: date | None = None) -> SignalSummary:
    """Return one signal's metadata and backtest stats."""
    try:
        get_signal(signal_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    settings = request.app.state.settings
    store = request.app.state.store
    return SignalSummary.model_validate(signal_summary_payload(settings, store, signal_name, start, end))
