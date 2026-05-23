"""Signals API router."""

from __future__ import annotations

import asyncio
from datetime import date

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
    names = signal_names()
    loop = asyncio.get_event_loop()
    payloads = await loop.run_in_executor(
        None,
        lambda: [signal_summary_payload(settings, store, n, start, end) for n in names],
    )
    return [SignalSummary.model_validate(p) for p in payloads]


@router.get("/{signal_name}")
async def get_signal_summary(request: Request, signal_name: str, start: date | None = None, end: date | None = None) -> SignalSummary:
    """Return one signal's metadata and backtest stats."""
    try:
        get_signal(signal_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    settings = request.app.state.settings
    store = request.app.state.store
    loop = asyncio.get_event_loop()
    payload = await loop.run_in_executor(
        None,
        lambda: signal_summary_payload(settings, store, signal_name, start, end),
    )
    return SignalSummary.model_validate(payload)
