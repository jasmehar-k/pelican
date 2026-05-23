"""Factor tearsheet API router."""

from __future__ import annotations

import asyncio
from datetime import date

from fastapi import APIRouter, HTTPException, Request

from pelican.api.models import SignalTearsheet
from pelican.api.services import build_tearsheet
from pelican.backtest.signals import get_signal

router = APIRouter(prefix="/factors", tags=["factors"])


@router.get("/{signal_name}/tearsheet")
async def get_tearsheet(
    request: Request,
    signal_name: str,
    start: date | None = None,
    end: date | None = None,
) -> SignalTearsheet:
    """Return full tearsheet data for a signal."""
    try:
        get_signal(signal_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    settings = request.app.state.settings
    store = request.app.state.store
    loop = asyncio.get_event_loop()
    payload = await loop.run_in_executor(
        None,
        lambda: build_tearsheet(settings, store, signal_name, start, end),
    )
    return SignalTearsheet.model_validate(payload)
