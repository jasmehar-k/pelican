"""Portfolio construction API router."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from pelican.api.models import (
    PortfolioBacktestResponse,
    PortfolioOptimizeRequest,
    PortfolioOptimizeResponse,
)
from pelican.api.services import optimize_portfolio, run_portfolio_backtest

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/optimize")
async def optimize_route(request: Request, payload: PortfolioOptimizeRequest) -> PortfolioOptimizeResponse:
    """Optimize a long/short portfolio for the selected signals."""
    settings = request.app.state.settings
    store = request.app.state.store
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: optimize_portfolio(settings, store, payload))
        return PortfolioOptimizeResponse.model_validate(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/backtest")
async def backtest_route(request: Request, payload: PortfolioOptimizeRequest) -> PortfolioBacktestResponse:
    """Walk-forward IC-weighted backtest for the selected signal combination."""
    settings = request.app.state.settings
    store = request.app.state.store
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: run_portfolio_backtest(settings, store, payload))
        return PortfolioBacktestResponse.model_validate(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
