"""Portfolio construction API router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from pelican.api.models import PortfolioOptimizeRequest, PortfolioOptimizeResponse
from pelican.api.services import optimize_portfolio

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/optimize")
async def optimize_route(request: Request, payload: PortfolioOptimizeRequest) -> PortfolioOptimizeResponse:
    """Optimize a long/short portfolio for the selected signals."""
    settings = request.app.state.settings
    store = request.app.state.store
    try:
        return PortfolioOptimizeResponse.model_validate(optimize_portfolio(settings, store, payload))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
