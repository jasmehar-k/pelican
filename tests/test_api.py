"""Integration tests for the FastAPI layer."""

from __future__ import annotations

import asyncio
from datetime import date

import httpx
import pytest

from pelican.api.main import create_app
from pelican.api.models import AgentRunRequest
from pelican.data.store import DataStore
from pelican.utils.config import get_settings


@pytest.fixture()
def app() -> object:
    app = create_app()
    store = DataStore(":memory:")
    store.init_schema()
    app.state.store = store
    app.state.settings = get_settings()
    return app


@pytest.fixture()
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.anyio("asyncio")
async def test_signals_endpoint_lists_metadata_and_stats(client, monkeypatch):
    from pelican.api.routers import signals as signals_router

    monkeypatch.setattr(signals_router, "signal_names", lambda: ["MOM_1_12", "LOW_VOL"])
    monkeypatch.setattr(
        signals_router,
        "signal_summary_payload",
        lambda settings, store, name, start=None, end=None: {
            "name": name,
            "description": f"{name} desc",
            "lookback_days": 504,
            "requires_fundamentals": False,
            "requires_edgar": False,
            "data_deps": [],
            "edgar_data_deps": [],
            "expected_ic_range": [-0.1, 0.1],
            "data_frequency": "monthly",
            "min_score_coverage": None,
            "stats": {"ic_mean": 0.01, "icir": 0.2, "ic_tstat": 1.7},
            "error": None,
        },
    )

    response = await client.get("/signals")
    assert response.status_code == 200
    payload = response.json()
    assert [item["name"] for item in payload] == ["MOM_1_12", "LOW_VOL"]
    assert payload[0]["stats"]["ic_tstat"] == 1.7


@pytest.mark.anyio("asyncio")
async def test_factor_tearsheet_endpoint(client, monkeypatch):
    from pelican.api.routers import factors as factors_router

    monkeypatch.setattr(
        factors_router,
        "build_tearsheet",
        lambda settings, store, signal_name, start=None, end=None: {
            "summary": {
                "name": signal_name,
                "description": "demo",
                "lookback_days": 252,
                "requires_fundamentals": False,
                "requires_edgar": False,
                "data_deps": [],
                "edgar_data_deps": [],
                "expected_ic_range": [-0.1, 0.1],
                "data_frequency": "monthly",
                "min_score_coverage": None,
                "stats": {"ic_mean": 0.02},
                "error": None,
            },
            "config": {
                "start": date(2024, 1, 1),
                "end": date(2024, 12, 31),
                "cost_bps": 5.0,
                "min_universe_size": 50,
                "min_score_coverage": 0.5,
                "lookback_calendar_days": 800,
                "quintile_n": 5,
            },
            "period_returns": [{"date": date(2024, 1, 31), "q1": -0.01, "q5": 0.02}],
            "ic_series": [{"date": date(2024, 1, 31), "ic": 0.12}],
        },
    )

    response = await client.get("/factors/MOM_1_12/tearsheet")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["name"] == "MOM_1_12"
    assert payload["period_returns"][0]["q5"] == 0.02


@pytest.mark.anyio("asyncio")
async def test_portfolio_optimize_endpoint(client, monkeypatch):
    from pelican.api.routers import portfolio as portfolio_router

    monkeypatch.setattr(
        portfolio_router,
        "optimize_portfolio",
        lambda settings, store, request: {
            "signals": request.signals,
            "rebalance_date": date(2024, 12, 31),
            "objective": request.objective,
            "method": request.method,
            "status": "optimal",
            "expected_return": 0.12,
            "expected_variance": 0.04,
            "expected_sharpe": 1.8,
            "positions": [{"ticker": "AAPL", "weight": 0.05}, {"ticker": "MSFT", "weight": -0.05}],
            "risk_decomposition": None,
            "ic_weights": {"MOM_1_12": 1.0},
            "alpha_coverage": 2,
        },
    )

    response = await client.post(
        "/portfolio/optimize",
        json={"signals": ["MOM_1_12"], "objective": "max_sharpe", "method": "ic_weighted"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "optimal"
    assert payload["positions"][0]["ticker"] == "AAPL"


@pytest.mark.anyio("asyncio")
async def test_agents_status_alias_404s_for_missing_run(client):
    response = await client.get("/agents/status/missing-run-id")
    assert response.status_code == 404
