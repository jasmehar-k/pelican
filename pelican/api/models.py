"""Pydantic request and response models for the API layer."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, Field


def _default_start() -> date:
    return date.today() - timedelta(days=3 * 365)


def _default_end() -> date:
    return date.today() - timedelta(days=1)


class AgentRunRequest(BaseModel):
    theme: str
    model: str | None = None
    with_researcher: bool = True
    start: date = Field(default_factory=_default_start)
    end: date = Field(default_factory=_default_end)


class AgentRunSummary(BaseModel):
    run_id: str
    theme: str
    decision: str | None
    ic_tstat: float | None
    sharpe_net: float | None
    retry_count: int
    ts: str


class AgentStreamEvent(BaseModel):
    event: str  # node_start | node_complete | llm_token | run_complete | run_error
    node: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class ResearchPaper(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    arxiv_id: str | None = None
    url: str | None = None


class AgentRunLineage(BaseModel):
    run_id: str
    ts: str
    theme: str
    arxiv_ids: list[str] = Field(default_factory=list)
    papers: list[ResearchPaper] = Field(default_factory=list)
    signal_hypothesis: str | None = None
    signal_name: str | None = None
    generated_code: str | None = None
    decision: str | None = None
    ic_tstat: float | None = None
    sharpe_net: float | None = None
    feedback: str | None = None
    retry_count: int = 0


class SignalBacktestStats(BaseModel):
    ic_mean: float | None = None
    icir: float | None = None
    ic_tstat: float | None = None
    sharpe_gross: float | None = None
    sharpe_net: float | None = None
    max_drawdown_gross: float | None = None
    max_drawdown_net: float | None = None
    avg_turnover: float | None = None
    n_periods: int | None = None
    avg_universe_size: float | None = None


class SignalSummary(BaseModel):
    name: str
    description: str
    lookback_days: int
    requires_fundamentals: bool
    requires_edgar: bool
    data_deps: list[str] = Field(default_factory=list)
    edgar_data_deps: list[str] = Field(default_factory=list)
    expected_ic_range: tuple[float, float]
    data_frequency: str
    min_score_coverage: float | None = None
    source: str = "library"
    stats: SignalBacktestStats | None = None
    error: str | None = None


class BacktestConfigModel(BaseModel):
    start: date
    end: date
    cost_bps: float
    min_universe_size: int
    min_score_coverage: float
    lookback_calendar_days: int
    quintile_n: int


class BacktestPeriodRow(BaseModel):
    date: date
    q1: float | None = None
    q2: float | None = None
    q3: float | None = None
    q4: float | None = None
    q5: float | None = None
    ls_gross: float | None = None
    ls_net: float | None = None
    universe_size: int | None = None
    n_scored: int | None = None
    turnover: float | None = None


class BacktestICRow(BaseModel):
    date: date
    ic: float | None = None


class SignalTearsheet(BaseModel):
    summary: SignalSummary
    config: BacktestConfigModel
    period_returns: list[BacktestPeriodRow]
    ic_series: list[BacktestICRow]


class PortfolioOptimizeRequest(BaseModel):
    signals: list[str] = Field(min_length=1)
    objective: str = "max_sharpe"
    method: str = "ic_weighted"
    rebalance_date: date | None = None
    start: date = Field(default_factory=_default_start)
    end: date = Field(default_factory=_default_end)
    cost_bps: float = 5.0
    lambda_risk: float = 1.0
    max_weight: float = 0.05
    turnover_limit: float | None = None
    min_universe_size: int = 50
    min_score_coverage: float = 0.5
    lookback_calendar_days: int = 800


class PortfolioPosition(BaseModel):
    ticker: str
    weight: float


class RiskDecompositionSummary(BaseModel):
    tickers: list[str]
    weights: list[float]
    total_variance: float
    systematic_variance: float
    idiosyncratic_variance: float
    systematic_pct: float
    idiosyncratic_pct: float
    factor_contributions: list[float]
    idio_contributions: list[float]


class PortfolioOptimizeResponse(BaseModel):
    signals: list[str]
    rebalance_date: date
    objective: str
    method: str
    status: str
    expected_return: float
    expected_variance: float
    expected_sharpe: float
    positions: list[PortfolioPosition]
    risk_decomposition: RiskDecompositionSummary | None = None
    ic_weights: dict[str, float] = Field(default_factory=dict)
    alpha_coverage: int = 0
