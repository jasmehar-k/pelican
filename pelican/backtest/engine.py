"""
Vectorized cross-sectional backtesting engine (monthly rebalance, long/short).

On each monthly rebalance date:
  1. Queries the point-in-time S&P 500 universe from DuckDB.
  2. Joins signal scores for that date.
  3. Cross-sectionally ranks and z-scores signals within the universe.
  4. Forms quintile portfolios: long top quintile (Q5), short bottom (Q1).
  5. Holds for 21 trading days, then measures equal-weighted forward returns.

All operations are vectorized Polars expressions over a (date × ticker) panel.
Returns a BacktestResult with: per-period L/S spread returns, IC time series,
and the full quintile breakdown for tearsheet generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import polars as pl

from pelican.backtest.metrics import (
    compute_ic_stats,
    compute_max_drawdown,
    compute_sharpe,
    spearman_ic,
)
from pelican.backtest.signals import (
    SignalDef,
    build_cross_section_features,
    get_signal,
)
from pelican.backtest.universe import get_point_in_time_universe, get_rebalance_dates
from pelican.data.store import DataStore
from pelican.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestConfig:
    start: date
    end: date
    cost_bps: float = 5.0
    min_universe_size: int = 50
    min_score_coverage: float = 0.5
    lookback_calendar_days: int = 800
    quintile_n: int = 5


@dataclass
class BacktestResult:
    signal_name: str
    config: BacktestConfig
    # date, q1..q5, ls_gross, ls_net, universe_size, n_scored
    period_returns: pl.DataFrame
    # date, ic
    ic_series: pl.DataFrame
    ic_mean: float
    icir: float
    ic_tstat: float
    sharpe_gross: float
    sharpe_net: float
    max_drawdown_gross: float
    max_drawdown_net: float
    avg_turnover: float
    n_periods: int
    avg_universe_size: float


def run_backtest(
    signal_name: str,
    config: BacktestConfig,
    store: DataStore,
) -> BacktestResult:
    """Run a full cross-sectional backtest for `signal_name` over `config`."""
    sig: SignalDef = get_signal(signal_name)

    rebal_dates = get_rebalance_dates(config.start, config.end, store)
    if not rebal_dates:
        raise ValueError(f"No rebalance dates found between {config.start} and {config.end}")

    # Pull the full price panel once — wide enough to support all lags.
    # We need `lookback_calendar_days` before the first rebal date.
    from datetime import timedelta
    panel_start = rebal_dates[0] - timedelta(days=config.lookback_calendar_days)

    panel = store.query(
        """
        SELECT p.ticker, p.date, p.open, p.high, p.low, p.close,
               p.volume, p.log_return_1d, p.forward_return_21d
        FROM prices p
        INNER JOIN sp500_universe u
            ON p.ticker = u.ticker
           AND p.date >= u.entry_date
           AND (u.exit_date IS NULL OR p.date < u.exit_date)
        WHERE p.date >= ? AND p.date <= ?
        ORDER BY p.ticker, p.date
        """,
        [panel_start, config.end],
    )

    if panel.is_empty():
        raise ValueError("No price data found for the backtest period.")

    period_rows: list[dict] = []
    ic_rows: list[dict] = []
    prev_longs: set[str] = set()
    prev_shorts: set[str] = set()
    turnovers: list[float] = []

    for rebal_date in rebal_dates:
        universe = get_point_in_time_universe(rebal_date, store)
        if len(universe) < config.min_universe_size:
            log.warning("universe too small, skipping", date=rebal_date, size=len(universe))
            continue

        # Build cross-section features for this rebalance date.
        ticker_panel = panel.filter(pl.col("ticker").is_in(universe))
        cs = build_cross_section_features(ticker_panel, rebal_date)

        if cs.is_empty():
            continue

        # Compute signal scores.
        try:
            scores: pl.Series = sig.fn(cs)
        except Exception as exc:
            log.warning("signal compute failed", date=rebal_date, error=str(exc))
            continue

        cs = cs.with_columns(scores.alias("score"))

        # Require minimum coverage before proceeding.
        n_scored = cs["score"].drop_nulls().len()
        if n_scored < config.min_score_coverage * len(universe):
            log.warning("insufficient score coverage, skipping", date=rebal_date, n_scored=n_scored)
            continue

        # Cross-sectional rank → quintile (1=bottom, 5=top).
        cs = cs.with_columns(
            pl.col("score")
            .rank(method="average", descending=False)
            .alias("rank")
        )
        labels = [str(i) for i in range(1, config.quintile_n + 1)]
        cs = cs.with_columns(
            pl.col("rank")
            .qcut(config.quintile_n, labels=labels, allow_duplicates=True)
            .alias("quintile")
        )

        # Per-quintile equal-weighted forward returns.
        # Use None (→ Polars null) instead of float("nan") so that drop_nulls()
        # correctly removes missing periods in aggregate metric functions.
        quintile_returns: dict[str, float | None] = {}
        for q in range(1, config.quintile_n + 1):
            q_df = cs.filter(pl.col("quintile") == str(q))
            fwd = q_df["forward_return_21d"].fill_nan(None).drop_nulls()
            mean_fwd = float(fwd.mean()) if len(fwd) > 0 else None  # type: ignore[arg-type]
            quintile_returns[f"q{q}"] = mean_fwd

        q1_ret = quintile_returns.get("q1")
        q5_ret = quintile_returns.get(f"q{config.quintile_n}")
        ls_gross: float | None = (
            q5_ret - q1_ret  # type: ignore[operator]
            if (q1_ret is not None and q5_ret is not None)
            else None
        )

        # Turnover — average of long (Q5) and short (Q1) book turnover.
        curr_longs: set[str] = set(
            cs.filter(pl.col("quintile") == str(config.quintile_n))["ticker"].to_list()
        )
        curr_shorts: set[str] = set(
            cs.filter(pl.col("quintile") == "1")["ticker"].to_list()
        )
        long_to = _turnover(prev_longs, curr_longs)
        short_to = _turnover(prev_shorts, curr_shorts)
        turnover = (long_to + short_to) / 2.0
        turnovers.append(turnover)
        prev_longs = curr_longs
        prev_shorts = curr_shorts

        # Net return after transaction costs on both legs.
        cost = turnover * config.cost_bps / 10_000
        ls_net: float | None = ls_gross - cost if ls_gross is not None else None

        # IC for this period — store None rather than float("nan") so that
        # compute_ic_stats / compute_sharpe's drop_nulls() removes bad periods.
        ic_raw = spearman_ic(cs["score"], cs["forward_return_21d"])
        ic: float | None = ic_raw if not math.isnan(ic_raw) else None

        period_rows.append({
            "date": rebal_date,
            **quintile_returns,
            "ls_gross": ls_gross,
            "ls_net": ls_net,
            "universe_size": len(universe),
            "n_scored": n_scored,
            "turnover": turnover,
        })
        ic_rows.append({"date": rebal_date, "ic": ic})

    if not period_rows:
        raise ValueError("Backtest produced no periods — check data coverage and date range.")

    period_df = pl.DataFrame(period_rows)
    ic_df = pl.DataFrame(ic_rows)

    ic_stats = compute_ic_stats(ic_df["ic"])
    sharpe_gross = compute_sharpe(period_df["ls_gross"])
    sharpe_net = compute_sharpe(period_df["ls_net"])
    max_dd_gross = compute_max_drawdown(period_df["ls_gross"])
    max_dd_net = compute_max_drawdown(period_df["ls_net"])
    avg_turnover = float(period_df["turnover"].mean()) if len(period_df) > 0 else float("nan")

    return BacktestResult(
        signal_name=signal_name,
        config=config,
        period_returns=period_df,
        ic_series=ic_df,
        ic_mean=ic_stats["ic_mean"],
        icir=ic_stats["icir"],
        ic_tstat=ic_stats["ic_tstat"],
        sharpe_gross=sharpe_gross,
        sharpe_net=sharpe_net,
        max_drawdown_gross=max_dd_gross,
        max_drawdown_net=max_dd_net,
        avg_turnover=avg_turnover,
        n_periods=len(period_df),
        avg_universe_size=float(period_df["universe_size"].mean()),
    )


def _turnover(prev: set[str], curr: set[str]) -> float:
    """Fraction of the portfolio that changed between two rebalance periods.

    Defined as (|entered| + |exited|) / (|prev| + |curr|).
    Returns 1.0 for the first period (no prior portfolio).
    """
    if not prev:
        return 1.0
    entered = len(curr - prev)
    exited = len(prev - curr)
    denom = len(prev) + len(curr)
    return (entered + exited) / denom if denom > 0 else 0.0
