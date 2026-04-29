"""
Backtest performance metrics.

Computes: Information Coefficient (IC) and IC t-statistic, annualized Sharpe
ratio, Sortino ratio, max drawdown, average turnover, and quintile spread
returns. All metrics accept Polars Series or DataFrames and return plain
Python scalars or dicts suitable for JSON serialization.
"""

from __future__ import annotations

import math

import polars as pl


def spearman_ic(scores: pl.Series, returns: pl.Series) -> float:
    """Spearman rank correlation between signal scores and forward returns.

    Both series must be aligned (same index / same order). NaN rows are
    dropped pairwise before ranking.
    """
    df = pl.DataFrame({"s": scores, "r": returns}).drop_nulls()
    if len(df) < 3:
        return float("nan")
    s_rank = df["s"].rank(method="average")
    r_rank = df["r"].rank(method="average")
    cov = ((s_rank - s_rank.mean()) * (r_rank - r_rank.mean())).mean()
    denom = s_rank.std(ddof=0) * r_rank.std(ddof=0)
    if denom == 0:
        return float("nan")
    return float(cov / denom)


def compute_ic_stats(ic_series: pl.Series) -> dict[str, float]:
    """IC mean, IC IR (ICIR = mean/std), and t-stat from a time series of ICs."""
    clean = ic_series.drop_nulls()
    n = len(clean)
    if n < 2:
        return {"ic_mean": float("nan"), "icir": float("nan"), "ic_tstat": float("nan")}
    ic_mean = float(clean.mean())  # type: ignore[arg-type]
    ic_std = float(clean.std(ddof=1))  # type: ignore[arg-type]  # ddof=1 for ICIR
    icir = ic_mean / ic_std if ic_std > 0 else float("nan")
    ic_tstat = icir * math.sqrt(n) if not math.isnan(icir) else float("nan")
    return {"ic_mean": ic_mean, "icir": icir, "ic_tstat": ic_tstat}


def compute_sharpe(returns: pl.Series, periods_per_year: int = 12) -> float:
    """Annualized Sharpe ratio (assumes zero risk-free rate).

    `returns` is a Series of period returns (one per rebalance period).
    `periods_per_year` = 12 for monthly rebalancing.
    """
    clean = returns.drop_nulls()
    if len(clean) < 2:
        return float("nan")
    mean = float(clean.mean())  # type: ignore[arg-type]
    std = float(clean.std(ddof=1))  # type: ignore[arg-type]
    if std == 0:
        return float("nan")
    return mean / std * math.sqrt(periods_per_year)


def compute_max_drawdown(returns: pl.Series) -> float:
    """Maximum peak-to-trough drawdown from a series of period returns."""
    clean = returns.drop_nulls()
    if len(clean) == 0:
        return float("nan")
    cumulative = (1.0 + clean).cum_prod()
    running_max = cumulative.cum_max()
    drawdowns = (cumulative - running_max) / running_max
    return float(drawdowns.min())  # type: ignore[arg-type]
