"""
Quality factors.

QUALITY_ROE       — Return on equity (Novy-Marx 2013).
QUALITY_LEVERAGE  — Negative debt-to-equity (lower leverage = higher quality).
"""

from __future__ import annotations

import polars as pl

from pelican.backtest.signals import SignalSpec, register


def _winsorize(s: pl.Series, lo: float = 0.01, hi: float = 0.99) -> pl.Series:
    """Clip a Series at the lo-th and hi-th quantiles."""
    clean = s.drop_nulls()
    if len(clean) < 10:
        return s
    p_lo = float(clean.quantile(lo))
    p_hi = float(clean.quantile(hi))
    return s.clip(p_lo, p_hi)


@register(SignalSpec(
    name="QUALITY_ROE",
    description="Return on equity (Novy-Marx 2013 gross profitability proxy). "
                "Higher ROE firms expected to outperform. "
                "Winsorized at 1st/99th percentile to limit outlier influence.",
    lookback_days=126,
    requires_fundamentals=True,
    data_deps=("roe",),
    expected_ic_range=(0.01, 0.04),
    data_frequency="quarterly",
))
def _quality_roe(cs: pl.DataFrame) -> pl.Series:
    raw = cs["roe"]
    return _winsorize(raw).alias("QUALITY_ROE")


@register(SignalSpec(
    name="QUALITY_LEVERAGE",
    description="Negative debt-to-equity ratio. "
                "Lower leverage firms earn higher risk-adjusted returns. "
                "Null when equity is zero or negative.",
    lookback_days=126,
    requires_fundamentals=True,
    data_deps=("debt_to_equity",),
    expected_ic_range=(0.00, 0.03),
    data_frequency="quarterly",
))
def _quality_leverage(cs: pl.DataFrame) -> pl.Series:
    return (-cs["debt_to_equity"]).alias("QUALITY_LEVERAGE")
