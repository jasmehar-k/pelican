"""
Low-volatility factor.

LOW_VOL — Ang et al. (2006) negative realized volatility.
"""

from __future__ import annotations

import polars as pl

from pelican.backtest.signals import SignalSpec, register


@register(SignalSpec(
    name="LOW_VOL",
    description="Negative 63-day realized volatility (Ang et al. 2006). "
                "Low-volatility stocks earn positive risk-adjusted returns. "
                "Negated so lower vol = higher score (more bullish).",
    lookback_days=126,
    requires_fundamentals=False,
    expected_ic_range=(0.01, 0.04),
    data_frequency="monthly",
))
def _low_vol(cs: pl.DataFrame) -> pl.Series:
    return (-cs["vol_63d"]).alias("LOW_VOL")
