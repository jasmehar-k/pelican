"""
Size factor.

SIZE — Fama & French (1992) small-minus-big (SMB).
Smaller market cap = higher expected return (size premium).
"""

from __future__ import annotations

import math

import polars as pl

from pelican.backtest.signals import SignalSpec, register


@register(SignalSpec(
    name="SIZE",
    description="Negative log market cap (Fama & French 1992 size factor). "
                "Smaller firms expected to outperform. "
                "Negated so smaller cap = higher score (more bullish).",
    lookback_days=126,
    requires_fundamentals=True,
    data_deps=("market_cap",),
    expected_ic_range=(-0.02, 0.02),
    data_frequency="quarterly",
))
def _size(cs: pl.DataFrame) -> pl.Series:
    return cs.select(
        pl.when(pl.col("market_cap") > 0)
        .then(-pl.col("market_cap").log(base=math.e))
        .otherwise(None)
        .alias("SIZE")
    )["SIZE"]
