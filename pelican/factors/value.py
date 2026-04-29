"""
Value factors.

VALUE_PE  — Earnings yield (inverse P/E).
VALUE_PB  — Book yield (inverse P/B).
"""

from __future__ import annotations

import polars as pl

from pelican.backtest.signals import SignalSpec, register


@register(SignalSpec(
    name="VALUE_PE",
    description="Earnings yield = 1 / P/E ratio (Basu 1977). "
                "High earnings yield (low P/E) stocks expected to outperform. "
                "Null for negative or zero earnings.",
    lookback_days=126,
    requires_fundamentals=True,
    data_deps=("pe_ratio",),
    expected_ic_range=(0.01, 0.05),
    data_frequency="quarterly",
))
def _value_pe(cs: pl.DataFrame) -> pl.Series:
    return cs.select(
        pl.when(pl.col("pe_ratio") > 0)
        .then(1.0 / pl.col("pe_ratio"))
        .otherwise(None)
        .alias("VALUE_PE")
    )["VALUE_PE"]


@register(SignalSpec(
    name="VALUE_PB",
    description="Book yield = 1 / P/B ratio (Fama & French 1992). "
                "High book yield (low P/B) stocks expected to outperform. "
                "Null for negative or zero book value.",
    lookback_days=126,
    requires_fundamentals=True,
    data_deps=("pb_ratio",),
    expected_ic_range=(0.01, 0.04),
    data_frequency="quarterly",
))
def _value_pb(cs: pl.DataFrame) -> pl.Series:
    return cs.select(
        pl.when(pl.col("pb_ratio") > 0)
        .then(1.0 / pl.col("pb_ratio"))
        .otherwise(None)
        .alias("VALUE_PB")
    )["VALUE_PB"]
