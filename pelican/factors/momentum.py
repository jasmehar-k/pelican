"""
Momentum and short-term reversal factors.

MOM_1_12  — Jegadeesh & Titman (1993) 12-1 month momentum.
REVERSAL_1M — De Bondt & Thaler (1985) 1-month reversal.
"""

from __future__ import annotations

import polars as pl

from pelican.backtest.signals import SignalSpec, register


@register(SignalSpec(
    name="MOM_1_12",
    description="12-1 month momentum (Jegadeesh & Titman 1993). "
                "Long past winners, short past losers. "
                "Skips the most recent month to avoid short-term reversal.",
    lookback_days=504,
    requires_fundamentals=False,
    expected_ic_range=(0.01, 0.06),
    data_frequency="monthly",
))
def _mom_1_12(cs: pl.DataFrame) -> pl.Series:
    # close_252d = price ~12 months ago; close_21d = price ~1 month ago.
    # Return over months 2–12 (skipping most recent month).
    return (cs["close_21d"] / cs["close_252d"] - 1.0).alias("MOM_1_12")


@register(SignalSpec(
    name="REVERSAL_1M",
    description="Negative 1-month return (short-term reversal). "
                "Recent losers expected to mean-revert. "
                "Negated so higher score = more bullish.",
    lookback_days=63,
    requires_fundamentals=False,
    expected_ic_range=(-0.04, -0.01),
    data_frequency="monthly",
))
def _reversal_1m(cs: pl.DataFrame) -> pl.Series:
    # Negate: past losers (negative return) become high-scoring (bullish).
    return (-(cs["close"] / cs["close_21d"] - 1.0)).alias("REVERSAL_1M")
