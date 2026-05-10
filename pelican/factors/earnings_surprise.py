"""
Post-Earnings Announcement Drift (PEAD) signal.

EARNINGS_SURPRISE — most recent quarterly EPS surprise percentage as reported
by yfinance.  Stocks that beat consensus estimates outperform for several months
after the announcement; stocks that miss continue to underperform.

Reference: Ball & Brown (1968), Bernard & Thomas (1989), Foster, Olsen & Shevlin
(1984).  PEAD is among the most replicated anomalies in empirical finance.

Requires earnings_surprises table to be seeded via scripts/seed_earnings.py.
Seeding is fast (~1–2 min for the full S&P 500) and requires no API key or LLM.
"""

from __future__ import annotations

import polars as pl

from pelican.backtest.signals import SignalSpec, register


@register(SignalSpec(
    name="EARNINGS_SURPRISE",
    description=(
        "Most recent quarterly EPS surprise (%) — (reported - estimate) / |estimate| × 100. "
        "Positive surprise predicts positive forward returns via post-earnings drift. "
        "Ball & Brown (1968); Bernard & Thomas (1989)."
    ),
    lookback_days=400,
    requires_earnings=True,
    expected_ic_range=(0.02, 0.08),
    data_frequency="quarterly",
    # Alternative-data signal: only seeded for a subset of the universe by default.
    # Set coverage floor low so the engine runs even with partial seeding.
    min_score_coverage=0.10,
))
def _earnings_surprise(cs: pl.DataFrame) -> pl.Series:
    return cs["surprise_pct"].alias("EARNINGS_SURPRISE")
