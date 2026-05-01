"""
EDGAR sentiment factor.

EDGAR_SENTIMENT — YoY tone shift in 10-K/10-Q MD&A sections as scored by an LLM.
A positive tone_delta indicates improving management outlook, which predicts
positive forward returns (Tetlock et al. 2007; Loughran & McDonald 2011).

Requires edgar_sentiment table to be seeded via scripts/seed_edgar.py.
"""

from __future__ import annotations

import polars as pl

from pelican.backtest.signals import SignalSpec, register


@register(SignalSpec(
    name="EDGAR_SENTIMENT",
    description=(
        "YoY change in LLM-scored MD&A tone from 10-K/10-Q filings. "
        "Positive delta = improving management outlook. "
        "Based on Loughran & McDonald (2011) textual analysis of SEC filings."
    ),
    lookback_days=400,
    requires_edgar=True,
    edgar_data_deps=("tone_delta",),
    expected_ic_range=(0.01, 0.05),
    data_frequency="quarterly",
    # Alternative-data signals are only seeded for a subset of the universe.
    # Set coverage floor to 0.5% so the engine runs even with partial seeding
    # (e.g. 5 tickers out of 500).  IC validity still requires >= 20 tickers
    # for meaningful statistics — seed more tickers for production use.
    min_score_coverage=0.005,
))
def _edgar_sentiment(cs: pl.DataFrame) -> pl.Series:
    return cs["tone_delta"].alias("EDGAR_SENTIMENT")
