"""
News sentiment factor.

NEWS_SENTIMENT — average LLM-scored headline sentiment for recent news articles
as fetched via yfinance. A positive avg_score indicates bullish news flow,
which predicts positive forward returns (Tetlock 2007; Loughran & McDonald 2011).

Requires news_sentiment table to be seeded via scripts/seed_news.py.
"""

from __future__ import annotations

import polars as pl

from pelican.backtest.signals import SignalSpec, register


@register(SignalSpec(
    name="NEWS_SENTIMENT",
    description=(
        "Average LLM-scored sentiment of recent news headlines in [-1, +1]. "
        "Positive score = bullish news flow. "
        "Based on Tetlock (2007) media pessimism and return predictability."
    ),
    lookback_days=30,
    requires_news=True,
    expected_ic_range=(0.005, 0.04),
    data_frequency="daily",
    # Alternative-data signal: only seeded for a subset of the universe.
    # Set coverage floor to 0.5% so the engine runs even with partial seeding.
    min_score_coverage=0.005,
))
def _news_sentiment(cs: pl.DataFrame) -> pl.Series:
    return cs["avg_score"].alias("NEWS_SENTIMENT")
