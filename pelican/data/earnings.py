"""
Earnings surprise data ingestion via yfinance.

Fetches historical quarterly earnings (EPS estimate vs reported) per ticker and
stores the surprise percentage in `earnings_surprises`.  No LLM is required —
yfinance returns the data directly.

Post-Earnings Announcement Drift (PEAD) is one of the most robust anomalies in
finance (Ball & Brown 1968; Bernard & Thomas 1989): stocks that beat estimates
continue to outperform, and stocks that miss continue to underperform, for
several months after the announcement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pelican.data.store import DataStore


def fetch_earnings(ticker: str) -> pl.DataFrame:
    """Fetch historical quarterly earnings surprise data from yfinance.

    Returns a DataFrame with columns: ticker, date, surprise_pct,
    reported_eps, estimated_eps.  Rows where the announcement date is missing
    or the estimate is zero (undefined surprise) are dropped.
    """
    empty = pl.DataFrame(schema={
        "ticker": pl.Utf8,
        "date": pl.Date,
        "surprise_pct": pl.Float64,
        "reported_eps": pl.Float64,
        "estimated_eps": pl.Float64,
    })
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.earnings_history
        if hist is None or hist.empty:
            return empty

        df = hist.reset_index()
        # yfinance column names vary slightly by version — normalize
        col_map = {}
        for col in df.columns:
            lc = col.lower()
            if "estimate" in lc:
                col_map[col] = "estimated_eps"
            elif "reported" in lc:
                col_map[col] = "reported_eps"
            elif "surprise" in lc and "%" in col:
                col_map[col] = "surprise_pct"
            elif col.lower() in ("earnings date", "earningsdate", "date"):
                col_map[col] = "date"
        df = df.rename(columns=col_map)

        required = {"date", "estimated_eps", "reported_eps", "surprise_pct"}
        if not required.issubset(df.columns):
            return empty

        import pandas as pd
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
        df = df[["ticker", "date", "surprise_pct", "reported_eps", "estimated_eps"]].dropna(subset=["date"])
        # Drop rows where estimate is zero (surprise is undefined/infinite)
        df = df[df["estimated_eps"].abs() > 1e-6]
        df = df.drop_duplicates(subset=["date"])

        return pl.from_pandas(df).with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("surprise_pct").cast(pl.Float64),
            pl.col("reported_eps").cast(pl.Float64),
            pl.col("estimated_eps").cast(pl.Float64),
        ])
    except Exception:
        return empty


def seed_earnings(
    store: DataStore,
    tickers: list[str],
    on_progress: Callable[[str], None] | None = None,
) -> int:
    """Seed earnings_surprises table for the given tickers.

    Returns total number of rows written.
    """
    total = 0
    for ticker in tickers:
        rows = fetch_earnings(ticker)
        if not rows.is_empty():
            store.store_earnings(rows)
            total += len(rows)
        if on_progress:
            on_progress(ticker)
    return total
