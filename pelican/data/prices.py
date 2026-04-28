"""Daily OHLCV price data: download via yfinance, compute returns, store in DuckDB."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import polars as pl
import yfinance as yf

from pelican.data.store import DataStore
from pelican.utils.logging import get_logger

log = get_logger(__name__)


def download_prices(
    tickers: list[str],
    start: date,
    end: date,
    batch_size: int = 50,
) -> pl.DataFrame:
    """Download adjusted OHLCV from yfinance in batches.

    Returns a flat (ticker, date, open, high, low, close, volume) DataFrame.
    Delisted tickers that return partial data are included. Tickers that fail
    entirely are logged at WARNING and excluded.
    """
    all_frames: list[pl.DataFrame] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        log.info("downloading prices", batch_start=i, batch_size=len(batch))
        try:
            raw = yf.download(
                batch,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as exc:
            log.warning("yfinance batch failed", error=str(exc))
            continue

        if raw.empty:
            continue

        # yfinance returns MultiIndex columns when multiple tickers requested.
        # Single-ticker requests return flat columns.
        if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, "levels"):
            for t in batch:
                try:
                    sub = raw[t].dropna(how="all")
                except KeyError:
                    log.warning("no data for ticker", ticker=t)
                    continue
                if sub.empty:
                    continue
                frame = _pandas_to_polars(sub, t)
                if frame is not None:
                    all_frames.append(frame)
        else:
            # Single ticker — flat columns
            t = batch[0]
            frame = _pandas_to_polars(raw.dropna(how="all"), t)
            if frame is not None:
                all_frames.append(frame)

    if not all_frames:
        return pl.DataFrame(schema={
            "ticker": pl.Utf8, "date": pl.Date,
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
            "close": pl.Float64, "volume": pl.Int64,
        })

    return pl.concat(all_frames, how="diagonal_relaxed")


def _pandas_to_polars(df: Any, ticker: str) -> pl.DataFrame | None:
    """Convert a yfinance single-ticker pandas DataFrame to Polars."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        return None
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "volume"})
    pdf = df[["date", "open", "high", "low", "close", "volume"]].copy()
    pdf.insert(0, "ticker", ticker)
    try:
        pl_df = pl.from_pandas(pdf)
    except Exception:
        return None
    return pl_df.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    )


def compute_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Add log_return_1d and forward_return_21d to an OHLCV DataFrame.

    Both are computed per-ticker via .over("ticker").
    - log_return_1d: uses past data only — safe as a signal input.
    - forward_return_21d: uses future prices — this is the backtest LABEL,
      not a signal input. NaN for the last 21 rows per ticker.
    """
    return (
        df.sort(["ticker", "date"])
        .with_columns([
            (
                pl.col("close") / pl.col("close").shift(1).over("ticker")
            ).log(base=math.e).alias("log_return_1d"),
            (
                pl.col("close").shift(-21).over("ticker") / pl.col("close") - 1.0
            ).alias("forward_return_21d"),
        ])
    )


def load_prices(
    store: DataStore,
    tickers: list[str],
    start: date,
    end: date,
    batch_size: int = 50,
) -> None:
    """Download prices, compute returns, and write to DuckDB."""
    raw = download_prices(tickers, start, end, batch_size=batch_size)
    if raw.is_empty():
        log.warning("no price data downloaded")
        return
    enriched = compute_returns(raw)
    n = store.write(enriched, "prices")
    log.info("prices written", rows=n)


def get_prices(
    tickers: list[str],
    start: date,
    end: date,
    store: DataStore,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Read prices for specific tickers and date range from DuckDB.

    Returns (date, ticker, ...) sorted by (ticker, date).
    """
    col_clause = ", ".join(columns) if columns else "*"
    placeholders = ", ".join("?" * len(tickers))
    return store.query(
        f"""
        SELECT {col_clause}
        FROM prices
        WHERE ticker IN ({placeholders})
          AND date >= ?
          AND date <= ?
        ORDER BY ticker, date
        """,
        [*tickers, start, end],
    )


def get_panel(
    start: date,
    end: date,
    store: DataStore,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Return the full (date × ticker) panel for S&P 500 members in [start, end].

    Joins prices against point-in-time universe membership.
    """
    col_clause = (
        ", ".join(f"p.{c}" for c in columns)
        if columns
        else "p.*"
    )
    return store.query(
        f"""
        SELECT {col_clause}
        FROM prices p
        INNER JOIN sp500_universe u
            ON p.ticker = u.ticker
           AND p.date >= u.entry_date
           AND (u.exit_date IS NULL OR p.date < u.exit_date)
        WHERE p.date >= ?
          AND p.date <= ?
        ORDER BY p.date, p.ticker
        """,
        [start, end],
    )
