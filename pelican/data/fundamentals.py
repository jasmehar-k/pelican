"""
Quarterly fundamental data via yfinance.

Downloads balance sheet and income statement data for S&P 500 tickers,
computes financial ratios with a point-in-time filing lag, and writes
to the DuckDB fundamentals table.

Filing lag: period_end + 45 days (conservative quarterly lag).
This ensures the signal can only use data the market already had access to.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from pelican.data.store import DataStore
from pelican.utils.logging import get_logger

log = get_logger(__name__)

QUARTERLY_LAG_DAYS = 45

# yfinance row names in the transposed panel
_BS_SHARES = "Ordinary Shares Number"
_BS_EQUITY = "Stockholders Equity"
_BS_DEBT = "Total Debt"
_IS_NET_INCOME = "Net Income"


def fetch_fundamentals(ticker: str) -> pl.DataFrame | None:
    """Download quarterly balance sheet + income statement for `ticker`.

    Returns a DataFrame with columns:
      ticker, period_end, shares_outstanding, equity, total_debt, net_income
    one row per fiscal quarter. Returns None on any error.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed")
        return None

    try:
        t = yf.Ticker(ticker)
        bs = t.quarterly_balance_sheet
        inc = t.quarterly_income_stmt
    except Exception as exc:
        log.warning("yfinance fetch failed", ticker=ticker, error=str(exc))
        return None

    if bs is None or bs.empty or inc is None or inc.empty:
        log.warning("no fundamental data", ticker=ticker)
        return None

    try:
        rows = []
        for col in bs.columns:
            period_end = col.date() if hasattr(col, "date") else date.fromisoformat(str(col)[:10])

            def _get(frame, row_name):
                if row_name in frame.index and col in frame.columns:
                    v = frame.loc[row_name, col]
                    import math
                    return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
                return None

            shares = _get(bs, _BS_SHARES)
            equity = _get(bs, _BS_EQUITY)
            total_debt = _get(bs, _BS_DEBT)
            net_income = _get(inc, _IS_NET_INCOME) if col in inc.columns else None

            rows.append({
                "ticker": ticker,
                "period_end": period_end,
                "shares_outstanding": shares,
                "equity": equity,
                "total_debt": total_debt,
                "net_income": net_income,
            })

        if not rows:
            return None

        return pl.DataFrame(rows, schema={
            "ticker": pl.Utf8,
            "period_end": pl.Date,
            "shares_outstanding": pl.Float64,
            "equity": pl.Float64,
            "total_debt": pl.Float64,
            "net_income": pl.Float64,
        })

    except Exception as exc:
        log.warning("fundamental parse failed", ticker=ticker, error=str(exc))
        return None


def compute_fundamental_ratios(
    fund_df: pl.DataFrame,
    prices_df: pl.DataFrame,
) -> pl.DataFrame:
    """Join quarterly fundamentals with prices at period_end to compute ratios.

    Adds: market_cap, pe_ratio, pb_ratio, roe, debt_to_equity, available_date.
    Undefined ratios (zero/negative equity, zero shares) → null.
    """
    # Get the closing price on (or nearest before) each period_end date per ticker.
    # We join on exact date; missing dates stay null and ratios will be null.
    price_snap = (
        prices_df.select(["ticker", "date", "close"])
        .rename({"date": "period_end", "close": "close_at_period_end"})
    )

    df = fund_df.join(price_snap, on=["ticker", "period_end"], how="left")

    df = df.with_columns([
        # market_cap = shares * price
        (pl.col("shares_outstanding") * pl.col("close_at_period_end")).alias("market_cap"),
        # earnings per share for ratio computation
        (pl.col("net_income") / pl.col("shares_outstanding")).alias("_eps"),
        # book value per share
        (pl.col("equity") / pl.col("shares_outstanding")).alias("_bvps"),
    ])

    df = df.with_columns([
        # pe_ratio = price / eps; null if eps <= 0 or null
        pl.when(pl.col("_eps") > 0)
          .then(pl.col("close_at_period_end") / pl.col("_eps"))
          .otherwise(None)
          .alias("pe_ratio"),
        # pb_ratio = price / bvps; null if bvps <= 0 or null
        pl.when(pl.col("_bvps") > 0)
          .then(pl.col("close_at_period_end") / pl.col("_bvps"))
          .otherwise(None)
          .alias("pb_ratio"),
        # roe = net_income / equity; null if equity <= 0 or null
        pl.when(pl.col("equity") > 0)
          .then(pl.col("net_income") / pl.col("equity"))
          .otherwise(None)
          .alias("roe"),
        # debt_to_equity; null if equity <= 0 or null
        pl.when(pl.col("equity") > 0)
          .then(pl.col("total_debt") / pl.col("equity"))
          .otherwise(None)
          .alias("debt_to_equity"),
        # point-in-time anchor: period_end + filing lag
        (pl.col("period_end") + timedelta(days=QUARTERLY_LAG_DAYS)).alias("available_date"),
    ])

    return df.select([
        "ticker", "available_date", "period_end",
        "market_cap", "pe_ratio", "pb_ratio", "roe", "debt_to_equity",
    ])


def load_fundamentals(store: DataStore, tickers: list[str]) -> int:
    """Fetch quarterly fundamentals for `tickers` and upsert into DuckDB.

    Returns total rows written.
    """
    # Pull the prices panel once for ratio computation.
    prices_df = store.query(
        "SELECT ticker, date, close FROM prices WHERE ticker IN ({})".format(
            ", ".join(f"'{t}'" for t in tickers)
        )
    )

    total = 0
    for ticker in tickers:
        raw = fetch_fundamentals(ticker)
        if raw is None or raw.is_empty():
            continue
        try:
            ratios = compute_fundamental_ratios(
                raw,
                prices_df.filter(pl.col("ticker") == ticker),
            )
            if ratios.is_empty():
                continue
            written = store.write(ratios, "fundamentals")
            total += written
            log.info("fundamentals loaded", ticker=ticker, rows=written)
        except Exception as exc:
            log.warning("fundamentals write failed", ticker=ticker, error=str(exc))

    return total
