"""
Point-in-time S&P 500 universe filtering for the backtest engine.

Given a rebalance date, returns the subset of S&P 500 constituents that:
  (a) were index members on that date (entry_date <= date < exit_date)
  (b) had non-null daily OHLCV data for the prior 252 trading days
  (c) had all required signal input columns available as of that date

Enforces the rule: no data field may have a timestamp after the rebalance date.
Typical universe size per date: 480–500 tickers.
"""

from __future__ import annotations

from datetime import date

from pelican.data.store import DataStore


def get_rebalance_dates(start: date, end: date, store: DataStore) -> list[date]:
    """Return the first trading day of each calendar month in [start, end].

    Derived from the prices table so only months with actual data are included.
    """
    result = store.query(
        """
        SELECT DATE_TRUNC('month', date) AS month_start, MIN(date) AS rebal_date
        FROM prices
        WHERE date >= ? AND date <= ?
        GROUP BY month_start
        ORDER BY rebal_date
        """,
        [start, end],
    )
    return result["rebal_date"].to_list()


def get_point_in_time_universe(
    rebal_date: date,
    store: DataStore,
    min_history_days: int = 252,
) -> list[str]:
    """Tickers in the S&P 500 on `rebal_date` with sufficient price history.

    A ticker is included if:
      - It was an S&P 500 member on `rebal_date` (entry_date <= rebal_date < exit_date)
      - It has at least `min_history_days` rows of price data on or before `rebal_date`
      - Its most recent price row is on or before `rebal_date` (no look-ahead)
    """
    result = store.query(
        """
        SELECT u.ticker
        FROM sp500_universe u
        INNER JOIN (
            SELECT ticker, COUNT(*) AS n_days
            FROM prices
            WHERE date <= ?
            GROUP BY ticker
        ) p ON u.ticker = p.ticker
        WHERE u.entry_date <= ?
          AND (u.exit_date IS NULL OR u.exit_date > ?)
          AND p.n_days >= ?
        ORDER BY u.ticker
        """,
        [rebal_date, rebal_date, rebal_date, min_history_days],
    )
    return result["ticker"].to_list()
