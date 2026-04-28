"""Seed the DuckDB data store with S&P 500 universe history and OHLCV prices.

Usage:
  python scripts/seed_data.py [--start 2014-01-01] [--end 2024-01-01] \
                               [--batch-size 50] [--db-path ./data/pelican.duckdb]

Expected runtime: ~20–40 min. Expected disk usage: ~5–15 GB depending on date range.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Pelican data store")
    parser.add_argument("--start", default="2014-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--batch-size", type=int, default=50, help="Tickers per yfinance batch")
    parser.add_argument(
        "--db-path", default="./data/pelican.duckdb", help="Path to DuckDB file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy imports so the script fails fast on bad args before loading heavy deps.
    from pelican.utils.logging import configure_logging, get_logger
    configure_logging(dev=True)
    log = get_logger("seed")

    from pelican.data.prices import load_prices
    from pelican.data.store import DataStore
    from pelican.data.universe import get_universe, load_universe

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("seeding started", db=str(db_path), start=str(start), end=str(end))
    t0 = time.monotonic()

    with DataStore(db_path) as store:
        store.init_schema()

        # --- Universe ---
        log.info("fetching S&P 500 universe from Wikipedia")
        load_universe(store)
        # Collect every ticker that ever appeared in the index
        all_tickers = store.query(
            "SELECT DISTINCT ticker FROM sp500_universe ORDER BY ticker"
        )["ticker"].to_list()
        log.info("universe loaded", total_tickers=len(all_tickers))

        # Show point-in-time snapshot as a sanity check
        current = get_universe(date.today(), store)
        log.info("current universe size", tickers=len(current))

        # --- Prices ---
        log.info(
            "downloading prices",
            tickers=len(all_tickers),
            batch_size=args.batch_size,
        )
        load_prices(store, all_tickers, start, end, batch_size=args.batch_size)

        price_rows = store.query("SELECT count(*) AS n FROM prices").item(0, 0)
        log.info("prices written", rows=price_rows)

    elapsed = time.monotonic() - t0
    log.info("seeding complete", elapsed_seconds=round(elapsed, 1))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
