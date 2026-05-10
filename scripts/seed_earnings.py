#!/usr/bin/env python3
"""Seed earnings_surprises table with historical EPS surprise data from yfinance.

Usage:
    python scripts/seed_earnings.py                           # full S&P 500 universe
    python scripts/seed_earnings.py --tickers AAPL MSFT GOOG # specific tickers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing.
sys.path.insert(0, str(Path(__file__).parent.parent))

from pelican.data.earnings import seed_earnings
from pelican.data.store import DataStore
from pelican.utils.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed earnings_surprises table.")
    parser.add_argument("--tickers", nargs="+", help="Tickers to seed (default: full universe from DB)")
    args = parser.parse_args()

    settings = get_settings()
    store = DataStore(settings.duckdb_path)
    store.init_schema()

    if args.tickers:
        tickers = args.tickers
    else:
        universe = store.query(
            "SELECT DISTINCT ticker FROM sp500_universe ORDER BY ticker"
        )
        tickers = universe["ticker"].to_list()

    print(f"Seeding earnings surprises for {len(tickers)} tickers…")
    done = 0

    def on_progress(ticker: str) -> None:
        nonlocal done
        done += 1
        if done % 50 == 0 or done == len(tickers):
            print(f"  {done}/{len(tickers)} tickers processed", flush=True)

    total = seed_earnings(store, tickers, on_progress=on_progress)
    print(f"Done — {total} rows written to earnings_surprises.")
    store.close()


if __name__ == "__main__":
    main()
