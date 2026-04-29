"""
Seed the DuckDB fundamentals table with quarterly yfinance data.

Downloads quarterly balance sheet and income statement data for all
historical S&P 500 constituents and computes point-in-time financial ratios.

Usage:
    python scripts/seed_fundamentals.py
    python scripts/seed_fundamentals.py --db-path ./data/pelican.duckdb --batch-size 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pelican.data.fundamentals import load_fundamentals
from pelican.data.store import DataStore
from pelican.utils.config import get_settings
from pelican.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="Seed fundamentals table")
    p.add_argument("--db-path", default=str(s.duckdb_path))
    p.add_argument("--batch-size", type=int, default=20)
    return p.parse_args(argv)


def main(argv=None) -> None:
    configure_logging(dev=True)
    args = parse_args(argv)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        print("Run:  python scripts/seed_data.py  first.")
        sys.exit(1)

    store = DataStore(db_path)
    store.init_schema()

    # Collect all tickers that ever appeared in the S&P 500.
    all_tickers_df = store.query("SELECT DISTINCT ticker FROM sp500_universe ORDER BY ticker")
    if all_tickers_df.is_empty():
        print("ERROR: no tickers in sp500_universe — seed price data first.")
        store.close()
        sys.exit(1)

    all_tickers = all_tickers_df["ticker"].to_list()
    print(f"\nSeeding fundamentals for {len(all_tickers)} tickers in batches of {args.batch_size}...")

    t0 = time.perf_counter()
    total_rows = 0
    no_data: list[str] = []

    for i in range(0, len(all_tickers), args.batch_size):
        batch = all_tickers[i : i + args.batch_size]
        batch_label = f"[{i + 1}–{min(i + args.batch_size, len(all_tickers))}/{len(all_tickers)}]"
        print(f"  {batch_label} {', '.join(batch[:3])}{'...' if len(batch) > 3 else ''}")

        before = store.query("SELECT COUNT(*) AS n FROM fundamentals").item(0, 0)
        load_fundamentals(store, batch)
        after = store.query("SELECT COUNT(*) AS n FROM fundamentals").item(0, 0)
        written = int(after) - int(before)
        total_rows += written

        if written == 0:
            no_data.extend(batch)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Rows written : {total_rows}")
    print(f"  No data      : {len(no_data)} tickers")
    if no_data:
        print(f"    {no_data[:10]}{'...' if len(no_data) > 10 else ''}")

    store.close()


if __name__ == "__main__":
    main()
