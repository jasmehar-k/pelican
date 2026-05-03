#!/usr/bin/env python3
"""Seed news_sentiment table with LLM-scored headlines from yfinance.

Usage:
    python scripts/seed_news.py
    python scripts/seed_news.py --tickers AAPL MSFT GOOGL
    python scripts/seed_news.py --model meta-llama/llama-3.3-70b-instruct:free
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pelican.data.news import seed_news_sentiment
from pelican.data.store import DataStore
from pelican.utils.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed news sentiment data")
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Tickers to process (default: full S&P 500 universe from DB)"
    )
    parser.add_argument(
        "--model", default=None,
        help="OpenRouter model ID for scoring (default: settings.edgar_tone_model)"
    )
    args = parser.parse_args()

    s = get_settings()
    store = DataStore(s.duckdb_path)
    store.init_schema()

    if args.tickers:
        tickers = args.tickers
    else:
        try:
            universe_df = store.query(
                "SELECT DISTINCT ticker FROM sp500_universe ORDER BY ticker"
            )
            tickers = universe_df["ticker"].to_list()
        except Exception:
            print("No universe in DB — pass --tickers explicitly", file=sys.stderr)
            sys.exit(1)

    print(f"Seeding news sentiment for {len(tickers)} tickers…")
    done = [0]

    def progress(ticker: str) -> None:
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(tickers):
            print(f"  {done[0]}/{len(tickers)} tickers processed")

    n = seed_news_sentiment(store, tickers, model=args.model, on_progress=progress)
    print(f"Done — wrote {n} rows to news_sentiment")
    store.close()


if __name__ == "__main__":
    main()
