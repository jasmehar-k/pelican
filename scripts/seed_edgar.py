"""
Seed the edgar_sentiment table with MD&A tone scores for S&P 500 tickers.

Downloads 10-K and 10-Q primary documents from EDGAR, extracts the MD&A
section, scores management tone with an LLM, and computes the YoY delta.

Usage:
    python scripts/seed_edgar.py
    python scripts/seed_edgar.py --tickers AAPL MSFT GOOGL
    python scripts/seed_edgar.py --after 2020-01-01 --limit 12
    python scripts/seed_edgar.py --filing-types 10-K
    python scripts/seed_edgar.py --model deepseek/deepseek-chat:free

Rate limiting: ~0.11 s/request to EDGAR (under SEC 10 req/s cap).
Expected runtime: ~30 min for the full S&P 500 with default 8 filings/ticker.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn
from rich.table import Table

from pelican.data.edgar import seed_edgar_sentiment
from pelican.data.store import DataStore
from pelican.utils.config import get_settings


def parse_args(argv=None) -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="Seed edgar_sentiment table")
    p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers to process (default: all from sp500_universe)",
    )
    p.add_argument("--after", default=None, help="Only filings on/after YYYY-MM-DD")
    p.add_argument("--before", default=None, help="Only filings on/before YYYY-MM-DD")
    p.add_argument("--limit", type=int, default=8,
                   help="Max filings per (ticker, filing-type) pair (default: 8)")
    p.add_argument("--filing-types", nargs="+", default=["10-K", "10-Q"],
                   metavar="TYPE", help="Filing types to download (default: 10-K 10-Q)")
    p.add_argument("--model", default=None, help="OpenRouter model ID for tone scoring")
    p.add_argument("--db-path", default=str(s.duckdb_path))
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    console = Console()

    db_path = Path(args.db_path)
    if not db_path.exists():
        console.print(f"[bold red]ERROR[/] database not found at {db_path}")
        sys.exit(1)

    store = DataStore(db_path)
    store.init_schema()

    # Resolve tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        try:
            df = store.query("SELECT DISTINCT ticker FROM sp500_universe ORDER BY ticker")
            tickers = df["ticker"].to_list()
        except Exception as exc:
            console.print(f"[bold red]ERROR[/] could not load universe: {exc}")
            sys.exit(1)

    after = date.fromisoformat(args.after) if args.after else None
    before = date.fromisoformat(args.before) if args.before else None
    filing_types = tuple(t.upper() for t in args.filing_types)

    console.print()
    console.print(Panel(
        f"Tickers: [bold]{len(tickers)}[/]  "
        f"Filing types: [bold]{', '.join(filing_types)}[/]  "
        f"Limit: [bold]{args.limit}[/] per type",
        title="[bold cyan]EDGAR Sentiment Seeding[/]",
        border_style="cyan",
    ))
    console.print()

    t0 = time.monotonic()
    total_rows = 0

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Processing tickers", total=len(tickers))

        def on_progress(ticker: str) -> None:
            progress.advance(task)

        total_rows = seed_edgar_sentiment(
            store=store,
            tickers=tickers,
            filing_types=filing_types,
            model=args.model,
            after=after,
            before=before,
            limit=args.limit,
            on_progress=on_progress,
        )

    elapsed = time.monotonic() - t0

    # Summary
    try:
        coverage = store.get_edgar_coverage()
        t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        t.add_column("Ticker", style="bold")
        t.add_column("Filings", justify="right")
        t.add_column("First filing")
        t.add_column("Last filing")
        for row in coverage.to_dicts()[:20]:
            t.add_row(
                row["ticker"],
                str(row["n_filings"]),
                str(row["first_filing"]),
                str(row["last_filing"]),
            )
        if len(coverage) > 20:
            t.add_row(f"… and {len(coverage) - 20} more", "", "", "")
        console.print(Panel(t, title="[bold green]Coverage summary[/]", border_style="green"))
    except Exception:
        pass

    console.print(
        f"\n[bold green]Done[/] — {total_rows} rows written in {elapsed:.1f}s"
    )
    store.close()


if __name__ == "__main__":
    main()
