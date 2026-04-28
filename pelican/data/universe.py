"""S&P 500 equity universe management: survivorship-bias-free historical membership."""

from __future__ import annotations

import re
from datetime import date
from html.parser import HTMLParser

import httpx
import polars as pl

from pelican.data.store import DataStore

# S&P 500 inception date used for tickers with no recorded addition date.
_INCEPTION = date(1993, 1, 29)

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ---------------------------------------------------------------------------
# HTML table parser (stdlib, no BeautifulSoup)
# ---------------------------------------------------------------------------

class _TableParser(HTMLParser):
    """Extract all <table> blocks from HTML into a list of lists-of-lists."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_cell = False
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif tag == "tr" and self._in_table:
            self._current_row = []
        elif tag in ("td", "th") and self._in_table:
            self._in_cell = True
            self._current_cell = []
        elif tag == "a" and self._in_cell:
            # Capture href text will come from handle_data
            pass

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            if self._current_table:
                self.tables.append(self._current_table)
            self._in_table = False
        elif tag == "tr" and self._in_table:
            if self._current_row:
                self._current_table.append(self._current_row)
        elif tag in ("td", "th") and self._in_table:
            self._in_cell = False
            self._current_row.append(" ".join(self._current_cell).strip())

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)


def _parse_tables(html: str) -> list[list[list[str]]]:
    parser = _TableParser()
    parser.feed(html)
    return parser.tables


def _parse_date(s: str) -> date | None:
    s = s.strip()
    if not s or s == "-":
        return None
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # Try partial: "2020" → ignore
    if re.fullmatch(r"\d{4}", s):
        return None
    return None


# ---------------------------------------------------------------------------
# Public ingestion functions
# ---------------------------------------------------------------------------

def fetch_sp500_constituents() -> pl.DataFrame:
    """Fetch current S&P 500 constituents from Wikipedia table 0.

    Returns a DataFrame with columns: ticker, company, date_added.
    """
    resp = httpx.get(_WIKI_URL, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    tables = _parse_tables(resp.text)

    if not tables:
        raise RuntimeError("No tables found on Wikipedia S&P 500 page")

    # Table 0: current constituents.
    # Columns vary but ticker is always col 0, company col 1.
    rows = tables[0][1:]  # skip header
    tickers, companies, dates_added = [], [], []
    for row in rows:
        if len(row) < 2:
            continue
        ticker = row[0].strip().replace("\n", "")
        company = row[1].strip()
        date_added = _parse_date(row[6]) if len(row) > 6 else None
        if ticker:
            tickers.append(ticker)
            companies.append(company)
            dates_added.append(date_added)

    return pl.DataFrame({
        "ticker": tickers,
        "company": companies,
        "date_added": pl.Series(dates_added, dtype=pl.Date),
    })


def fetch_sp500_changes() -> pl.DataFrame:
    """Fetch historical S&P 500 addition/removal changes from Wikipedia table 1.

    Returns a DataFrame with: date, added_ticker, added_company,
    removed_ticker, removed_company.
    """
    resp = httpx.get(_WIKI_URL, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    tables = _parse_tables(resp.text)

    if len(tables) < 2:
        # Return empty frame — no changes history available
        return pl.DataFrame({
            "date": pl.Series([], dtype=pl.Date),
            "added_ticker": pl.Series([], dtype=pl.Utf8),
            "added_company": pl.Series([], dtype=pl.Utf8),
            "removed_ticker": pl.Series([], dtype=pl.Utf8),
            "removed_company": pl.Series([], dtype=pl.Utf8),
        })

    # Table 1 columns: Date | Added Symbol | Added Security | Removed Symbol | Removed Security
    rows = tables[1][1:]
    dates, added_tickers, added_companies = [], [], []
    removed_tickers, removed_companies = [], []

    for row in rows:
        if len(row) < 4:
            continue
        dt = _parse_date(row[0])
        added_t = row[1].strip()
        added_c = row[2].strip() if len(row) > 2 else ""
        removed_t = row[3].strip() if len(row) > 3 else ""
        removed_c = row[4].strip() if len(row) > 4 else ""

        dates.append(dt)
        added_tickers.append(added_t)
        added_companies.append(added_c)
        removed_tickers.append(removed_t)
        removed_companies.append(removed_c)

    return pl.DataFrame({
        "date": pl.Series(dates, dtype=pl.Date),
        "added_ticker": added_tickers,
        "added_company": added_companies,
        "removed_ticker": removed_tickers,
        "removed_company": removed_companies,
    })


def build_universe_history(
    constituents: pl.DataFrame,
    changes: pl.DataFrame,
) -> pl.DataFrame:
    """Reconstruct (ticker, entry_date, exit_date, company) from Wikipedia tables.

    Algorithm:
    1. Current constituents start as "in" with exit_date = NULL.
       entry_date is taken from the `date_added` column when available,
       otherwise defaults to _INCEPTION.
    2. Walk through the changes table:
       - Added ticker → creates an entry row (entry_date = change date)
       - Removed ticker → creates an exit row (exit_date = change date)
    3. Merge: for every added ticker in changes that is NOT in the current
       constituent list, it was added and subsequently removed → assign the
       exit_date from the removal record.
    """
    records: list[dict] = []

    # Current constituents
    current_tickers: set[str] = set(constituents["ticker"].to_list())
    ticker_to_company: dict[str, str] = dict(
        zip(constituents["ticker"].to_list(), constituents["company"].to_list())
    )
    ticker_to_date_added: dict[str, date | None] = dict(
        zip(constituents["ticker"].to_list(), constituents["date_added"].to_list())
    )

    # Seed current constituents; entry_date resolved below from changes.
    # We build a dict: ticker → list of (entry_date, exit_date, company)
    windows: dict[str, list[tuple[date | None, date | None, str]]] = {}

    for t in current_tickers:
        windows[t] = [(ticker_to_date_added.get(t), None, ticker_to_company.get(t, ""))]

    # Process changes sorted by date
    sorted_changes = changes.sort("date", nulls_last=True)

    for row in sorted_changes.iter_rows(named=True):
        change_date: date | None = row["date"]
        if change_date is None:
            continue

        added_t = (row["added_ticker"] or "").strip()
        removed_t = (row["removed_ticker"] or "").strip()

        # A ticker was added on this date
        if added_t:
            if added_t not in windows:
                windows[added_t] = []
            # Open a new window
            company = row.get("added_company") or ticker_to_company.get(added_t, "")
            windows[added_t].append((change_date, None, company))

        # A ticker was removed on this date
        if removed_t:
            if removed_t not in windows:
                windows[removed_t] = []
                # Ticker we've never seen added — assume it was there since inception
                company = row.get("removed_company") or ticker_to_company.get(removed_t, "")
                windows[removed_t].append((_INCEPTION, None, company))
            # Close the most recent open window for this ticker
            ticker_windows = windows[removed_t]
            for i in range(len(ticker_windows) - 1, -1, -1):
                entry_d, exit_d, co = ticker_windows[i]
                if exit_d is None:
                    ticker_windows[i] = (entry_d, change_date, co)
                    break

    # Build final records
    for ticker, wins in windows.items():
        for entry_d, exit_d, company in wins:
            # Current constituents: if entry_date is still None, use inception
            if entry_d is None:
                entry_d = _INCEPTION
            records.append({
                "ticker": ticker,
                "entry_date": entry_d,
                "exit_date": exit_d,
                "company": company,
            })

    # Deduplicate by (ticker, entry_date) — the constituent seed and the changes table
    # can both produce a window for the same entry date.  When there is a conflict,
    # prefer the record that carries an exit_date (more information).
    seen: dict[tuple[str, date], dict] = {}
    for rec in records:
        key = (rec["ticker"], rec["entry_date"])
        if key not in seen or (rec["exit_date"] is not None and seen[key]["exit_date"] is None):
            seen[key] = rec
    records = list(seen.values())

    if not records:
        return pl.DataFrame(schema={
            "ticker": pl.Utf8,
            "entry_date": pl.Date,
            "exit_date": pl.Date,
            "company": pl.Utf8,
        })

    return pl.DataFrame(records, schema={
        "ticker": pl.Utf8,
        "entry_date": pl.Date,
        "exit_date": pl.Date,
        "company": pl.Utf8,
    }).sort(["ticker", "entry_date"])


def load_universe(store: DataStore) -> None:
    """Fetch S&P 500 history from Wikipedia and write to DuckDB."""
    constituents = fetch_sp500_constituents()
    changes = fetch_sp500_changes()
    history = build_universe_history(constituents, changes)
    store.write(history, "sp500_universe")


def get_universe(query_date: date, store: DataStore) -> list[str]:
    """Return tickers in the S&P 500 on `query_date` (point-in-time)."""
    result = store.query(
        """
        SELECT ticker
        FROM sp500_universe
        WHERE entry_date <= ?
          AND (exit_date IS NULL OR exit_date > ?)
        ORDER BY ticker
        """,
        [query_date, query_date],
    )
    return result["ticker"].to_list()
