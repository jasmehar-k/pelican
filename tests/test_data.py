"""pytest suite for the data foundation layer. No network calls — synthetic data only."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import polars as pl
import pytest

from pelican.data.prices import compute_returns, download_prices, get_panel, get_prices
from pelican.data.store import DataStore
from pelican.data.universe import (
    _parse_date,
    _parse_tables,
    build_universe_history,
    fetch_sp500_changes,
    fetch_sp500_constituents,
    get_universe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> DataStore:
    s = DataStore(":memory:")
    s.init_schema()
    return s


def _universe_df(**rows: dict) -> pl.DataFrame:
    """Helper: build a sp500_universe DataFrame from keyword-list args."""
    return pl.DataFrame(rows, schema={
        "ticker": pl.Utf8,
        "entry_date": pl.Date,
        "exit_date": pl.Date,
        "company": pl.Utf8,
    })


def _price_series(ticker: str, n: int = 60, start: date = date(2020, 1, 2)) -> pl.DataFrame:
    """Generate a synthetic monotonically-increasing price series for `ticker`."""
    dates = [start + timedelta(days=i) for i in range(n)]
    closes = [100.0 + i for i in range(n)]
    return pl.DataFrame({
        "ticker": [ticker] * n,
        "date": dates,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1_000_000] * n,
    })


# ---------------------------------------------------------------------------
# DataStore tests
# ---------------------------------------------------------------------------

def test_store_write_read_roundtrip(store: DataStore) -> None:
    df = _universe_df(
        ticker=["AAPL"],
        entry_date=[date(2020, 1, 1)],
        exit_date=[None],
        company=["Apple"],
    )
    store.write(df, "sp500_universe")
    result = store.query("SELECT * FROM sp500_universe")
    assert result["ticker"][0] == "AAPL"
    assert result["entry_date"][0] == date(2020, 1, 1)
    assert result["exit_date"][0] is None


def test_store_upsert_no_duplicates(store: DataStore) -> None:
    df = _universe_df(
        ticker=["AAPL"],
        entry_date=[date(2020, 1, 1)],
        exit_date=[None],
        company=["Apple"],
    )
    store.write(df, "sp500_universe")
    store.write(df, "sp500_universe")  # same PK, should not duplicate
    count = store.query("SELECT count(*) AS n FROM sp500_universe").item(0, 0)
    assert count == 1


def test_store_init_schema_idempotent(store: DataStore) -> None:
    store.init_schema()
    tables = store.query("SHOW TABLES")
    assert set(tables["name"].to_list()) == {"prices", "sp500_universe", "fundamentals", "research_log", "signal_memos"}


def test_store_query_preserves_empty_schema(store: DataStore) -> None:
    result = store.query("SELECT ticker, entry_date FROM sp500_universe WHERE 1 = 0")
    assert result.columns == ["ticker", "entry_date"]
    assert result.schema["ticker"] == pl.Utf8
    assert result.schema["entry_date"] == pl.Date
    assert result.is_empty()


def test_store_upsert_replaces_existing_row(store: DataStore) -> None:
    original = _universe_df(
        ticker=["AAPL"],
        entry_date=[date(2020, 1, 1)],
        exit_date=[None],
        company=["Apple Inc."],
    )
    updated = _universe_df(
        ticker=["AAPL"],
        entry_date=[date(2020, 1, 1)],
        exit_date=[date(2024, 1, 1)],
        company=["Apple, Updated"],
    )
    store.write(original, "sp500_universe")
    store.write(updated, "sp500_universe")
    result = store.query("SELECT * FROM sp500_universe WHERE ticker = 'AAPL'")
    assert result["company"][0] == "Apple, Updated"
    assert result["exit_date"][0] == date(2024, 1, 1)


def test_store_context_manager_closes_connection() -> None:
    with DataStore(":memory:") as managed_store:
        managed_store.init_schema()
        managed_store.query("SELECT 1")

    with pytest.raises(Exception):
        managed_store.query("SELECT 1")


# ---------------------------------------------------------------------------
# Universe tests
# ---------------------------------------------------------------------------

def test_universe_includes_active_ticker(store: DataStore) -> None:
    df = _universe_df(
        ticker=["AAPL"],
        entry_date=[date(2019, 1, 1)],
        exit_date=[None],
        company=["Apple"],
    )
    store.write(df, "sp500_universe")
    result = get_universe(date(2022, 6, 1), store)
    assert "AAPL" in result


def test_universe_excludes_exited_ticker(store: DataStore) -> None:
    df = _universe_df(
        ticker=["XYZ"],
        entry_date=[date(2015, 1, 1)],
        exit_date=[date(2020, 1, 1)],  # left before query date
        company=["XYZ Corp"],
    )
    store.write(df, "sp500_universe")
    result = get_universe(date(2022, 6, 1), store)
    assert "XYZ" not in result


def test_universe_excludes_future_entry(store: DataStore) -> None:
    df = _universe_df(
        ticker=["NEWCO"],
        entry_date=[date(2025, 1, 1)],  # future
        exit_date=[None],
        company=["NewCo"],
    )
    store.write(df, "sp500_universe")
    result = get_universe(date(2022, 6, 1), store)
    assert "NEWCO" not in result


def test_universe_boundary_same_day_exit(store: DataStore) -> None:
    """Ticker that exits ON the query date should not be included."""
    df = _universe_df(
        ticker=["GONE"],
        entry_date=[date(2018, 1, 1)],
        exit_date=[date(2022, 6, 1)],  # exits on query date
        company=["Gone Corp"],
    )
    store.write(df, "sp500_universe")
    result = get_universe(date(2022, 6, 1), store)
    assert "GONE" not in result


def test_universe_multi_window_ticker(store: DataStore) -> None:
    """Ticker that left and rejoined appears in the right windows only."""
    df = _universe_df(
        ticker=["FLEX", "FLEX"],
        entry_date=[date(2010, 1, 1), date(2021, 1, 1)],
        exit_date=[date(2018, 1, 1), None],
        company=["Flex", "Flex"],
    )
    store.write(df, "sp500_universe")
    assert "FLEX" not in get_universe(date(2019, 6, 1), store)   # between windows
    assert "FLEX" in get_universe(date(2022, 6, 1), store)        # in second window


def test_build_universe_history_current_no_changes() -> None:
    """Without any changes, every current constituent gets entry_date=1993-01-29."""
    constituents = pl.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "company": ["Apple", "Microsoft"],
        "date_added": [None, None],
    })
    changes = pl.DataFrame({
        "date": pl.Series([], dtype=pl.Date),
        "added_ticker": pl.Series([], dtype=pl.Utf8),
        "added_company": pl.Series([], dtype=pl.Utf8),
        "removed_ticker": pl.Series([], dtype=pl.Utf8),
        "removed_company": pl.Series([], dtype=pl.Utf8),
    })
    history = build_universe_history(constituents, changes)
    tickers = history["ticker"].to_list()
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    # both should have the default inception entry date
    aapl_row = history.filter(pl.col("ticker") == "AAPL")
    assert aapl_row["entry_date"][0] == date(1993, 1, 29)
    assert aapl_row["exit_date"][0] is None


def test_parse_tables_extracts_multiple_tables() -> None:
    html = """
    <html><body>
      <table>
        <tr><th>Ticker</th><th>Company</th></tr>
        <tr><td>AAPL</td><td>Apple</td></tr>
      </table>
      <table>
        <tr><th>Date</th><th>Added</th></tr>
        <tr><td>January 3, 2020</td><td>TSLA</td></tr>
      </table>
    </body></html>
    """
    tables = _parse_tables(html)
    assert len(tables) == 2
    assert tables[0][1] == ["AAPL", "Apple"]
    assert tables[1][1] == ["January 3, 2020", "TSLA"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("January 3, 2020", date(2020, 1, 3)),
        ("2020-01-03", date(2020, 1, 3)),
        ("Jan 3, 2020", date(2020, 1, 3)),
        ("2020", None),
        ("-", None),
        ("", None),
    ],
)
def test_parse_date_variants(raw: str, expected: date | None) -> None:
    assert _parse_date(raw) == expected


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_fetch_sp500_constituents_parses_table(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <table>
      <tr>
        <th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Sub Industry</th>
        <th>HQ</th><th>CIK</th><th>Date added</th>
      </tr>
      <tr>
        <td>AAPL</td><td>Apple Inc.</td><td>IT</td><td>Hardware</td>
        <td>Cupertino</td><td>0000320193</td><td>1982-11-30</td>
      </tr>
      <tr><td>SHORT</td></tr>
    </table>
    """
    monkeypatch.setattr(
        "pelican.data.universe.httpx.get",
        lambda *args, **kwargs: _FakeResponse(html),
    )
    result = fetch_sp500_constituents()
    assert result.to_dict(as_series=False) == {
        "ticker": ["AAPL"],
        "company": ["Apple Inc."],
        "date_added": [date(1982, 11, 30)],
    }


def test_fetch_sp500_constituents_raises_when_no_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pelican.data.universe.httpx.get",
        lambda *args, **kwargs: _FakeResponse("<html><body>No tables</body></html>"),
    )
    with pytest.raises(RuntimeError, match="No tables found"):
        fetch_sp500_constituents()


def test_fetch_sp500_changes_parses_table(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <table>
      <tr><th>Symbol</th></tr>
      <tr><td>AAPL</td></tr>
    </table>
    <table>
      <tr>
        <th>Date</th><th>Added Symbol</th><th>Added Security</th>
        <th>Removed Symbol</th><th>Removed Security</th>
      </tr>
      <tr>
        <td>January 3, 2020</td><td>TSLA</td><td>Tesla</td><td>OLD</td><td>Old Co</td>
      </tr>
      <tr><td>bad row</td><td>ONLY</td></tr>
    </table>
    """
    monkeypatch.setattr(
        "pelican.data.universe.httpx.get",
        lambda *args, **kwargs: _FakeResponse(html),
    )
    result = fetch_sp500_changes()
    assert result.to_dict(as_series=False) == {
        "date": [date(2020, 1, 3)],
        "added_ticker": ["TSLA"],
        "added_company": ["Tesla"],
        "removed_ticker": ["OLD"],
        "removed_company": ["Old Co"],
    }


def test_build_universe_history_respects_explicit_date_added() -> None:
    constituents = pl.DataFrame(
        {
            "ticker": ["AAPL"],
            "company": ["Apple"],
            "date_added": [date(1982, 11, 30)],
        }
    )
    changes = pl.DataFrame(
        {
            "date": pl.Series([], dtype=pl.Date),
            "added_ticker": pl.Series([], dtype=pl.Utf8),
            "added_company": pl.Series([], dtype=pl.Utf8),
            "removed_ticker": pl.Series([], dtype=pl.Utf8),
            "removed_company": pl.Series([], dtype=pl.Utf8),
        }
    )
    history = build_universe_history(constituents, changes)
    assert history["entry_date"][0] == date(1982, 11, 30)


def test_build_universe_history_handles_removed_ticker_seen_only_in_changes() -> None:
    constituents = pl.DataFrame({"ticker": [], "company": [], "date_added": []}, schema={
        "ticker": pl.Utf8,
        "company": pl.Utf8,
        "date_added": pl.Date,
    })
    changes = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "added_ticker": [""],
            "added_company": [""],
            "removed_ticker": ["LEGACY"],
            "removed_company": ["Legacy Co"],
        }
    )
    history = build_universe_history(constituents, changes)
    assert history.to_dict(as_series=False) == {
        "ticker": ["LEGACY"],
        "entry_date": [date(1993, 1, 29)],
        "exit_date": [date(2020, 1, 3)],
        "company": ["Legacy Co"],
    }


def test_build_universe_history_creates_reentry_windows() -> None:
    constituents = pl.DataFrame(
        {
            "ticker": ["FLEX"],
            "company": ["Flex"],
            "date_added": [date(2021, 1, 1)],
        }
    )
    changes = pl.DataFrame(
        {
            "date": [date(2015, 1, 1), date(2018, 1, 1), date(2021, 1, 1)],
            "added_ticker": ["FLEX", "", "FLEX"],
            "added_company": ["Flex", "", "Flex"],
            "removed_ticker": ["", "FLEX", ""],
            "removed_company": ["", "Flex", ""],
        }
    )
    history = build_universe_history(constituents, changes)
    flex = history.filter(pl.col("ticker") == "FLEX").sort("entry_date")
    assert flex["entry_date"].to_list() == [date(2015, 1, 1), date(2021, 1, 1)]
    assert flex["exit_date"].to_list() == [date(2018, 1, 1), None]


# ---------------------------------------------------------------------------
# Returns computation tests
# ---------------------------------------------------------------------------

def test_log_return_correctness() -> None:
    df = _price_series("AAPL", n=5)
    result = compute_returns(df)
    # row 0: no previous → NaN
    assert result["log_return_1d"][0] is None or math.isnan(result["log_return_1d"][0])
    # row 1: log(101/100) ≈ 0.00995
    expected = math.log(101.0 / 100.0)
    assert abs(result["log_return_1d"][1] - expected) < 1e-9


def test_log_return_uses_past_only() -> None:
    """log_return_1d at position i must equal log(close[i] / close[i-1])."""
    df = _price_series("AAPL", n=10)
    result = compute_returns(df)
    closes = result["close"].to_list()
    returns = result["log_return_1d"].to_list()
    for i in range(1, len(closes)):
        expected = math.log(closes[i] / closes[i - 1])
        assert abs(returns[i] - expected) < 1e-9, f"Mismatch at row {i}"


def test_forward_return_correctness() -> None:
    df = _price_series("AAPL", n=30)
    result = compute_returns(df)
    closes = result["close"].to_list()
    fwds = result["forward_return_21d"].to_list()
    # row 0: close=100, close[21]=121 → fwd = (121/100) - 1 = 0.21
    expected = (closes[21] / closes[0]) - 1.0
    assert abs(fwds[0] - expected) < 1e-9


def test_forward_return_nan_at_tail() -> None:
    """Last 21 rows must have NaN forward returns."""
    df = _price_series("AAPL", n=40)
    result = compute_returns(df)
    fwds = result["forward_return_21d"].to_list()
    for i in range(len(fwds) - 21, len(fwds)):
        assert fwds[i] is None or math.isnan(fwds[i]), f"Expected NaN at row {i}"


def test_forward_return_label_not_nan_before_tail() -> None:
    """Rows before the last 21 must have non-NaN forward returns."""
    df = _price_series("AAPL", n=40)
    result = compute_returns(df)
    fwds = result["forward_return_21d"].to_list()
    for i in range(0, len(fwds) - 21):
        assert fwds[i] is not None and not math.isnan(fwds[i]), f"Unexpected NaN at row {i}"


def test_returns_multi_ticker_boundaries() -> None:
    """compute_returns must compute per-ticker, not bleed across tickers."""
    df_a = _price_series("AAPL", n=30)
    df_b = _price_series("MSFT", n=30, start=date(2021, 1, 2))
    df = pl.concat([df_a, df_b])
    result = compute_returns(df)
    # First row of each ticker should have NaN log_return (no prior close)
    for ticker in ["AAPL", "MSFT"]:
        first = result.filter(pl.col("ticker") == ticker).sort("date").head(1)
        val = first["log_return_1d"][0]
        assert val is None or math.isnan(val), f"Expected NaN for first row of {ticker}"


def test_compute_returns_sorts_unsorted_input() -> None:
    df = pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "date": [date(2020, 1, 3), date(2020, 1, 1), date(2020, 1, 2)],
            "open": [102.0, 100.0, 101.0],
            "high": [102.0, 100.0, 101.0],
            "low": [102.0, 100.0, 101.0],
            "close": [102.0, 100.0, 101.0],
            "volume": [1, 1, 1],
        }
    )
    result = compute_returns(df)
    assert result["date"].to_list() == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    assert abs(result["log_return_1d"][2] - math.log(102.0 / 101.0)) < 1e-9


def test_download_prices_multi_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2020-01-02", periods=2, name="Date")
    columns = pd.MultiIndex.from_product(
        [["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Volume"]]
    )
    raw = pd.DataFrame(
        [
            [100, 101, 99, 100, 10, 200, 201, 199, 200, 20],
            [101, 102, 100, 101, 11, 201, 202, 200, 201, 21],
        ],
        index=index,
        columns=columns,
    )
    monkeypatch.setattr("pelican.data.prices.yf.download", lambda *args, **kwargs: raw)
    result = download_prices(["AAPL", "MSFT"], date(2020, 1, 1), date(2020, 1, 10))
    assert set(result["ticker"].unique().to_list()) == {"AAPL", "MSFT"}
    assert len(result) == 4


def test_download_prices_single_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [10, 11],
        },
        index=pd.date_range("2020-01-02", periods=2, name="Date"),
    )
    monkeypatch.setattr("pelican.data.prices.yf.download", lambda *args, **kwargs: raw)
    result = download_prices(["AAPL"], date(2020, 1, 1), date(2020, 1, 10))
    assert result["ticker"].to_list() == ["AAPL", "AAPL"]
    assert result["close"].to_list() == [100.5, 101.5]


def test_download_prices_skips_missing_ticker_in_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2020-01-02", periods=2, name="Date")
    columns = pd.MultiIndex.from_product([["AAPL"], ["Open", "High", "Low", "Close", "Volume"]])
    raw = pd.DataFrame(
        [[100, 101, 99, 100, 10], [101, 102, 100, 101, 11]],
        index=index,
        columns=columns,
    )
    monkeypatch.setattr("pelican.data.prices.yf.download", lambda *args, **kwargs: raw)
    result = download_prices(["AAPL", "MSFT"], date(2020, 1, 1), date(2020, 1, 10))
    assert result["ticker"].unique().to_list() == ["AAPL"]


def test_download_prices_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pelican.data.prices.yf.download", lambda *args, **kwargs: pd.DataFrame())
    result = download_prices(["AAPL"], date(2020, 1, 1), date(2020, 1, 10))
    assert result.is_empty()
    assert result.columns == ["ticker", "date", "open", "high", "low", "close", "volume"]


def test_download_prices_provider_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr("pelican.data.prices.yf.download", _boom)
    result = download_prices(["AAPL"], date(2020, 1, 1), date(2020, 1, 10))
    assert result.is_empty()


# ---------------------------------------------------------------------------
# Prices schema + persistence tests
# ---------------------------------------------------------------------------

def test_prices_schema(store: DataStore) -> None:
    df = _price_series("AAPL", n=30)
    result = compute_returns(df)
    store.write(result, "prices")
    stored = store.query("SELECT * FROM prices WHERE ticker = 'AAPL' ORDER BY date")
    expected_cols = {"ticker", "date", "open", "high", "low", "close", "volume",
                     "log_return_1d", "forward_return_21d"}
    assert expected_cols == set(stored.columns)
    assert stored["ticker"].dtype == pl.Utf8
    assert stored["date"].dtype == pl.Date
    assert stored["close"].dtype == pl.Float64


def test_delisted_ticker_preserved(store: DataStore) -> None:
    """A ticker with only 30 rows is stored fully — no silent drop."""
    df = compute_returns(_price_series("GONE", n=30))
    store.write(df, "prices")
    count = store.query("SELECT count(*) AS n FROM prices WHERE ticker = 'GONE'").item(0, 0)
    assert count == 30


def test_get_prices_date_bounds(store: DataStore) -> None:
    df = compute_returns(_price_series("AAPL", n=60))
    store.write(df, "prices")
    start, end = date(2020, 1, 10), date(2020, 1, 20)
    result = get_prices(["AAPL"], start, end, store)
    assert result["date"].min() >= start
    assert result["date"].max() <= end


def test_get_prices_ticker_filter(store: DataStore) -> None:
    for ticker in ["AAPL", "MSFT", "GOOG"]:
        store.write(compute_returns(_price_series(ticker, n=30)), "prices")
    result = get_prices(["AAPL", "MSFT"], date(2020, 1, 1), date(2022, 1, 1), store)
    assert set(result["ticker"].unique().to_list()) == {"AAPL", "MSFT"}


def test_get_panel_date_bounds(store: DataStore) -> None:
    """get_panel returns no rows outside [start, end]."""
    for ticker in ["AAPL", "MSFT"]:
        store.write(compute_returns(_price_series(ticker, n=60)), "prices")

    # Seed a minimal universe so get_panel can join
    uni = _universe_df(
        ticker=["AAPL", "MSFT"],
        entry_date=[date(2020, 1, 1), date(2020, 1, 1)],
        exit_date=[None, None],
        company=["Apple", "Microsoft"],
    )
    store.write(uni, "sp500_universe")

    start, end = date(2020, 1, 10), date(2020, 1, 20)
    panel = get_panel(start, end, store)
    assert panel["date"].min() >= start
    assert panel["date"].max() <= end
    assert set(panel["ticker"].unique().to_list()) == {"AAPL", "MSFT"}


def test_get_panel_excludes_same_day_exit(store: DataStore) -> None:
    store.write(compute_returns(_price_series("EXIT", n=10)), "prices")
    store.write(
        _universe_df(
            ticker=["EXIT"],
            entry_date=[date(2020, 1, 1)],
            exit_date=[date(2020, 1, 5)],
            company=["Exit Co"],
        ),
        "sp500_universe",
    )
    panel = get_panel(date(2020, 1, 5), date(2020, 1, 5), store)
    assert panel.is_empty()


def test_get_panel_honors_reentry_windows(store: DataStore) -> None:
    store.write(compute_returns(_price_series("FLEX", n=30)), "prices")
    store.write(
        _universe_df(
            ticker=["FLEX", "FLEX"],
            entry_date=[date(2020, 1, 1), date(2020, 1, 20)],
            exit_date=[date(2020, 1, 10), None],
            company=["Flex", "Flex"],
        ),
        "sp500_universe",
    )
    panel = get_panel(date(2020, 1, 1), date(2020, 1, 25), store)
    included_dates = panel["date"].to_list()
    assert date(2020, 1, 15) not in included_dates
    assert date(2020, 1, 5) in included_dates
    assert date(2020, 1, 20) in included_dates


def test_get_panel_column_projection(store: DataStore) -> None:
    store.write(compute_returns(_price_series("AAPL", n=30)), "prices")
    store.write(
        _universe_df(
            ticker=["AAPL"],
            entry_date=[date(2020, 1, 1)],
            exit_date=[None],
            company=["Apple"],
        ),
        "sp500_universe",
    )
    panel = get_panel(date(2020, 1, 1), date(2020, 1, 10), store, columns=["ticker", "date", "close"])
    assert panel.columns == ["ticker", "date", "close"]
