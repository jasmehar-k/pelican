"""pytest suite for the data foundation layer. No network calls — synthetic data only."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from pelican.data.prices import compute_returns, get_panel, get_prices
from pelican.data.store import DataStore
from pelican.data.universe import build_universe_history, get_universe

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
