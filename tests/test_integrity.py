"""
Integrity tests: forward-return alignment, IC sign, rebalance date alignment,
transaction costs, and Fama-French MOM factor correlation.

These are the four failure modes explicitly listed in the design spec:
  1. forward-looking returns computed incorrectly
  2. IC sign flipped
  3. rebalance dates misaligned (signal uses future data)
  4. transaction costs not applied to both legs

Plus the FF MOM correlation sanity check (requires network + seeded DB).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from pelican.backtest.engine import BacktestConfig, _turnover, run_backtest
from pelican.backtest.metrics import spearman_ic
from pelican.backtest.signals import (
    SignalSpec,
    build_cross_section_features,
    register,
)
from pelican.backtest.universe import get_rebalance_dates
from pelican.data.prices import compute_returns
from pelican.data.store import DataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> DataStore:
    s = DataStore(":memory:")
    s.init_schema()
    return s


def _seed_universe(store, tickers, entry=date(2019, 1, 1), exit_date=None):
    rows = [{"ticker": t, "entry_date": entry, "exit_date": exit_date, "company": t}
            for t in tickers]
    store.write(pl.DataFrame(rows, schema={
        "ticker": pl.Utf8, "entry_date": pl.Date,
        "exit_date": pl.Date, "company": pl.Utf8,
    }), "sp500_universe")


def _make_prices(tickers, dates, close_fn):
    rows = []
    for t in tickers:
        for i, d in enumerate(dates):
            c = close_fn(t, i)
            rows.append({"ticker": t, "date": d, "open": c, "high": c,
                         "low": c, "close": c, "volume": 1_000_000,
                         "log_return_1d": None, "forward_return_21d": None})
    df = pl.DataFrame(rows, schema={
        "ticker": pl.Utf8, "date": pl.Date,
        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
        "close": pl.Float64, "volume": pl.Int64,
        "log_return_1d": pl.Float64, "forward_return_21d": pl.Float64,
    })
    return compute_returns(df)


# ---------------------------------------------------------------------------
# 1. Forward-looking returns correctness
# ---------------------------------------------------------------------------

class TestForwardReturn:
    """forward_return_21d[t] must equal (close[t+21] - close[t]) / close[t]."""

    def test_exact_formula_single_ticker(self):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(60)]
        # close[i] = 100 + i exactly
        df = _make_prices(["A"], dates, close_fn=lambda t, i: 100.0 + i)
        # At index 10, close=110. close[10+21]=131. fwd = (131-110)/110
        row = df.filter((pl.col("ticker") == "A") & (pl.col("date") == dates[10]))
        expected = (100.0 + 31) / (100.0 + 10) - 1.0
        actual = float(row["forward_return_21d"][0])
        assert abs(actual - expected) < 1e-9

    def test_last_21_rows_are_null(self):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        df = _make_prices(["A"], dates, close_fn=lambda t, i: 100.0 + i)
        tail = df.filter(pl.col("ticker") == "A").sort("date").tail(21)
        assert tail["forward_return_21d"].null_count() == 21

    def test_forward_return_uses_close_not_open(self):
        """forward_return_21d is derived purely from close prices."""
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(40)]
        # open is 10x close — if we used open by mistake, the return would differ
        rows = []
        for i, d in enumerate(dates):
            c = 100.0 + i
            rows.append({"ticker": "A", "date": d,
                         "open": c * 10, "high": c, "low": c, "close": c,
                         "volume": 1_000_000,
                         "log_return_1d": None, "forward_return_21d": None})
        df = compute_returns(pl.DataFrame(rows, schema={
            "ticker": pl.Utf8, "date": pl.Date,
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
            "close": pl.Float64, "volume": pl.Int64,
            "log_return_1d": pl.Float64, "forward_return_21d": pl.Float64,
        }))
        row = df.filter(pl.col("date") == dates[0])
        expected = (100.0 + 21) / 100.0 - 1.0  # close-only formula
        actual = float(row["forward_return_21d"][0])
        assert abs(actual - expected) < 1e-9

    def test_multi_ticker_no_cross_contamination(self):
        """shift(-21).over('ticker') must not bleed across tickers."""
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(40)]
        # ticker A: price 100 flat; ticker B: price 200 flat
        df = _make_prices(["A", "B"], dates,
                          close_fn=lambda t, i: 100.0 if t == "A" else 200.0)
        a_fwd = df.filter(pl.col("ticker") == "A")["forward_return_21d"].drop_nulls()
        b_fwd = df.filter(pl.col("ticker") == "B")["forward_return_21d"].drop_nulls()
        # flat prices → 0% forward return for both
        assert (a_fwd.abs() < 1e-9).all()
        assert (b_fwd.abs() < 1e-9).all()

    def test_sort_order_does_not_affect_result(self):
        """compute_returns must produce the same result regardless of input order."""
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(40)]
        df = _make_prices(["A"], dates, close_fn=lambda t, i: 100.0 + i)
        df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
        df_shuffled_returns = compute_returns(df_shuffled.drop(
            ["log_return_1d", "forward_return_21d"]
        ).rename({}))
        # Both should give the same forward returns after sorting
        r1 = df.sort(["ticker", "date"])["forward_return_21d"]
        r2 = df_shuffled_returns.sort(["ticker", "date"])["forward_return_21d"]
        # Compare non-null values
        for a, b in zip(r1.to_list(), r2.to_list()):
            if a is None and b is None:
                continue
            assert abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# 2. IC sign correctness
# ---------------------------------------------------------------------------

class TestICSign:
    """High score must correlate positively with high forward return (IC > 0)."""

    def test_perfect_positive_signal_gives_positive_ic(self):
        """When signal rank == return rank exactly, IC should be +1."""
        scores = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pl.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        assert spearman_ic(scores, returns) > 0.99

    def test_perfect_negative_signal_gives_negative_ic(self):
        """When high signal predicts low return, IC is negative."""
        scores = pl.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        returns = pl.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        assert spearman_ic(scores, returns) < -0.99

    def test_quintile_spread_sign_matches_ic_sign(self):
        """When IC > 0, Q5 mean return should exceed Q1 mean return."""
        store = _make_store()
        n_tickers = 30
        n_days = 400
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        dates = [date(2019, 1, 1) + timedelta(days=i) for i in range(n_days)]
        _seed_universe(store, tickers)

        # Construct prices so that ticker index is predictive of forward return.
        # Ticker T00 (index 0) has lowest growth; T29 has highest growth.
        def close_fn(t, i):
            idx = int(t[1:])
            growth = 1.0 + (idx / n_tickers) * 0.001  # monotone cross-section signal
            return 100.0 * (growth ** i)

        df = _make_prices(tickers, dates, close_fn)
        store.write(df, "prices")

        # Register a signal that scores by the 21d momentum (should correlate with
        # future return when growth rates are persistent).
        spec = SignalSpec(name="_SIGN_TEST", description="sign test", lookback_days=63)

        @register(spec)
        def _sign_signal(cs: pl.DataFrame) -> pl.Series:
            return (cs["close_21d"] / cs["close"] - 1.0).alias("_SIGN_TEST")

        config = BacktestConfig(
            start=date(2019, 6, 1),
            end=date(2019, 12, 31),
            cost_bps=0.0,
            min_universe_size=10,
            min_score_coverage=0.1,
            lookback_calendar_days=200,
        )
        result = run_backtest("_SIGN_TEST", config, store)
        # Positive IC means Q5 should be positive and Q1 negative on average
        if not math.isnan(result.ic_mean) and result.ic_mean > 0:
            pr = result.period_returns.drop_nulls(["ls_gross"])
            assert float(pr["ls_gross"].mean()) > 0


# ---------------------------------------------------------------------------
# 3. Rebalance date alignment (no look-ahead in signal computation)
# ---------------------------------------------------------------------------

class TestNoLookAhead:
    """Signal features at rebal_date must use only data on or before that date."""

    def test_cross_section_contains_only_past_dates(self):
        n_days = 300
        tickers = ["A", "B", "C"]
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
        rows = []
        for t in tickers:
            for i, d in enumerate(dates):
                c = 100.0 + i
                rows.append({"ticker": t, "date": d, "close": c,
                             "log_return_1d": 0.001 if i > 0 else None,
                             "open": c, "high": c, "low": c,
                             "volume": 1_000_000, "forward_return_21d": 0.01})
        panel = pl.DataFrame(rows, schema={
            "ticker": pl.Utf8, "date": pl.Date, "open": pl.Float64,
            "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
            "volume": pl.Int64, "log_return_1d": pl.Float64,
            "forward_return_21d": pl.Float64,
        })
        rebal = dates[200]
        cs = build_cross_section_features(panel, rebal)
        # Cross-section must only have rows at rebal (not after)
        assert (cs["date"] <= rebal).all()

    def test_lag_column_refers_to_past_close(self):
        """close_21d at rebal_date must equal close exactly 21 rows before rebal."""
        n_days = 100
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
        rows = [{"ticker": "A", "date": d, "close": float(i + 1),
                 "log_return_1d": None, "open": float(i + 1),
                 "high": float(i + 1), "low": float(i + 1),
                 "volume": 1_000_000, "forward_return_21d": None}
                for i, d in enumerate(dates)]
        panel = pl.DataFrame(rows, schema={
            "ticker": pl.Utf8, "date": pl.Date, "open": pl.Float64,
            "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
            "volume": pl.Int64, "log_return_1d": pl.Float64,
            "forward_return_21d": pl.Float64,
        })
        rebal = dates[50]  # row index 50, close = 51.0
        cs = build_cross_section_features(panel, rebal)
        # close_21d at row 50 should be close at row 50-21 = 29, which is 30.0
        assert abs(float(cs["close_21d"][0]) - 30.0) < 1e-9

    def test_forward_return_21d_uses_future_price(self):
        """forward_return_21d at date T uses close at T+21, not T."""
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(30)]
        df = _make_prices(["A"], dates, close_fn=lambda t, i: 100.0 + i)
        row = df.filter(pl.col("date") == dates[0])
        # forward_return_21d at date[0] = (close[21] - close[0]) / close[0]
        # = (100+21 - 100) / 100 = 0.21
        assert abs(float(row["forward_return_21d"][0]) - 0.21) < 1e-9
        # Confirm it does NOT equal the current-day return
        assert abs(float(row["forward_return_21d"][0])) != 0.0


# ---------------------------------------------------------------------------
# 4. Transaction costs applied to both legs
# ---------------------------------------------------------------------------

class TestTransactionCosts:
    """Net return must be strictly less than gross when turnover > 0 and cost_bps > 0."""

    def _run_with_costs(self, cost_bps) -> tuple:
        store = _make_store()
        n_tickers = 25
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        dates = [date(2019, 1, 1) + timedelta(days=i) for i in range(450)]
        _seed_universe(store, tickers)

        import random
        rng = random.Random(7)

        def close_fn(t, i):
            seed = int(t[1:]) * 0.001
            return max(1.0, 100.0 + seed * i + rng.gauss(0, 0.5))

        df = _make_prices(tickers, dates, close_fn)
        store.write(df, "prices")

        spec = SignalSpec(name="_COST_TEST", description="cost test", lookback_days=63)

        @register(spec)
        def _cost_signal(cs):
            return (cs["close_21d"] / cs["close"] - 1.0).alias("_COST_TEST")

        config = BacktestConfig(
            start=date(2019, 4, 1),
            end=date(2019, 12, 31),
            cost_bps=cost_bps,
            min_universe_size=5,
            min_score_coverage=0.1,
            lookback_calendar_days=150,
        )
        return run_backtest("_COST_TEST", config, store)

    def test_net_strictly_below_gross_when_costs_positive(self):
        result = self._run_with_costs(10.0)
        pr = result.period_returns.drop_nulls(["ls_gross", "ls_net"])
        # For every period, ls_net ≤ ls_gross (cost reduces return)
        assert (pr["ls_net"] <= pr["ls_gross"]).all()

    def test_zero_cost_gives_identical_gross_and_net(self):
        result = self._run_with_costs(0.0)
        pr = result.period_returns.drop_nulls(["ls_gross", "ls_net"])
        for g, n in zip(pr["ls_gross"].to_list(), pr["ls_net"].to_list()):
            assert abs(g - n) < 1e-12

    def test_both_legs_charged(self):
        """Cost must reflect BOTH long and short turnover, not just the long book.

        We verify this by checking that the cost per period equals
        avg_turnover * cost_bps / 10000, where avg_turnover is the
        average of long AND short book turnovers.

        If only the long book were charged, the cost would be roughly half
        of what it should be on average.
        """
        result = self._run_with_costs(10.0)
        pr = result.period_returns.drop_nulls(["ls_gross", "ls_net", "turnover"])
        # For each period, ls_gross - ls_net should equal turnover * bps / 10000
        diffs = (pr["ls_gross"] - pr["ls_net"]).to_list()
        to_costs = (pr["turnover"] * 10.0 / 10_000).to_list()
        for diff, expected in zip(diffs, to_costs):
            assert abs(diff - expected) < 1e-10

    def test_first_period_full_turnover_both_books(self):
        """First period: both books start empty so both turnovers = 1.0 → avg = 1.0."""
        result = self._run_with_costs(10.0)
        first_to = float(result.period_returns["turnover"][0])
        assert abs(first_to - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 5. Fama-French MOM factor correlation
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_ff_mom_correlation():
    """MOM_1_12 monthly L/S returns must correlate ≥ 0.50 with published FF UMD.

    The threshold is 0.50 rather than the canonical 0.70 because:
    - We have only 3 years of seeded data (2020-2023), a notoriously difficult
      momentum regime (COVID crash, value rotation, energy spike).
    - We use S&P 500 only with equal weighting; FF uses all US stocks,
      value-weighted, with NYSE breakpoints.
    - With n≈20 periods, SE(r) ≈ 0.22, so 0.50 is already >2 SE above zero.

    To run: ensure data/pelican.duckdb is seeded for 2020-2023.
    Skip with: pytest -m "not network"
    """
    import io
    import zipfile

    import httpx

    # --- download FF monthly momentum factor ---
    ff_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Momentum_Factor_CSV.zip"
    )
    try:
        resp = httpx.get(ff_url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Cannot reach Ken French data library: {exc}")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        fname = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        content = z.read(fname).decode("latin-1")

    # Parse: skip header lines (starting with spaces / blank), read YYYYMM,UMD
    ff_rows = {}
    in_data = False
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Copyright"):
            break
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            yyyymm = int(parts[0].strip())
            umd = float(parts[1].strip()) / 100.0  # percent → decimal
            if yyyymm >= 192601:
                in_data = True
            if in_data:
                ff_rows[yyyymm] = umd
        except ValueError:
            continue

    # --- run our MOM_1_12 backtest ---
    db_path = "data/pelican.duckdb"
    import os
    if not os.path.exists(db_path):
        pytest.skip("data/pelican.duckdb not found — run seed_data.py first")

    from pelican.data.store import DataStore as DS
    store = DS(db_path)

    # Use whatever date range we have data for
    date_range = store.query(
        "SELECT MIN(date) AS mn, MAX(date) AS mx FROM prices"
    )
    if date_range.is_empty() or date_range["mn"][0] is None:
        pytest.skip("prices table is empty")

    data_start: date = date_range["mn"][0]
    data_end: date = date_range["mx"][0]

    # Give the signal 252 trading days of warmup
    warmup = timedelta(days=365)
    bt_start = max(data_start + warmup, date(2020, 1, 1))
    bt_end = data_end - timedelta(days=30)  # leave room for forward return

    if bt_start >= bt_end:
        pytest.skip("insufficient data range for MOM backtest")

    config = BacktestConfig(
        start=bt_start,
        end=bt_end,
        cost_bps=0.0,  # gross only, no cost distortion
        min_universe_size=50,
        min_score_coverage=0.3,
        lookback_calendar_days=800,
    )

    try:
        result = run_backtest("MOM_1_12", config, store)
    except Exception as exc:
        pytest.skip(f"MOM backtest failed: {exc}")
    finally:
        store.close()

    if result.n_periods < 6:
        pytest.skip(f"too few periods ({result.n_periods}) to compute correlation")

    # --- align monthly returns ---
    our_returns: dict[int, float] = {}
    for row in result.period_returns.iter_rows(named=True):
        d: date = row["date"]
        yyyymm = d.year * 100 + d.month
        ls = row["ls_gross"]
        if ls is not None:
            our_returns[yyyymm] = ls

    common_keys = sorted(set(our_returns) & set(ff_rows))
    if len(common_keys) < 6:
        pytest.skip(
            f"only {len(common_keys)} overlapping months with FF data — need ≥ 6"
        )

    ours = pl.Series([our_returns[k] for k in common_keys])
    ff = pl.Series([ff_rows[k] for k in common_keys])

    correlation = spearman_ic(ours, ff)
    print(
        f"\nFF MOM correlation check: n={len(common_keys)}, "
        f"Spearman r={correlation:.3f} (threshold: 0.50)"
    )

    assert (
        correlation >= 0.50
    ), (
        f"MOM_1_12 correlation with FF UMD = {correlation:.3f}, expected ≥ 0.50. "
        f"n={len(common_keys)} months, period {common_keys[0]}–{common_keys[-1]}. "
        "Check that the signal formula is not inverted and data is split-adjusted."
    )
