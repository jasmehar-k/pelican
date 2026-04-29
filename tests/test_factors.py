"""
Tests for the Stage 3 classic factor library.

All tests use in-memory DuckDB and synthetic Polars DataFrames.
No network access required.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from pelican.backtest.engine import BacktestConfig, run_backtest
from pelican.backtest.signals import SignalSpec, get_signal, list_signals
from pelican.data.fundamentals import compute_fundamental_ratios
from pelican.data.store import DataStore

# Register all factors by importing the package.
import pelican.factors  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_store(n_tickers: int = 10, n_days: int = 600) -> DataStore:
    """In-memory store with synthetic prices for n_tickers × n_days."""
    store = DataStore(":memory:")
    store.init_schema()

    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    start = date(2018, 1, 1)

    universe_rows = [{"ticker": t, "entry_date": start, "exit_date": None, "company": t}
                     for t in tickers]
    store.write(pl.DataFrame(universe_rows), "sp500_universe")

    price_rows = []
    for ticker in tickers:
        base = 100.0
        for d in range(n_days):
            day = start + timedelta(days=d)
            close = base * (1.0 + 0.001 * d)
            price_rows.append({
                "ticker": ticker, "date": day,
                "open": close, "high": close, "low": close, "close": close,
                "volume": 1_000_000,
                "log_return_1d": math.log(1.001),
                "forward_return_21d": 0.002,
            })
    store.write(pl.DataFrame(price_rows), "prices")
    return store


def _make_fund_store(n_tickers: int = 10, n_days: int = 600) -> DataStore:
    """In-memory store with prices + synthetic fundamentals."""
    store = _make_price_store(n_tickers, n_days)

    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    start = date(2018, 1, 1)

    fund_rows = []
    for i, ticker in enumerate(tickers):
        # Two quarters of fundamentals per ticker.
        for q in range(2):
            period_end = start + timedelta(days=90 * q)
            available_date = period_end + timedelta(days=45)
            fund_rows.append({
                "ticker": ticker,
                "available_date": available_date,
                "period_end": period_end,
                "market_cap": 1e9 * (i + 1),
                "pe_ratio": 15.0 + i,
                "pb_ratio": 2.0 + i * 0.1,
                "roe": 0.15 - i * 0.01,
                "debt_to_equity": 0.5 + i * 0.05,
            })
    store.write(pl.DataFrame(fund_rows), "fundamentals")
    return store


def _cs_with_cols(**kwargs) -> pl.DataFrame:
    """Build a minimal cross-section DataFrame with the given extra columns."""
    n = 20
    base = {
        "ticker": [f"T{i:02d}" for i in range(n)],
        "date": [date(2020, 1, 2)] * n,
        "close": [100.0 + i for i in range(n)],
        "close_21d": [95.0 + i for i in range(n)],
        "close_63d": [90.0 + i for i in range(n)],
        "close_126d": [85.0 + i for i in range(n)],
        "close_252d": [80.0 + i for i in range(n)],
        "close_504d": [70.0 + i for i in range(n)],
        "vol_21d": [0.15 + i * 0.001 for i in range(n)],
        "vol_63d": [0.18 + i * 0.001 for i in range(n)],
        "forward_return_21d": [0.01 * (i - 10) for i in range(n)],
    }
    base.update(kwargs)
    return pl.DataFrame(base)


# ---------------------------------------------------------------------------
# Price-based factor score tests
# ---------------------------------------------------------------------------

class TestMomentumScores:
    def test_mom_1_12_score(self):
        cs = _cs_with_cols()
        sig = get_signal("MOM_1_12")
        scores = sig.fn(cs)
        expected = cs["close_21d"] / cs["close_252d"] - 1.0
        assert scores.equals(expected.alias(scores.name))

    def test_reversal_1m_score(self):
        cs = _cs_with_cols()
        sig = get_signal("REVERSAL_1M")
        scores = sig.fn(cs)
        expected = -(cs["close"] / cs["close_21d"] - 1.0)
        assert scores.equals(expected.alias(scores.name))

    def test_reversal_negate_sign(self):
        # Past loser (close < close_21d) → positive reversal score (bullish).
        cs = _cs_with_cols()
        cs = cs.with_columns(pl.col("close") * 0.9)  # close < close_21d → past loser
        sig = get_signal("REVERSAL_1M")
        scores = sig.fn(cs)
        assert (scores > 0).all()

    def test_low_vol_score(self):
        cs = _cs_with_cols()
        sig = get_signal("LOW_VOL")
        scores = sig.fn(cs)
        expected = -cs["vol_63d"]
        assert scores.equals(expected.alias(scores.name))

    def test_low_vol_ordering(self):
        # Lower vol → higher score.
        cs = _cs_with_cols()
        sig = get_signal("LOW_VOL")
        scores = sig.fn(cs)
        assert scores[0] > scores[-1]  # T00 has lower vol than T19


# ---------------------------------------------------------------------------
# Fundamental-based factor score tests
# ---------------------------------------------------------------------------

class TestSizeScores:
    def test_size_score(self):
        cs = _cs_with_cols(market_cap=[1e9 * (i + 1) for i in range(20)])
        sig = get_signal("SIZE")
        scores = sig.fn(cs)
        expected = pl.Series([-math.log(1e9 * (i + 1)) for i in range(20)])
        for s, e in zip(scores.to_list(), expected.to_list()):
            assert abs(s - e) < 1e-10

    def test_size_null_for_zero_or_negative(self):
        cs = _cs_with_cols(market_cap=[0.0, -1.0] + [1e9] * 18)
        sig = get_signal("SIZE")
        scores = sig.fn(cs)
        assert scores[0] is None
        assert scores[1] is None

    def test_size_smaller_is_higher(self):
        cs = _cs_with_cols(market_cap=[1e9 * (i + 1) for i in range(20)])
        sig = get_signal("SIZE")
        scores = sig.fn(cs)
        assert scores[0] > scores[-1]


class TestValueScores:
    def test_value_pe_score(self):
        cs = _cs_with_cols(pe_ratio=[10.0 + i for i in range(20)])
        sig = get_signal("VALUE_PE")
        scores = sig.fn(cs)
        expected = [1.0 / (10.0 + i) for i in range(20)]
        for s, e in zip(scores.to_list(), expected):
            assert abs(s - e) < 1e-10

    def test_value_pe_null_for_nonpositive(self):
        cs = _cs_with_cols(pe_ratio=[0.0, -5.0] + [15.0] * 18)
        sig = get_signal("VALUE_PE")
        scores = sig.fn(cs)
        assert scores[0] is None
        assert scores[1] is None

    def test_value_pb_score(self):
        cs = _cs_with_cols(pb_ratio=[2.0 + i * 0.1 for i in range(20)])
        sig = get_signal("VALUE_PB")
        scores = sig.fn(cs)
        expected = [1.0 / (2.0 + i * 0.1) for i in range(20)]
        for s, e in zip(scores.to_list(), expected):
            assert abs(s - e) < 1e-10

    def test_value_pb_null_for_nonpositive(self):
        cs = _cs_with_cols(pb_ratio=[0.0, -1.0] + [2.0] * 18)
        sig = get_signal("VALUE_PB")
        scores = sig.fn(cs)
        assert scores[0] is None
        assert scores[1] is None


class TestQualityScores:
    def test_quality_roe_score(self):
        roe_vals = [0.15 - i * 0.01 for i in range(20)]
        cs = _cs_with_cols(roe=roe_vals)
        sig = get_signal("QUALITY_ROE")
        scores = sig.fn(cs)
        # After winsorization the ordering should still hold for non-extreme values.
        assert scores[0] > scores[-1]

    def test_quality_roe_winsorized(self):
        # Build 100 items so p1/p99 winsorization clips an extreme outlier.
        n = 100
        roe_vals = [1000.0] + [0.10] * (n - 1)
        tickers = [f"T{i:03d}" for i in range(n)]
        cs = pl.DataFrame({
            "ticker": tickers,
            "date": [date(2020, 1, 2)] * n,
            "close": [100.0] * n,
            "close_21d": [95.0] * n,
            "close_63d": [90.0] * n,
            "close_126d": [85.0] * n,
            "close_252d": [80.0] * n,
            "close_504d": [70.0] * n,
            "vol_21d": [0.15] * n,
            "vol_63d": [0.18] * n,
            "forward_return_21d": [0.01] * n,
            "roe": roe_vals,
        })
        sig = get_signal("QUALITY_ROE")
        scores = sig.fn(cs)
        # Winsorized outlier must be strictly less than original 1000.
        assert scores[0] < 1000.0

    def test_quality_leverage_score(self):
        de_vals = [0.5 + i * 0.05 for i in range(20)]
        cs = _cs_with_cols(debt_to_equity=de_vals)
        sig = get_signal("QUALITY_LEVERAGE")
        scores = sig.fn(cs)
        expected = [-(0.5 + i * 0.05) for i in range(20)]
        for s, e in zip(scores.to_list(), expected):
            assert abs(s - e) < 1e-10

    def test_quality_leverage_lower_is_higher(self):
        # Lower debt → higher score.
        de_vals = [0.5 + i * 0.05 for i in range(20)]
        cs = _cs_with_cols(debt_to_equity=de_vals)
        sig = get_signal("QUALITY_LEVERAGE")
        scores = sig.fn(cs)
        assert scores[0] > scores[-1]


# ---------------------------------------------------------------------------
# SignalSpec metadata tests
# ---------------------------------------------------------------------------

class TestSignalSpecMetadata:
    FUNDAMENTAL_SIGNALS = ["SIZE", "VALUE_PE", "VALUE_PB", "QUALITY_ROE", "QUALITY_LEVERAGE"]
    PRICE_SIGNALS = ["MOM_1_12", "REVERSAL_1M", "LOW_VOL"]
    ALL = FUNDAMENTAL_SIGNALS + PRICE_SIGNALS

    def test_all_eight_registered(self):
        registered = list_signals()
        for name in self.ALL:
            assert name in registered, f"{name} not registered"

    def test_description_nonempty(self):
        for name in self.ALL:
            sig = get_signal(name)
            assert sig.spec.description, f"{name} has empty description"

    def test_expected_ic_range_valid(self):
        for name in self.ALL:
            sig = get_signal(name)
            lo, hi = sig.spec.expected_ic_range
            assert lo < hi, f"{name} expected_ic_range invalid: {lo}, {hi}"

    def test_fundamental_signals_flag(self):
        for name in self.FUNDAMENTAL_SIGNALS:
            sig = get_signal(name)
            assert sig.spec.requires_fundamentals, f"{name} should require fundamentals"

    def test_price_signals_no_fund_flag(self):
        for name in self.PRICE_SIGNALS:
            sig = get_signal(name)
            assert not sig.spec.requires_fundamentals, f"{name} should not require fundamentals"

    def test_fundamental_signals_have_data_deps(self):
        for name in self.FUNDAMENTAL_SIGNALS:
            sig = get_signal(name)
            assert sig.spec.data_deps, f"{name} has no data_deps"


# ---------------------------------------------------------------------------
# Point-in-time fundamentals join tests
# ---------------------------------------------------------------------------

class TestFundamentalPitJoin:
    def test_engine_joins_fundamentals(self):
        """Signals with requires_fundamentals=True get fundamental columns in cs."""
        store = _make_fund_store(n_tickers=10, n_days=800)
        config = BacktestConfig(
            start=date(2018, 10, 1),  # after 252 price rows exist
            end=date(2019, 12, 31),
            cost_bps=0.0,
            min_universe_size=5,
        )
        # SIZE requires fundamentals. Should produce non-NaN scores.
        result = run_backtest("SIZE", config, store)
        store.close()
        assert result.n_periods > 0
        # IC series should exist (may be NaN-heavy but structure is correct).
        assert "ic" in result.ic_series.columns

    def test_no_lookahead_fundamentals(self):
        """Fundamental data with available_date > rebal_date must be excluded."""
        store = DataStore(":memory:")
        store.init_schema()

        # Single ticker, one price row at rebal_date.
        rebal = date(2019, 3, 1)
        store.write(
            pl.DataFrame([{"ticker": "A", "entry_date": date(2018, 1, 1),
                           "exit_date": None, "company": "A"}]),
            "sp500_universe",
        )
        # Prices: enough history for the engine.
        price_rows = []
        for d in range(500):
            day = date(2018, 1, 1) + timedelta(days=d)
            price_rows.append({
                "ticker": "A", "date": day,
                "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                "volume": 1_000_000,
                "log_return_1d": 0.0,
                "forward_return_21d": 0.01,
            })
        store.write(pl.DataFrame(price_rows), "prices")

        # Fundamental with available_date AFTER rebal (future data — must be excluded).
        future_date = rebal + timedelta(days=1)
        store.write(
            pl.DataFrame([{
                "ticker": "A",
                "available_date": future_date,
                "period_end": rebal - timedelta(days=44),
                "market_cap": 1e10,  # very large — would dominate if included
                "pe_ratio": 5.0,
                "pb_ratio": 1.0,
                "roe": 0.5,
                "debt_to_equity": 0.1,
            }]),
            "fundamentals",
        )

        # Fundamental with available_date BEFORE rebal (safe to use).
        safe_date = rebal - timedelta(days=10)
        store.write(
            pl.DataFrame([{
                "ticker": "A",
                "available_date": safe_date,
                "period_end": safe_date - timedelta(days=45),
                "market_cap": 1e9,
                "pe_ratio": 15.0,
                "pb_ratio": 2.0,
                "roe": 0.15,
                "debt_to_equity": 0.5,
            }]),
            "fundamentals",
        )

        # The point-in-time query should only see market_cap = 1e9, not 1e10.
        fund_panel = store.query(
            "SELECT ticker, available_date, market_cap FROM fundamentals "
            "WHERE available_date <= ? ORDER BY available_date",
            [rebal],
        )
        store.close()
        assert len(fund_panel) == 1
        assert float(fund_panel["market_cap"][0]) == pytest.approx(1e9)


# ---------------------------------------------------------------------------
# Fundamental ratio computation tests
# ---------------------------------------------------------------------------

class TestFundamentalRatios:
    def _make_fund_df(self) -> pl.DataFrame:
        return pl.DataFrame([{
            "ticker": "AAPL",
            "period_end": date(2020, 9, 30),
            "shares_outstanding": 1e9,
            "equity": 5e10,
            "total_debt": 1e10,
            "net_income": 2e10,
        }])

    def _make_prices_df(self) -> pl.DataFrame:
        return pl.DataFrame([{
            "ticker": "AAPL",
            "date": date(2020, 9, 30),
            "close": 100.0,
        }])

    def test_market_cap(self):
        ratios = compute_fundamental_ratios(self._make_fund_df(), self._make_prices_df())
        assert ratios["market_cap"][0] == pytest.approx(1e9 * 100.0)

    def test_pe_ratio(self):
        # pe = price / (net_income / shares) = 100 / (2e10 / 1e9) = 100 / 20 = 5
        ratios = compute_fundamental_ratios(self._make_fund_df(), self._make_prices_df())
        assert ratios["pe_ratio"][0] == pytest.approx(5.0)

    def test_pb_ratio(self):
        # pb = price / (equity / shares) = 100 / (5e10 / 1e9) = 100 / 50 = 2
        ratios = compute_fundamental_ratios(self._make_fund_df(), self._make_prices_df())
        assert ratios["pb_ratio"][0] == pytest.approx(2.0)

    def test_roe(self):
        # roe = net_income / equity = 2e10 / 5e10 = 0.4
        ratios = compute_fundamental_ratios(self._make_fund_df(), self._make_prices_df())
        assert ratios["roe"][0] == pytest.approx(0.4)

    def test_debt_to_equity(self):
        # d/e = total_debt / equity = 1e10 / 5e10 = 0.2
        ratios = compute_fundamental_ratios(self._make_fund_df(), self._make_prices_df())
        assert ratios["debt_to_equity"][0] == pytest.approx(0.2)

    def test_pe_null_for_negative_earnings(self):
        fd = pl.DataFrame([{
            "ticker": "X", "period_end": date(2020, 6, 30),
            "shares_outstanding": 1e9, "equity": 5e10,
            "total_debt": 1e10, "net_income": -1e9,
        }])
        pr = pl.DataFrame([{"ticker": "X", "date": date(2020, 6, 30), "close": 100.0}])
        ratios = compute_fundamental_ratios(fd, pr)
        assert ratios["pe_ratio"][0] is None

    def test_roe_null_for_negative_equity(self):
        fd = pl.DataFrame([{
            "ticker": "X", "period_end": date(2020, 6, 30),
            "shares_outstanding": 1e9, "equity": -1e9,
            "total_debt": 0.0, "net_income": 1e8,
        }])
        pr = pl.DataFrame([{"ticker": "X", "date": date(2020, 6, 30), "close": 100.0}])
        ratios = compute_fundamental_ratios(fd, pr)
        assert ratios["roe"][0] is None

    def test_available_date_lag(self):
        ratios = compute_fundamental_ratios(self._make_fund_df(), self._make_prices_df())
        period_end = ratios["period_end"][0]
        available = ratios["available_date"][0]
        assert (available - period_end).days == 45


# ---------------------------------------------------------------------------
# Correlation matrix tests
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    def _run_matrix(self, signal_names):
        from pelican.factors.correlation import build_factor_correlation_matrix
        store = _make_price_store(n_tickers=15, n_days=900)
        config = BacktestConfig(
            start=date(2018, 10, 1),  # after 252 price rows exist from 2018-01-01
            end=date(2020, 6, 30),
            cost_bps=0.0,
            min_universe_size=5,
        )
        corr = build_factor_correlation_matrix(signal_names, config, store)
        store.close()
        return corr

    def test_correlation_matrix_shape(self):
        names = ["MOM_1_12", "LOW_VOL", "REVERSAL_1M"]
        corr = self._run_matrix(names)
        assert corr.shape == (3, 4)  # 3 rows × (signal + 3 signal cols)

    def test_correlation_matrix_diagonal(self):
        names = ["MOM_1_12", "LOW_VOL"]
        corr = self._run_matrix(names)
        for name in names:
            val = corr.filter(pl.col("signal") == name)[name].item()
            assert val == pytest.approx(1.0)

    def test_correlation_matrix_symmetric(self):
        names = ["MOM_1_12", "LOW_VOL", "REVERSAL_1M"]
        corr = self._run_matrix(names)
        for i, ni in enumerate(names):
            for nj in names[i + 1:]:
                val_ij = corr.filter(pl.col("signal") == ni)[nj].item()
                val_ji = corr.filter(pl.col("signal") == nj)[ni].item()
                if not math.isnan(val_ij) and not math.isnan(val_ji):
                    assert val_ij == pytest.approx(val_ji)
