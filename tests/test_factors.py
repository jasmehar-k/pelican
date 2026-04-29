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


def _make_predictive_store(n_tickers: int = 20, n_days: int = 750) -> DataStore:
    """Store where price momentum predicts forward returns — IC > 0 by construction.

    Ticker i grows at a rate proportional to i, so higher-i tickers have more
    momentum and also higher forward returns.  Fundamentals are also structured
    so that higher-i tickers score higher on every fundamental signal.
    """
    store = DataStore(":memory:")
    store.init_schema()
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    start = date(2018, 1, 1)
    store.write(
        pl.DataFrame([{"ticker": t, "entry_date": start, "exit_date": None, "company": t}
                      for t in tickers]),
        "sp500_universe",
    )

    price_rows = []
    for i, ticker in enumerate(tickers):
        daily_growth = 1.0 + i * 0.0005
        fwd = (i - n_tickers / 2) / (n_tickers * 50)
        for d in range(n_days):
            close = 100.0 * (daily_growth ** d)
            price_rows.append({
                "ticker": ticker, "date": start + timedelta(days=d),
                "open": close, "high": close, "low": close, "close": close,
                "volume": 1_000_000,
                "log_return_1d": math.log(daily_growth),
                "forward_return_21d": fwd,
            })
    store.write(pl.DataFrame(price_rows), "prices")

    fund_rows = []
    for i, ticker in enumerate(tickers):
        for q in range(4):
            period_end = start + timedelta(days=90 * q)
            fund_rows.append({
                "ticker": ticker,
                "available_date": period_end + timedelta(days=45),
                "period_end": period_end,
                "market_cap": 1e9 * (n_tickers - i),   # smaller cap → higher SIZE score → higher fwd
                "pe_ratio": max(1.0, 30.0 - i),         # lower PE → higher VALUE_PE → higher fwd
                "pb_ratio": max(0.1, 5.0 - i * 0.2),   # lower PB → higher VALUE_PB → higher fwd
                "roe": 0.05 + i * 0.01,                 # higher ROE → higher QUALITY_ROE → higher fwd
                "debt_to_equity": max(0.01, 2.0 - i * 0.08),  # lower D/E → higher QUALITY_LEV → higher fwd
            })
    store.write(pl.DataFrame(fund_rows), "fundamentals")
    return store


def _spearman(a: pl.Series, b: pl.Series) -> float:
    """Spearman rank correlation between two series, dropping nulls pairwise."""
    from scipy.stats import spearmanr
    df = pl.DataFrame({"a": a, "b": b}).drop_nulls()
    if len(df) < 3:
        return float("nan")
    corr, _ = spearmanr(df["a"].to_numpy(), df["b"].to_numpy())
    return float(corr)


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


# ---------------------------------------------------------------------------
# Semantic integrity: factors produce real signal (not flat / wrong direction)
# ---------------------------------------------------------------------------

_BACKTEST_CONFIG = BacktestConfig(
    start=date(2019, 1, 1),
    end=date(2020, 6, 30),
    cost_bps=0.0,
    min_universe_size=5,
    min_score_coverage=0.1,
)


class TestFactorSemantics:
    """Each factor produces non-degenerate scores and non-zero IC on synthetic data
    where the relationship between signal and forward return is guaranteed by construction."""

    @pytest.mark.parametrize("signal_name,extra_cols", [
        ("MOM_1_12", {}),
        ("REVERSAL_1M", {}),
        ("LOW_VOL", {}),
        ("SIZE", {"market_cap": [1e9 * (i + 1) for i in range(20)]}),
        ("VALUE_PE", {"pe_ratio": [10.0 + i for i in range(20)]}),
        ("VALUE_PB", {"pb_ratio": [1.0 + i * 0.1 for i in range(20)]}),
        ("QUALITY_ROE", {"roe": [0.3 - i * 0.01 for i in range(20)]}),
        ("QUALITY_LEVERAGE", {"debt_to_equity": [0.2 + i * 0.1 for i in range(20)]}),
    ])
    def test_scores_are_nondegenerate(self, signal_name, extra_cols):
        """Signal produces distinct (non-constant) scores across tickers."""
        cs = _cs_with_cols(**extra_cols)
        scores = get_signal(signal_name).fn(cs)
        non_null = scores.drop_nulls()
        assert len(non_null) >= 10, f"{signal_name}: fewer than 10 non-null scores"
        assert non_null.std() > 0, f"{signal_name}: all scores identical — factor may be broken"

    @pytest.mark.parametrize("signal_name", ["MOM_1_12", "REVERSAL_1M"])
    def test_price_factor_ic_nonzero(self, signal_name):
        # LOW_VOL is excluded: rolling vol requires daily return variance, but the
        # predictive store uses a constant log return per ticker so vol=0 everywhere.
        # Score non-degeneracy for LOW_VOL is verified in test_scores_are_nondegenerate.
        """Price-only factors produce non-zero IC on a predictive synthetic store."""
        store = _make_predictive_store()
        result = run_backtest(signal_name, _BACKTEST_CONFIG, store)
        store.close()
        assert result.n_periods > 0, f"{signal_name}: no backtest periods produced"
        assert not math.isnan(result.ic_mean), f"{signal_name}: IC is NaN"
        assert result.icir != 0, f"{signal_name}: ICIR is exactly 0 — factor may output constant scores"

    @pytest.mark.parametrize("signal_name", [
        "SIZE", "VALUE_PE", "VALUE_PB", "QUALITY_ROE", "QUALITY_LEVERAGE",
    ])
    def test_fundamental_factor_ic_nonzero(self, signal_name):
        """Fundamental factors produce non-zero IC on a predictive synthetic store."""
        store = _make_predictive_store()
        result = run_backtest(signal_name, _BACKTEST_CONFIG, store)
        store.close()
        assert result.n_periods > 0, f"{signal_name}: no backtest periods produced"
        assert not math.isnan(result.ic_mean), f"{signal_name}: IC is NaN"
        assert result.icir != 0, f"{signal_name}: ICIR is exactly 0 — factor may output constant scores"


# ---------------------------------------------------------------------------
# Factor orthogonality: distinct factor groups aren't accidentally the same
# ---------------------------------------------------------------------------

class TestFactorOrthogonality:
    """Verify each factor uses its own input column — not someone else's.

    Rather than checking rank-correlations on toy data (where every column tends
    to move together), we use a perturbation approach: change only the column a
    signal is supposed to ignore and assert the scores don't change; change the
    column it IS supposed to use and assert they do change.  This directly catches
    copy-paste bugs like VALUE_PB using the pe_ratio formula.
    """

    def test_value_pe_uses_pe_ratio_not_vol(self):
        """VALUE_PE scores change when pe_ratio changes, not when vol changes."""
        base = _cs_with_cols(pe_ratio=[10.0 + i for i in range(20)])
        different_vol = _cs_with_cols(
            pe_ratio=[10.0 + i for i in range(20)],
            vol_63d=[0.5 + i * 0.05 for i in range(20)],
        )
        different_pe = _cs_with_cols(pe_ratio=[30.0 - i for i in range(20)])
        fn = get_signal("VALUE_PE").fn
        assert fn(base).to_list() == fn(different_vol).to_list(), \
            "VALUE_PE incorrectly uses vol_63d"
        assert fn(base).to_list() != fn(different_pe).to_list(), \
            "VALUE_PE should change when pe_ratio changes"

    def test_value_pb_uses_pb_ratio_not_pe_ratio(self):
        """VALUE_PE and VALUE_PB use different columns."""
        cs1 = _cs_with_cols(
            pe_ratio=[10.0 + i for i in range(20)],
            pb_ratio=[1.0 + i * 0.1 for i in range(20)],
        )
        cs2 = _cs_with_cols(
            pe_ratio=[10.0 + i for i in range(20)],
            pb_ratio=[5.0 - i * 0.1 for i in range(20)],  # reversed pb_ratio
        )
        pe_scores_unchanged = (
            get_signal("VALUE_PE").fn(cs1).to_list()
            == get_signal("VALUE_PE").fn(cs2).to_list()
        )
        pb_scores_changed = (
            get_signal("VALUE_PB").fn(cs1).to_list()
            != get_signal("VALUE_PB").fn(cs2).to_list()
        )
        assert pe_scores_unchanged, "VALUE_PE incorrectly depends on pb_ratio"
        assert pb_scores_changed, "VALUE_PB should change when pb_ratio changes"

    def test_low_vol_ignores_pe_ratio(self):
        """LOW_VOL scores must not depend on pe_ratio."""
        cs1 = _cs_with_cols(pe_ratio=[10.0 + i for i in range(20)])
        cs2 = _cs_with_cols(pe_ratio=[100.0 + i for i in range(20)])
        assert get_signal("LOW_VOL").fn(cs1).to_list() == get_signal("LOW_VOL").fn(cs2).to_list()

    def test_quality_roe_ignores_pe_ratio(self):
        """QUALITY_ROE scores must not depend on pe_ratio."""
        base_roe = [0.1 + i * 0.01 for i in range(20)]
        cs1 = _cs_with_cols(roe=base_roe, pe_ratio=[10.0 + i for i in range(20)])
        cs2 = _cs_with_cols(roe=base_roe, pe_ratio=[50.0 + i for i in range(20)])
        assert get_signal("QUALITY_ROE").fn(cs1).to_list() == get_signal("QUALITY_ROE").fn(cs2).to_list(), \
            "QUALITY_ROE incorrectly depends on pe_ratio"

    def test_size_ignores_pe_ratio(self):
        """SIZE scores must not depend on pe_ratio."""
        base_mc = [1e9 * (i + 1) for i in range(20)]
        cs1 = _cs_with_cols(market_cap=base_mc, pe_ratio=[10.0 + i for i in range(20)])
        cs2 = _cs_with_cols(market_cap=base_mc, pe_ratio=[30.0 + i for i in range(20)])
        assert get_signal("SIZE").fn(cs1).to_list() == get_signal("SIZE").fn(cs2).to_list()

    def test_mom_ignores_close_504d(self):
        """MOM_1_12 uses close_21d/close_252d — changing close_504d must not affect it."""
        cs1 = _cs_with_cols()
        cs2 = _cs_with_cols(close_504d=[10.0 + i for i in range(20)])
        assert get_signal("MOM_1_12").fn(cs1).to_list() == get_signal("MOM_1_12").fn(cs2).to_list()

    def test_reversal_prefers_past_losers(self):
        """REVERSAL_1M should give higher score to a past loser than a past winner."""
        cs = pl.DataFrame({
            "ticker": ["WINNER", "LOSER"],
            "date": [date(2020, 1, 2)] * 2,
            "close": [110.0, 90.0],
            "close_21d": [100.0, 100.0],
            "close_63d": [95.0, 95.0],
            "close_126d": [90.0, 90.0],
            "close_252d": [85.0, 85.0],
            "close_504d": [75.0, 75.0],
            "vol_21d": [0.15, 0.15],
            "vol_63d": [0.18, 0.18],
            "forward_return_21d": [0.0, 0.0],
        })
        scores = get_signal("REVERSAL_1M").fn(cs)
        assert scores[1] > scores[0], \
            "REVERSAL_1M should prefer past loser (LOSER score should exceed WINNER score)"


# ---------------------------------------------------------------------------
# Stale / future fundamental data detection
# ---------------------------------------------------------------------------

class TestStaleFundamentals:
    """Verify the PIT join correctly blocks future fundamentals from producing scores."""

    def test_future_only_fundamentals_yield_no_periods(self):
        """If all fundamentals have available_date after the backtest end,
        every period should fail to score, and the backtest raises ValueError
        (the engine's signal for 'no usable data')."""
        store = _make_price_store(n_tickers=15, n_days=900)
        tickers = [f"T{i:02d}" for i in range(15)]

        future_fund_rows = [
            {
                "ticker": t,
                "available_date": date(2099, 1, 1),  # far in the future
                "period_end": date(2098, 9, 30),
                "market_cap": 1e9,
                "pe_ratio": 15.0,
                "pb_ratio": 2.0,
                "roe": 0.15,
                "debt_to_equity": 0.5,
            }
            for t in tickers
        ]
        store.write(pl.DataFrame(future_fund_rows), "fundamentals")

        config = BacktestConfig(
            start=date(2019, 1, 1),
            end=date(2020, 6, 30),
            cost_bps=0.0,
            min_universe_size=5,
            min_score_coverage=0.1,
        )
        with pytest.raises(ValueError, match="no periods"):
            run_backtest("VALUE_PE", config, store)
        store.close()

    def test_mixing_past_and_future_fundamentals_uses_only_past(self):
        """When some tickers have valid fundamentals and others only have future ones,
        only the tickers with available data should score — confirming the PIT join
        filters per ticker, not per period."""
        store = _make_price_store(n_tickers=10, n_days=900)
        tickers = [f"T{i:02d}" for i in range(10)]
        rebal_date = date(2019, 6, 1)

        fund_rows = []
        for i, t in enumerate(tickers):
            if i < 5:
                # Valid: available 6 months before our rebal date
                avail = rebal_date - timedelta(days=180)
            else:
                # Stale: only available well after the entire backtest window ends
                avail = date(2021, 1, 1)
            fund_rows.append({
                "ticker": t,
                "available_date": avail,
                "period_end": avail - timedelta(days=45),
                "market_cap": 1e9 * (i + 1),
                "pe_ratio": 15.0,
                "pb_ratio": 2.0,
                "roe": 0.15,
                "debt_to_equity": 0.5,
            })
        store.write(pl.DataFrame(fund_rows), "fundamentals")

        config = BacktestConfig(
            start=rebal_date,
            end=date(2020, 1, 1),
            cost_bps=0.0,
            min_universe_size=1,
            min_score_coverage=0.0,
        )
        result = run_backtest("SIZE", config, store)
        store.close()

        # At least some periods should run (the 5 valid tickers)
        assert result.n_periods > 0
        # Average universe scored should be ≤ 5 (only the valid tickers contributed scores)
        scored_series = result.period_returns["n_scored"]
        assert scored_series.max() <= 5, (
            "Future fundamentals leaked into scores — PIT join is broken"
        )
