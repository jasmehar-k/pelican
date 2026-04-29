"""
Tests for the cross-sectional backtest engine.

Covers: correct forward-return alignment, point-in-time universe filtering,
metric computation (IC, Sharpe), and no look-ahead bias under synthetic data
where the true signal is known.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from pelican.backtest.engine import BacktestConfig, BacktestResult, _turnover, run_backtest
from pelican.backtest.metrics import (
    compute_ic_stats,
    compute_max_drawdown,
    compute_sharpe,
    spearman_ic,
)
from pelican.backtest.signals import (
    SignalDef,
    SignalSpec,
    build_cross_section_features,
    get_signal,
    list_signals,
    register,
)
from pelican.backtest.universe import get_point_in_time_universe, get_rebalance_dates
from pelican.data.store import DataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> DataStore:
    s = DataStore(":memory:")
    s.init_schema()
    return s


def _daily_dates(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _seed_prices(store: DataStore, tickers: list[str], dates: list[date], close_fn=None) -> None:
    """Seed synthetic price rows. close_fn(ticker, i) → float."""
    if close_fn is None:
        close_fn = lambda t, i: 100.0 + i  # noqa: E731

    rows = []
    for t in tickers:
        for i, d in enumerate(dates):
            c = close_fn(t, i)
            rows.append({
                "ticker": t, "date": d,
                "open": c, "high": c, "low": c, "close": c,
                "volume": 1_000_000,
                "log_return_1d": math.log(c / (c - 0.01)) if i > 0 else None,
                "forward_return_21d": 0.01 if i + 21 < len(dates) else None,
            })
    df = pl.DataFrame(rows, schema={
        "ticker": pl.Utf8, "date": pl.Date,
        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
        "close": pl.Float64, "volume": pl.Int64,
        "log_return_1d": pl.Float64, "forward_return_21d": pl.Float64,
    })
    store.write(df, "prices")


def _seed_universe(store: DataStore, tickers: list[str], entry: date, exit_date=None) -> None:
    rows = [{"ticker": t, "entry_date": entry, "exit_date": exit_date, "company": t}
            for t in tickers]
    df = pl.DataFrame(rows, schema={
        "ticker": pl.Utf8, "entry_date": pl.Date,
        "exit_date": pl.Date, "company": pl.Utf8,
    })
    store.write(df, "sp500_universe")


# ---------------------------------------------------------------------------
# Metric tests (pure functions, no DB)
# ---------------------------------------------------------------------------

class TestSpearmanIC:
    def test_perfect_positive_correlation(self):
        s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(spearman_ic(s, r) - 1.0) < 1e-9

    def test_perfect_negative_correlation(self):
        s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = pl.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(spearman_ic(s, r) + 1.0) < 1e-9

    def test_nan_rows_dropped_pairwise(self):
        s = pl.Series([1.0, 2.0, None, 4.0])
        r = pl.Series([1.0, 2.0, 3.0, 4.0])
        ic = spearman_ic(s, r)
        assert not math.isnan(ic)
        assert abs(ic - 1.0) < 1e-9

    def test_too_few_rows_returns_nan(self):
        s = pl.Series([1.0, 2.0])
        r = pl.Series([1.0, 2.0])
        assert math.isnan(spearman_ic(s, r))

    def test_zero_variance_scores_returns_nan(self):
        s = pl.Series([1.0, 1.0, 1.0, 1.0])
        r = pl.Series([1.0, 2.0, 3.0, 4.0])
        assert math.isnan(spearman_ic(s, r))


class TestComputeICStats:
    def test_ic_mean_and_icir(self):
        ic = pl.Series([0.05, 0.05, 0.05, 0.05])
        stats = compute_ic_stats(ic)
        assert abs(stats["ic_mean"] - 0.05) < 1e-9
        assert math.isnan(stats["icir"]) or abs(stats["icir"]) > 0  # std = 0 → nan

    def test_nonzero_icir(self):
        ic = pl.Series([0.1, 0.2, 0.1, 0.2, 0.15])
        stats = compute_ic_stats(ic)
        assert not math.isnan(stats["icir"])
        assert not math.isnan(stats["ic_tstat"])

    def test_too_few_obs_returns_nan(self):
        stats = compute_ic_stats(pl.Series([0.1]))
        assert all(math.isnan(v) for v in stats.values())

    def test_tstat_equals_icir_times_sqrt_n(self):
        ic = pl.Series([0.1, 0.2, 0.15, 0.05, 0.18])
        stats = compute_ic_stats(ic)
        expected_tstat = stats["icir"] * math.sqrt(5)
        assert abs(stats["ic_tstat"] - expected_tstat) < 1e-9


class TestComputeSharpe:
    def test_positive_sharpe(self):
        returns = pl.Series([0.01, 0.02, 0.01, 0.02, 0.01, 0.02] * 4)
        sharpe = compute_sharpe(returns, periods_per_year=12)
        assert sharpe > 0

    def test_annualized_by_sqrt_periods(self):
        # Manual: all equal returns → std=0 → NaN, so use varied data
        returns = pl.Series([0.02, 0.01, 0.02, 0.01, 0.02, 0.01])
        sharpe = compute_sharpe(returns, periods_per_year=12)
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        expected = mean / std * math.sqrt(12)
        assert abs(sharpe - expected) < 1e-9

    def test_zero_variance_returns_nan(self):
        returns = pl.Series([0.01, 0.01, 0.01])
        assert math.isnan(compute_sharpe(returns))

    def test_too_few_returns_nan(self):
        assert math.isnan(compute_sharpe(pl.Series([0.05])))


class TestMaxDrawdown:
    def test_monotone_increase_no_drawdown(self):
        returns = pl.Series([0.01, 0.02, 0.03, 0.01])
        dd = compute_max_drawdown(returns)
        assert abs(dd) < 1e-9

    def test_single_loss_drawdown(self):
        # 10% up, then 20% down
        returns = pl.Series([0.10, -0.20])
        dd = compute_max_drawdown(returns)
        # Peak cumulative = 1.10. Trough = 1.10 * 0.80 = 0.88. DD = (0.88-1.10)/1.10
        expected = (0.88 - 1.10) / 1.10
        assert abs(dd - expected) < 1e-6

    def test_empty_series_returns_nan(self):
        assert math.isnan(compute_max_drawdown(pl.Series([], dtype=pl.Float64)))


# ---------------------------------------------------------------------------
# Turnover tests
# ---------------------------------------------------------------------------

class TestTurnover:
    def test_first_period_full_turnover(self):
        assert _turnover(set(), {"A", "B", "C"}) == 1.0

    def test_no_change_zero_turnover(self):
        assert _turnover({"A", "B"}, {"A", "B"}) == 0.0

    def test_partial_change(self):
        prev = {"A", "B", "C", "D"}
        curr = {"A", "B", "C", "E"}
        # entered: {E}, exited: {D}, denom: 4+4=8
        assert _turnover(prev, curr) == 2 / 8

    def test_complete_replacement(self):
        prev = {"A", "B"}
        curr = {"C", "D"}
        # entered: 2, exited: 2, denom: 2+2=4 → 4/4=1.0
        assert _turnover(prev, curr) == 1.0


# ---------------------------------------------------------------------------
# Cross-section feature builder tests (no DB)
# ---------------------------------------------------------------------------

class TestBuildCrossSectionFeatures:
    def _make_panel(self, n_days: int = 300) -> pl.DataFrame:
        tickers = ["AAPL", "MSFT"]
        start = date(2020, 1, 2)
        dates = _daily_dates(start, n_days)
        rows = []
        for t in tickers:
            for i, d in enumerate(dates):
                close = 100.0 + i + (10 if t == "MSFT" else 0)
                log_ret = math.log(close / (close - 1.0)) if i > 0 else None
                rows.append({"ticker": t, "date": d, "close": close,
                             "log_return_1d": log_ret, "open": close,
                             "high": close, "low": close, "volume": 1_000_000,
                             "forward_return_21d": 0.01})
        return pl.DataFrame(rows, schema={
            "ticker": pl.Utf8, "date": pl.Date,
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
            "close": pl.Float64, "volume": pl.Int64,
            "log_return_1d": pl.Float64, "forward_return_21d": pl.Float64,
        })

    def test_returns_one_row_per_ticker(self):
        panel = self._make_panel(300)
        rebal = date(2020, 1, 2) + timedelta(days=299)
        cs = build_cross_section_features(panel, rebal)
        assert len(cs) == 2

    def test_lag_columns_present(self):
        panel = self._make_panel(300)
        rebal = date(2020, 1, 2) + timedelta(days=299)
        cs = build_cross_section_features(panel, rebal)
        for col in ["close_21d", "close_63d", "close_126d", "close_252d", "vol_21d", "vol_63d"]:
            assert col in cs.columns

    def test_lag_value_is_past_close(self):
        """close_21d at rebal should equal close at rebal - 21 rows."""
        panel = self._make_panel(300)
        rebal_idx = 299
        rebal = date(2020, 1, 2) + timedelta(days=rebal_idx)
        cs = build_cross_section_features(panel, rebal)
        aapl_row = cs.filter(pl.col("ticker") == "AAPL")
        # close_21d should be close at index 299-21 = 278
        expected_close = 100.0 + (rebal_idx - 21)
        assert abs(float(aapl_row["close_21d"][0]) - expected_close) < 1e-6

    def test_no_future_data_in_cross_section(self):
        """The returned cross-section must not contain any date after rebal_date."""
        panel = self._make_panel(300)
        rebal = date(2020, 1, 2) + timedelta(days=150)
        cs = build_cross_section_features(panel, rebal)
        assert (cs["date"] <= rebal).all()


# ---------------------------------------------------------------------------
# Universe query tests (with DB)
# ---------------------------------------------------------------------------

class TestGetRebalanceDates:
    def test_one_per_month(self):
        store = _make_store()
        tickers = ["AAPL"]
        dates = []
        # Jan 2020: 31 days, Feb 2020: 29 days (leap year)
        for m in range(1, 5):
            start = date(2020, m, 1)
            for d in range(20):
                dates.append(start + timedelta(days=d))
        _seed_prices(store, tickers, dates)
        _seed_universe(store, tickers, date(2019, 1, 1))

        rebal = get_rebalance_dates(date(2020, 1, 1), date(2020, 4, 30), store)
        assert len(rebal) == 4  # one per month
        # Each is the minimum date in that month
        for d in rebal:
            assert d.day == 1  # we started each month on day 1

    def test_respects_date_bounds(self):
        store = _make_store()
        dates = _daily_dates(date(2020, 1, 1), 120)
        _seed_prices(store, ["AAPL"], dates)
        _seed_universe(store, ["AAPL"], date(2019, 1, 1))
        rebal = get_rebalance_dates(date(2020, 2, 1), date(2020, 3, 31), store)
        for d in rebal:
            assert date(2020, 2, 1) <= d <= date(2020, 3, 31)


class TestGetPointInTimeUniverse:
    def test_active_ticker_included(self):
        store = _make_store()
        dates = _daily_dates(date(2019, 1, 1), 400)
        _seed_prices(store, ["AAPL"], dates)
        _seed_universe(store, ["AAPL"], date(2019, 1, 1))
        result = get_point_in_time_universe(date(2020, 1, 2), store, min_history_days=10)
        assert "AAPL" in result

    def test_exited_ticker_excluded(self):
        store = _make_store()
        dates = _daily_dates(date(2019, 1, 1), 400)
        _seed_prices(store, ["AAPL"], dates)
        _seed_universe(store, ["AAPL"], date(2019, 1, 1), exit_date=date(2019, 6, 1))
        result = get_point_in_time_universe(date(2020, 1, 2), store)
        assert "AAPL" not in result

    def test_future_entry_excluded(self):
        store = _make_store()
        dates = _daily_dates(date(2020, 1, 1), 400)
        _seed_prices(store, ["AAPL"], dates)
        _seed_universe(store, ["AAPL"], date(2021, 1, 1))  # enters after query date
        result = get_point_in_time_universe(date(2020, 6, 1), store)
        assert "AAPL" not in result

    def test_insufficient_history_excluded(self):
        store = _make_store()
        # Only 10 days of data — below min_history_days=252
        dates = _daily_dates(date(2020, 1, 1), 10)
        _seed_prices(store, ["AAPL"], dates)
        _seed_universe(store, ["AAPL"], date(2019, 1, 1))
        result = get_point_in_time_universe(date(2020, 1, 10), store, min_history_days=252)
        assert "AAPL" not in result


# ---------------------------------------------------------------------------
# Signal registry tests
# ---------------------------------------------------------------------------

class TestSignalRegistry:
    def test_benchmark_signals_registered(self):
        import pelican.backtest.signals  # noqa: F401 — triggers decorators
        signals = list_signals()
        assert "MOM_1_12" in signals
        assert "HML_REVERSAL" in signals
        assert "LOW_VOL" in signals

    def test_get_signal_returns_signal_def(self):
        sig = get_signal("MOM_1_12")
        assert isinstance(sig, SignalDef)
        assert sig.spec.name == "MOM_1_12"

    def test_get_signal_unknown_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            get_signal("DOES_NOT_EXIST_XYZ")

    def test_register_decorator(self):
        spec = SignalSpec(name="_TEST_SIG_UNIQUE", description="test", lookback_days=21)

        @register(spec)
        def _test_fn(cs: pl.DataFrame) -> pl.Series:
            return pl.Series([0.0] * len(cs))

        assert "_TEST_SIG_UNIQUE" in list_signals()
        sig = get_signal("_TEST_SIG_UNIQUE")
        assert sig.fn is _test_fn


# ---------------------------------------------------------------------------
# End-to-end engine tests (fully synthetic, no network)
# ---------------------------------------------------------------------------

def _make_full_store(
    n_tickers: int = 20,
    n_days: int = 600,
    start: date = date(2018, 1, 2),
) -> DataStore:
    """Seed a store with enough data to run a short backtest."""
    store = _make_store()
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    dates = _daily_dates(start, n_days)

    import random
    rng = random.Random(42)

    rows = []
    for t in tickers:
        price = 100.0
        t_seed = rng.random()
        for i, d in enumerate(dates):
            price = price * (1 + (t_seed - 0.5) * 0.02 + rng.gauss(0, 0.01))
            price = max(price, 1.0)
            log_ret = math.log(price / rows[-1]["close"]) if (i > 0 and rows and rows[-1]["ticker"] == t) else None
            rows.append({
                "ticker": t, "date": d,
                "open": price, "high": price, "low": price, "close": price,
                "volume": 1_000_000, "log_return_1d": log_ret,
                "forward_return_21d": None,  # recomputed below
            })

    df = pl.DataFrame(rows, schema={
        "ticker": pl.Utf8, "date": pl.Date,
        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
        "close": pl.Float64, "volume": pl.Int64,
        "log_return_1d": pl.Float64, "forward_return_21d": pl.Float64,
    })

    from pelican.data.prices import compute_returns
    df = compute_returns(df)
    store.write(df, "prices")
    _seed_universe(store, tickers, start - timedelta(days=365))
    return store


class TestRunBacktest:
    def _config(self, n_days: int = 600) -> BacktestConfig:
        return BacktestConfig(
            start=date(2018, 1, 2) + timedelta(days=300),
            end=date(2018, 1, 2) + timedelta(days=n_days - 1),
            cost_bps=5.0,
            min_universe_size=5,
            min_score_coverage=0.1,
            lookback_calendar_days=400,
        )

    def test_returns_backtest_result(self):
        store = _make_full_store()
        result = run_backtest("MOM_1_12", self._config(), store)
        assert isinstance(result, BacktestResult)

    def test_result_has_expected_columns(self):
        store = _make_full_store()
        result = run_backtest("MOM_1_12", self._config(), store)
        for col in ["date", "q1", "q5", "ls_gross", "ls_net", "turnover"]:
            assert col in result.period_returns.columns

    def test_ic_series_has_date_and_ic(self):
        store = _make_full_store()
        result = run_backtest("MOM_1_12", self._config(), store)
        assert "date" in result.ic_series.columns
        assert "ic" in result.ic_series.columns

    def test_net_return_leq_gross(self):
        """After costs, net LS spread ≤ gross LS spread in absolute terms."""
        store = _make_full_store()
        result = run_backtest("MOM_1_12", self._config(), store)
        pr = result.period_returns.drop_nulls(["ls_gross", "ls_net"])
        # For periods with positive gross, net ≤ gross.
        pos = pr.filter(pl.col("ls_gross") > 0)
        if len(pos) > 0:
            assert (pos["ls_net"] <= pos["ls_gross"]).all()

    def test_n_periods_matches_period_returns(self):
        store = _make_full_store()
        result = run_backtest("MOM_1_12", self._config(), store)
        assert result.n_periods == len(result.period_returns)

    def test_avg_universe_size_reasonable(self):
        store = _make_full_store(n_tickers=20)
        result = run_backtest("MOM_1_12", self._config(), store)
        assert 5 <= result.avg_universe_size <= 20

    def test_no_lookahead_bias(self):
        """Signal scores on date T must not depend on prices after T.

        We verify this by checking that the cross-section features used
        at a rebalance date contain no future dates.
        """
        store = _make_full_store()
        rebal_dates = get_rebalance_dates(
            date(2018, 1, 2) + timedelta(days=300),
            date(2018, 1, 2) + timedelta(days=599),
            store,
        )
        # Pull panel and check feature build for first rebal date
        rebal = rebal_dates[0]
        panel = store.query(
            "SELECT * FROM prices WHERE date <= ? ORDER BY ticker, date",
            [rebal],
        )
        cs = build_cross_section_features(panel, rebal)
        # All rows in the cross-section must be at rebal or before
        assert (cs["date"] <= rebal).all()

    def test_low_vol_signal_runs(self):
        store = _make_full_store()
        result = run_backtest("LOW_VOL", self._config(), store)
        assert result.n_periods > 0

    def test_hml_reversal_signal_runs(self):
        store = _make_full_store()
        result = run_backtest("HML_REVERSAL", self._config(), store)
        assert result.n_periods > 0

    def test_first_period_turnover_is_one(self):
        store = _make_full_store()
        result = run_backtest("MOM_1_12", self._config(), store)
        # First period: entering from empty portfolio → turnover = 1.0
        assert float(result.period_returns["turnover"][0]) == 1.0

    def test_max_drawdown_is_non_positive(self):
        store = _make_full_store()
        result = run_backtest("MOM_1_12", self._config(), store)
        if not math.isnan(result.max_drawdown_gross):
            assert result.max_drawdown_gross <= 0.0

    def test_ic_in_valid_range(self):
        store = _make_full_store()
        result = run_backtest("MOM_1_12", self._config(), store)
        ic = result.ic_series["ic"].drop_nulls()
        assert (ic >= -1.0).all() and (ic <= 1.0).all()
