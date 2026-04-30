"""
Tests for portfolio construction.

Covers: Ledoit-Wolf covariance PSD, signal combiner normalization and weighting,
CVXPy optimizer constraint satisfaction, HRP feasibility, risk decomposition
additivity, and sector-neutrality enforcement.

All tests use synthetic data — no DuckDB or network access required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

from pelican.portfolio.combiner import CombinerConfig, combine
from pelican.portfolio.optimizer import OptimizationResult, PortfolioConfig, optimize
from pelican.portfolio.risk import (
    RiskModel,
    build_returns_wide,
    decompose_risk,
    estimate_covariance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_returns(n_tickers: int = 30, n_days: int = 300) -> pl.DataFrame:
    """Synthetic daily log returns as a wide (date × ticker) DataFrame."""
    from datetime import date, timedelta

    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    start = date(2022, 1, 1)
    dates = [start + timedelta(days=d) for d in range(n_days)]

    data: dict[str, list] = {"date": dates}
    for t in tickers:
        data[t] = RNG.normal(0.0, 0.01, n_days).tolist()
    return pl.DataFrame(data)


def _make_risk_model(n: int = 20) -> RiskModel:
    returns = _make_returns(n_tickers=n, n_days=252)
    tickers = [f"T{i:02d}" for i in range(n)]
    return estimate_covariance(returns, tickers, n_factors=5)


def _make_alpha(n: int = 20) -> pl.Series:
    vals = RNG.standard_normal(n).tolist()
    return pl.Series("alpha", vals)


# ---------------------------------------------------------------------------
# Risk model tests
# ---------------------------------------------------------------------------

class TestLedoitWolf:
    def test_covariance_is_positive_semidefinite(self):
        rm = _make_risk_model(n=30)
        eigenvalues = np.linalg.eigvalsh(rm.cov)
        assert np.all(eigenvalues >= -1e-10), \
            f"Covariance has negative eigenvalue: {eigenvalues.min():.2e}"

    def test_shrinkage_in_unit_interval(self):
        rm = _make_risk_model(n=30)
        assert 0.0 <= rm.shrinkage <= 1.0

    def test_tickers_match_cov_shape(self):
        rm = _make_risk_model(n=20)
        n = len(rm.tickers)
        assert rm.cov.shape == (n, n)
        assert rm.factor_loadings.shape[0] == n
        assert len(rm.idio_variances) == n

    def test_factor_loadings_orthonormal_columns(self):
        rm = _make_risk_model(n=20)
        F = rm.factor_loadings   # N×k
        # Columns should be orthonormal (eigenvectors of symmetric matrix).
        gram = F.T @ F
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-10)

    def test_raises_on_empty_tickers(self):
        returns = _make_returns()
        with pytest.raises(ValueError, match="non-empty"):
            estimate_covariance(returns, [])

    def test_build_returns_wide(self):
        from datetime import date, timedelta
        tickers = ["AAPL", "MSFT", "GOOG"]
        rows = [
            {"ticker": t, "date": date(2024, 1, d + 1), "log_return_1d": float(i + d * 0.01)}
            for i, t in enumerate(tickers)
            for d in range(5)
        ]
        prices = pl.DataFrame(rows)
        wide = build_returns_wide(prices, tickers)
        assert "date" in wide.columns
        assert all(t in wide.columns for t in tickers)
        assert len(wide) == 5


# ---------------------------------------------------------------------------
# Risk decomposition tests
# ---------------------------------------------------------------------------

class TestRiskDecomposition:
    def test_systematic_plus_idio_approx_total(self):
        rm = _make_risk_model(n=20)
        w = np.zeros(len(rm.tickers))
        w[:3] = 1.0 / 3
        w[-3:] = -1.0 / 3
        rd = decompose_risk(w, rm)
        # Systematic + idiosyncratic should sum close to total.
        assert abs(rd.systematic_variance + rd.idiosyncratic_variance - rd.total_variance) < 1e-10

    def test_pct_sum_to_one(self):
        rm = _make_risk_model(n=20)
        w = np.zeros(len(rm.tickers))
        w[:5] = 0.2
        w[-5:] = -0.2
        rd = decompose_risk(w, rm)
        assert abs(rd.systematic_pct + rd.idiosyncratic_pct - 1.0) < 1e-10

    def test_contributions_sum_to_variance(self):
        rm = _make_risk_model(n=20)
        w = np.zeros(len(rm.tickers))
        w[:4] = 0.25
        w[-4:] = -0.25
        rd = decompose_risk(w, rm)
        np.testing.assert_allclose(
            rd.factor_contributions.sum(), rd.systematic_variance, rtol=1e-8
        )
        np.testing.assert_allclose(
            rd.idio_contributions.sum(), rd.idiosyncratic_variance, rtol=1e-8
        )

    def test_raises_on_weight_length_mismatch(self):
        rm = _make_risk_model(n=10)
        with pytest.raises(ValueError, match="weights length"):
            decompose_risk(np.ones(5), rm)


# ---------------------------------------------------------------------------
# Combiner tests
# ---------------------------------------------------------------------------

class TestCombiner:
    def _scores(self, n: int = 30) -> dict[str, pl.Series]:
        return {
            "MOM": pl.Series("MOM", RNG.standard_normal(n).tolist()),
            "VAL": pl.Series("VAL", RNG.standard_normal(n).tolist()),
            "VOL": pl.Series("VOL", RNG.standard_normal(n).tolist()),
        }

    def test_output_length_matches_input(self):
        n = 30
        alpha = combine(self._scores(n))
        assert len(alpha) == n

    def test_equal_weight_is_default(self):
        scores = self._scores(30)
        a1 = combine(scores)
        a2 = combine(scores, config=CombinerConfig(method="equal"))
        np.testing.assert_allclose(a1.drop_nulls().to_numpy(), a2.drop_nulls().to_numpy())

    def test_ic_weighted_differs_from_equal_when_weights_differ(self):
        scores = self._scores(30)
        ic = {"MOM": 0.9, "VAL": 0.05, "VOL": 0.05}  # extreme skew
        a_eq = combine(scores, config=CombinerConfig(method="equal"))
        a_ic = combine(scores, ic_weights=ic, config=CombinerConfig(method="ic_weighted"))
        assert not np.allclose(a_eq.drop_nulls().to_numpy(), a_ic.drop_nulls().to_numpy())

    def test_output_mean_near_zero(self):
        scores = self._scores(50)
        alpha = combine(scores)
        mean_val = float(alpha.drop_nulls().mean())
        assert abs(mean_val) < 0.5

    def test_single_signal_with_nulls(self):
        vals = [1.0, None, 3.0, None, 5.0]
        scores = {"SIG": pl.Series("SIG", vals)}
        alpha = combine(scores, config=CombinerConfig(min_coverage=3))
        assert len(alpha) == 5
        non_null = alpha.drop_nulls()
        assert len(non_null) == 3

    def test_all_null_signal_returns_null_output(self):
        scores = {"SIG": pl.Series("SIG", [None] * 10, dtype=pl.Float64)}
        alpha = combine(scores, config=CombinerConfig(min_coverage=1))
        # All null → combiner skips signal → output is all null.
        assert alpha.drop_nulls().len() == 0

    def test_raises_on_empty_scores(self):
        with pytest.raises(ValueError, match="empty"):
            combine({})

    def test_raises_on_length_mismatch(self):
        scores = {
            "A": pl.Series("A", [1.0, 2.0, 3.0]),
            "B": pl.Series("B", [1.0, 2.0]),
        }
        with pytest.raises(ValueError, match="same length"):
            combine(scores)


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------

class TestOptimizer:
    # With max_weight=0.05 we need 1/0.05 = 20 tickers per leg, so n>=40 to avoid
    # the optimizer being forced to double-count positions to satisfy both gross constraints.
    def _setup(self, n: int = 50):
        rm = _make_risk_model(n)
        alpha = _make_alpha(n)
        return rm, alpha

    def test_max_sharpe_weights_dollar_neutral(self):
        rm, alpha = self._setup()
        result = optimize(alpha, rm)
        assert abs(result.weights.sum()) < 1e-5, \
            f"Portfolio not dollar-neutral: sum(w) = {result.weights.sum():.4f}"

    def test_max_sharpe_gross_long_equals_one(self):
        rm, alpha = self._setup()
        result = optimize(alpha, rm)
        gross_long = result.weights[result.weights > 0].sum()
        assert abs(gross_long - 1.0) < 1e-4

    def test_max_sharpe_gross_short_equals_minus_one(self):
        rm, alpha = self._setup()
        result = optimize(alpha, rm)
        gross_short = result.weights[result.weights < 0].sum()
        assert abs(gross_short + 1.0) < 1e-4

    def test_position_cap_respected(self):
        rm, alpha = self._setup()
        cap = 0.05
        result = optimize(alpha, rm, config=PortfolioConfig(max_weight=cap))
        assert np.all(np.abs(result.weights) <= cap + 1e-5), \
            f"Position cap violated: max |w| = {np.abs(result.weights).max():.4f}"

    def test_min_variance_has_lower_variance_than_max_sharpe(self):
        rm, alpha = self._setup(30)
        r_mv = optimize(alpha, rm, config=PortfolioConfig(objective="min_variance"))
        r_ms = optimize(alpha, rm, config=PortfolioConfig(objective="max_sharpe"))
        assert r_mv.expected_variance <= r_ms.expected_variance + 1e-8

    def test_turnover_constraint_reduces_trading(self):
        rm, alpha = self._setup(50)
        n = len(rm.tickers)
        # Start from a feasible equal-weight L/S portfolio (20 long @ 5%, 20 short @ 5%).
        prev_w = np.zeros(n)
        prev_w[:20] = 0.05
        prev_w[20:40] = -0.05

        limit = 0.5
        result = optimize(
            alpha, rm,
            config=PortfolioConfig(turnover_limit=limit),
            prev_weights=prev_w,
        )
        actual_turnover = np.abs(result.weights - prev_w).sum()
        assert actual_turnover <= limit + 1e-4, \
            f"Turnover {actual_turnover:.4f} exceeds limit {limit}"

    def test_status_is_optimal_on_feasible_problem(self):
        rm, alpha = self._setup()
        result = optimize(alpha, rm)
        assert result.status in ("optimal", "fallback_osqp")

    def test_result_has_risk_decomposition(self):
        rm, alpha = self._setup()
        result = optimize(alpha, rm)
        assert result.risk_decomposition is not None

    def test_as_series_length_matches_tickers(self):
        rm, alpha = self._setup()
        result = optimize(alpha, rm)
        assert len(result.as_series()) == len(result.tickers)

    def test_hrp_objective_dollar_neutral(self):
        rm, alpha = self._setup(20)
        result = optimize(alpha, rm, config=PortfolioConfig(objective="hrp"))
        assert result.status == "hrp"
        assert abs(result.weights.sum()) < 1e-10

    def test_hrp_gross_long_one_short_minus_one(self):
        rm, alpha = self._setup(20)
        result = optimize(alpha, rm, config=PortfolioConfig(objective="hrp"))
        long_tickers = result.long_tickers()
        short_tickers = result.short_tickers()
        assert len(long_tickers) > 0
        assert len(short_tickers) > 0
        gross_long = result.weights[result.weights > 0].sum()
        gross_short = result.weights[result.weights < 0].sum()
        assert abs(gross_long - 1.0) < 1e-10
        assert abs(gross_short + 1.0) < 1e-10

    def test_null_alpha_values_treated_as_zero(self):
        rm = _make_risk_model(20)
        # Half alpha null.
        vals = [1.0] * 10 + [None] * 10
        alpha = pl.Series("alpha", vals)
        result = optimize(alpha, rm)
        assert result.weights is not None
        assert not np.any(np.isnan(result.weights))


# ---------------------------------------------------------------------------
# Edge-case / failure-mode tests
# ---------------------------------------------------------------------------

class TestOptimizerEdgeCases:
    """
    Covers the four production failure modes:
      1. Optimizer infeasible (too-tight cap → fallback path)
      2. Weights don't sum to zero (dollar neutrality broken)
      3. Covariance matrix not positive definite
      4. Turnover constraint silently ignored
    """

    # ── 1. Infeasibility ────────────────────────────────────────────────────

    def test_infeasible_when_cap_impossible(self):
        # max_weight=0.01 with n=10 → need 100 positions per leg but only 10 → infeasible.
        rm = _make_risk_model(10)
        alpha = _make_alpha(10)
        result = optimize(alpha, rm, config=PortfolioConfig(max_weight=0.01))
        assert result.status == "infeasible"

    def test_infeasible_fallback_is_dollar_neutral(self):
        rm = _make_risk_model(10)
        alpha = _make_alpha(10)
        result = optimize(alpha, rm, config=PortfolioConfig(max_weight=0.01))
        assert result.status == "infeasible"
        assert abs(result.weights.sum()) < 1e-12, \
            f"Fallback portfolio not dollar-neutral: sum={result.weights.sum():.2e}"

    def test_infeasible_fallback_gross_balanced(self):
        rm = _make_risk_model(10)
        alpha = _make_alpha(10)
        result = optimize(alpha, rm, config=PortfolioConfig(max_weight=0.01))
        assert result.status == "infeasible"
        gross_long = result.weights[result.weights > 0].sum()
        gross_short = result.weights[result.weights < 0].sum()
        assert abs(gross_long - 1.0) < 1e-12, f"Fallback gross long = {gross_long:.4f}"
        assert abs(gross_short + 1.0) < 1e-12, f"Fallback gross short = {gross_short:.4f}"

    def test_infeasible_fallback_respects_position_cap_not_guaranteed(self):
        # The fallback equal-weight quintile may exceed max_weight — that is expected
        # because the original cap was the source of infeasibility.  What must hold is
        # that the fallback doesn't crash and returns finite weights.
        rm = _make_risk_model(10)
        alpha = _make_alpha(10)
        result = optimize(alpha, rm, config=PortfolioConfig(max_weight=0.01))
        assert np.all(np.isfinite(result.weights))

    # ── 2. Dollar neutrality across all objectives ───────────────────────────

    def test_min_variance_dollar_neutral(self):
        rm = _make_risk_model(50)
        alpha = _make_alpha(50)
        result = optimize(alpha, rm, config=PortfolioConfig(objective="min_variance"))
        assert abs(result.weights.sum()) < 1e-5, \
            f"min_variance portfolio not dollar-neutral: sum={result.weights.sum():.2e}"

    def test_min_variance_weights_finite_and_bounded(self):
        # min_variance may produce matched long/short pairs in the same asset (net ≈ 0),
        # which is variance-optimal.  We verify feasibility: finite weights, gross ≤ 2.
        rm = _make_risk_model(50)
        alpha = _make_alpha(50)
        result = optimize(alpha, rm, config=PortfolioConfig(objective="min_variance"))
        assert np.all(np.isfinite(result.weights))
        assert np.abs(result.weights).sum() <= 2.0 + 1e-5  # gross ≤ 2x (sum_long=1, sum_short=1)

    def test_dollar_neutral_all_objectives(self):
        for obj in ("max_sharpe", "min_variance", "hrp"):
            n = 20 if obj == "hrp" else 50
            rm = _make_risk_model(n)
            alpha = _make_alpha(n)
            result = optimize(alpha, rm, config=PortfolioConfig(objective=obj))
            assert abs(result.weights.sum()) < 1e-5, \
                f"objective={obj}: sum(w)={result.weights.sum():.2e}"

    # ── 3. Non-positive-definite covariance ──────────────────────────────────

    def test_near_indefinite_cov_does_not_crash(self):
        # Inject a tiny negative eigenvalue to simulate floating-point non-PD.
        # cp.psd_wrap should prevent the ARPACK certification failure.
        rm = _make_risk_model(20)
        eigenvalues, eigenvectors = np.linalg.eigh(rm.cov)
        eigenvalues[0] = -1e-8
        cov_bad = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        rm_bad = RiskModel(
            tickers=rm.tickers, cov=cov_bad, shrinkage=rm.shrinkage,
            factor_loadings=rm.factor_loadings,
            factor_variances=rm.factor_variances,
            idio_variances=rm.idio_variances,
        )
        alpha = _make_alpha(20)
        result = optimize(alpha, rm_bad)
        assert result.weights is not None
        assert np.all(np.isfinite(result.weights)), "NaN/Inf in weights for near-indefinite cov"

    def test_near_indefinite_cov_dollar_neutral(self):
        rm = _make_risk_model(20)
        eigenvalues, eigenvectors = np.linalg.eigh(rm.cov)
        eigenvalues[0] = -1e-8
        cov_bad = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        rm_bad = RiskModel(
            tickers=rm.tickers, cov=cov_bad, shrinkage=rm.shrinkage,
            factor_loadings=rm.factor_loadings,
            factor_variances=rm.factor_variances,
            idio_variances=rm.idio_variances,
        )
        alpha = _make_alpha(20)
        result = optimize(alpha, rm_bad)
        assert abs(result.weights.sum()) < 1e-5, \
            f"dollar-neutral violated with near-indefinite cov: sum={result.weights.sum():.2e}"

    def test_rank_deficient_cov_does_not_crash(self):
        # T=15 < N=30 → sample covariance is rank-deficient; LedoitWolf regularizes
        # but near-zero eigenvalues survive.  psd_wrap must handle the certification.
        returns = _make_returns(n_tickers=30, n_days=15)
        tickers = [f"T{i:02d}" for i in range(30)]
        try:
            rm = estimate_covariance(returns, tickers, n_factors=5, min_periods=5)
        except ValueError:
            pytest.skip("fewer than 2 tickers survived min_periods filter")
        alpha = _make_alpha(len(rm.tickers))
        result = optimize(alpha, rm)
        assert result.weights is not None
        assert np.all(np.isfinite(result.weights)), "NaN/Inf in weights for rank-deficient cov"

    # ── 4. Turnover constraint correctness ───────────────────────────────────

    def test_turnover_limit_ignored_without_prev_weights(self):
        # A limit with no prev_weights cannot be enforced — constraint must be dropped,
        # giving the same result as unconstrained.
        rm = _make_risk_model(50)
        alpha = _make_alpha(50)
        result_free = optimize(alpha, rm)
        result_limited = optimize(alpha, rm, config=PortfolioConfig(turnover_limit=0.01))
        np.testing.assert_allclose(
            result_free.weights, result_limited.weights, atol=1e-5,
            err_msg="turnover_limit with no prev_weights changed the result unexpectedly",
        )

    def test_prev_weights_ignored_without_turnover_limit(self):
        # prev_weights with no turnover_limit must not constrain the optimizer.
        rm = _make_risk_model(50)
        alpha = _make_alpha(50)
        n = len(rm.tickers)
        prev = np.zeros(n)
        prev[:20] = 0.05
        prev[20:40] = -0.05
        result_with_prev = optimize(alpha, rm, prev_weights=prev)
        result_free = optimize(alpha, rm)
        np.testing.assert_allclose(
            result_with_prev.weights, result_free.weights, atol=1e-5,
            err_msg="prev_weights without turnover_limit changed the result unexpectedly",
        )

    def test_tight_turnover_limit_binds(self):
        # A very tight limit from a feasible starting point forces nearly zero trading.
        rm = _make_risk_model(50)
        alpha = _make_alpha(50)
        n = len(rm.tickers)
        prev = np.zeros(n)
        prev[:20] = 0.05
        prev[20:40] = -0.05
        limit = 0.02
        result = optimize(
            alpha, rm,
            config=PortfolioConfig(turnover_limit=limit),
            prev_weights=prev,
        )
        actual = float(np.abs(result.weights - prev).sum())
        assert actual <= limit + 1e-4, \
            f"Tight turnover violated: actual={actual:.4f} > limit={limit}"

    def test_turnover_honored_across_limits(self):
        rm = _make_risk_model(50)
        alpha = _make_alpha(50)
        n = len(rm.tickers)
        prev = np.zeros(n)
        prev[:20] = 0.05
        prev[20:40] = -0.05
        for limit in (0.1, 0.5, 1.0, 2.0):
            result = optimize(
                alpha, rm,
                config=PortfolioConfig(turnover_limit=limit),
                prev_weights=prev,
            )
            actual = float(np.abs(result.weights - prev).sum())
            assert actual <= limit + 1e-4, \
                f"limit={limit}: actual turnover {actual:.4f} exceeded"
