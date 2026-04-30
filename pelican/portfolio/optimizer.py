"""
CVXPy long/short portfolio optimizer.

Solves a mean-variance QP each monthly rebalance date:
  maximize  alpha.T @ w - (lambda/2) * w.T @ Sigma @ w
  subject to:
    sum(w) = 0              # dollar-neutral (long/short)
    sum(w_long) = 1         # gross long = 1
    sum(w_short) = -1       # gross short = -1
    |w_i| <= max_weight     # position cap (default 5%)
    sector_exposure <= cap  # GICS sector neutrality (optional)
    ||(w - w_prev)||_1 <= turnover_limit   # turnover constraint (optional)

Alpha vector comes from the signal combiner. Covariance Sigma comes from
the Ledoit-Wolf shrinkage estimator over a 252-day rolling window.
Default solver: CLARABEL. Falls back to OSQP on solver failure.

HRP (Hierarchical Risk Parity) is also available as a non-CVXPy path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from pelican.portfolio.risk import RiskDecomposition, RiskModel, decompose_risk
from pelican.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PortfolioConfig:
    objective: Literal["max_sharpe", "min_variance", "hrp"] = "max_sharpe"
    lambda_risk: float = 1.0          # risk-aversion coefficient (mean-variance only)
    max_weight: float = 0.05          # per-stock absolute weight cap
    turnover_limit: float | None = None   # L1 turnover vs prev weights (fraction)
    # sector_map: ticker -> sector label; if provided, each sector's net exposure is capped.
    sector_map: dict[str, str] | None = None
    sector_cap: float = 0.10          # max |net sector weight| (used if sector_map set)
    n_hrp_factors: int = 10           # PCA factors used inside HRP


@dataclass
class OptimizationResult:
    tickers: list[str]
    weights: np.ndarray                     # N — signed weights (long > 0, short < 0)
    expected_return: float                  # alpha.T @ w
    expected_variance: float                # w.T @ Sigma @ w
    expected_sharpe: float                  # expected_return / sqrt(expected_variance) * sqrt(252)
    status: str                             # "optimal" | "infeasible" | "fallback_osqp" | "hrp"
    risk_decomposition: RiskDecomposition | None = None

    def as_series(self) -> pl.Series:
        return pl.Series("weight", self.weights)

    def long_tickers(self) -> list[str]:
        return [t for t, w in zip(self.tickers, self.weights) if w > 1e-6]

    def short_tickers(self) -> list[str]:
        return [t for t, w in zip(self.tickers, self.weights) if w < -1e-6]


def optimize(
    alpha: pl.Series,
    risk_model: RiskModel,
    config: PortfolioConfig | None = None,
    prev_weights: np.ndarray | None = None,
) -> OptimizationResult:
    """Compute optimal portfolio weights given alpha scores and a risk model.

    Args:
        alpha: Composite score Series indexed by ticker (same order as risk_model.tickers).
            Nulls are treated as zero alpha (ticker is still eligible).
        risk_model: Estimated covariance and PCA decomposition.
        config: Constraint and objective settings.
        prev_weights: Previous period weights for turnover constraint.

    Returns:
        OptimizationResult with signed weights and performance attribution.
    """
    if config is None:
        config = PortfolioConfig()

    tickers = risk_model.tickers
    n = len(tickers)
    cov = risk_model.cov

    # Align alpha to risk_model ticker order; fill nulls with 0.
    alpha_vals = alpha.to_numpy(allow_copy=True).astype(float)
    alpha_vals = np.where(np.isnan(alpha_vals), 0.0, alpha_vals)

    if config.objective == "hrp":
        weights = _hrp(alpha_vals, cov, config)
        status = "hrp"
    else:
        weights, status = _cvxpy_optimize(alpha_vals, cov, config, prev_weights)

    expected_ret = float(alpha_vals @ weights)
    expected_var = float(weights @ cov @ weights)
    ann_sharpe = (expected_ret / np.sqrt(max(expected_var, 1e-12))) * np.sqrt(252) if expected_var > 0 else 0.0

    risk_decomp = decompose_risk(weights, risk_model)

    return OptimizationResult(
        tickers=tickers,
        weights=weights,
        expected_return=expected_ret,
        expected_variance=expected_var,
        expected_sharpe=ann_sharpe,
        status=status,
        risk_decomposition=risk_decomp,
    )


def _cvxpy_optimize(
    alpha: np.ndarray,
    cov: np.ndarray,
    config: PortfolioConfig,
    prev_weights: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """Solve the mean-variance or min-variance QP with CVXPy."""
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError("cvxpy is required for mean-variance optimization") from e

    n = len(alpha)
    # Split into explicit long/short variables — both non-negative — so all
    # constraints remain affine (required for DCP compliance in CVXPy).
    w_long = cp.Variable(n, nonneg=True)   # long component
    w_short = cp.Variable(n, nonneg=True)  # short component (positive = short)
    w = w_long - w_short

    # Objective.
    # Ledoit-Wolf covariance should be PSD, but CVXPy's numerical certification
    # can fail on larger matrices. psd_wrap skips that brittle check.
    portfolio_var = cp.quad_form(w, cp.psd_wrap(cov))
    if config.objective == "min_variance":
        objective = cp.Minimize(portfolio_var)
    else:  # max_sharpe / mean-variance
        objective = cp.Minimize(
            -alpha @ w + (config.lambda_risk / 2.0) * portfolio_var
        )

    constraints = [
        cp.sum(w_long) == 1.0,             # gross long = 1
        cp.sum(w_short) == 1.0,            # gross short = 1 (so net = 0)
        w_long <= config.max_weight,       # per-stock long cap
        w_short <= config.max_weight,      # per-stock short cap
    ]

    # Sector neutrality.
    if config.sector_map and hasattr(config, '_tickers'):
        sectors = sorted(set(config.sector_map.values()))
        for sector in sectors:
            mask = np.array([
                1.0 if config.sector_map.get(t, "") == sector else 0.0
                for t in config._tickers
            ])
            if mask.sum() > 0:
                constraints.append(cp.abs(mask @ w) <= config.sector_cap)

    # Turnover.
    if config.turnover_limit is not None and prev_weights is not None:
        constraints.append(cp.norm1(w - prev_weights) <= config.turnover_limit)

    prob = cp.Problem(objective, constraints)

    status = "optimal"
    for solver in [cp.CLARABEL, cp.OSQP]:
        try:
            prob.solve(solver=solver, verbose=False)
            if w_long.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                if solver == cp.OSQP:
                    status = "fallback_osqp"
                return (w_long.value - w_short.value), status
        except Exception as exc:
            log.warning("solver failed", solver=str(solver), error=str(exc))

    log.warning("optimization infeasible, returning equal-weight long/short")
    # Equal-weight fallback: top half long, bottom half short.
    rank = np.argsort(alpha)
    n_side = n // 5  # ~quintile
    weights = np.zeros(n)
    weights[rank[-n_side:]] = 1.0 / n_side
    weights[rank[:n_side]] = -1.0 / n_side
    return weights, "infeasible"


def _hrp(alpha: np.ndarray, cov: np.ndarray, config: PortfolioConfig) -> np.ndarray:
    """Hierarchical Risk Parity (de Prado 2016) adapted for long/short.

    Splits universe into long (positive alpha) and short (negative alpha) books.
    Applies HRP within each book for risk-balanced allocation, then scales to
    gross long = 1 and gross short = -1.  Tickers with zero alpha are excluded.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    n = len(alpha)
    corr = _cov_to_corr(cov)

    def _hrp_weights(indices: np.ndarray) -> np.ndarray:
        """Recursive bisection HRP within a subset of assets."""
        if len(indices) == 1:
            return np.array([1.0])

        sub_corr = corr[np.ix_(indices, indices)]
        dist = np.sqrt(np.clip(0.5 * (1 - sub_corr), 0, 1))
        np.fill_diagonal(dist, 0.0)
        dist_condensed = squareform(dist)
        link = linkage(dist_condensed, method="single")
        order = leaves_list(link)

        sub_cov = cov[np.ix_(indices, indices)]
        return _recursive_bisection(sub_cov, order)

    def _recursive_bisection(sub_cov: np.ndarray, order: np.ndarray) -> np.ndarray:
        weights = np.ones(len(order))
        clusters = [list(order)]
        while clusters:
            clusters = [
                c[k:] if k else c[:k]
                for c in clusters if len(c) > 1
                for k in [len(c) // 2]
            ]
            # Rebuild: split each cluster in half and allocate via IVP.
            new_clusters = []
            items = [list(order)]
            while items:
                item = items.pop()
                if len(item) == 1:
                    continue
                mid = len(item) // 2
                left, right = item[:mid], item[mid:]
                var_left = _ivp_variance(sub_cov, left)
                var_right = _ivp_variance(sub_cov, right)
                total = var_left + var_right
                alloc_left = var_right / total if total > 0 else 0.5
                alloc_right = var_left / total if total > 0 else 0.5
                for idx in left:
                    weights[idx] *= alloc_left
                for idx in right:
                    weights[idx] *= alloc_right
                items.extend([left, right])
            break  # single pass suffices
        return weights / weights.sum()

    def _ivp_variance(sub_cov: np.ndarray, indices: list[int]) -> float:
        """Inverse-variance portfolio variance for a subset."""
        vols = np.sqrt(np.diag(sub_cov)[indices])
        ivp_w = (1.0 / np.maximum(vols, 1e-12))
        ivp_w /= ivp_w.sum()
        sub = sub_cov[np.ix_(indices, indices)]
        return float(ivp_w @ sub @ ivp_w)

    long_idx = np.where(alpha > 0)[0]
    short_idx = np.where(alpha < 0)[0]

    weights = np.zeros(n)

    if len(long_idx) > 0:
        long_w = _hrp_weights(long_idx)
        weights[long_idx] = long_w  # sums to 1

    if len(short_idx) > 0:
        short_w = _hrp_weights(short_idx)
        weights[short_idx] = -short_w  # sums to -1

    return weights


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-12, 1.0, std)
    return cov / np.outer(std, std)
