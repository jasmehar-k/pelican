"""
Risk model: covariance matrix estimation for ~500 S&P 500 stocks.

Uses Ledoit-Wolf analytical shrinkage (sklearn.covariance.LedoitWolf) over a
rolling 252-day window of daily log returns. At N=500, the 500×500 matrix is
dense but still tractable for CVXPy's CLARABEL solver (~1–3s per solve).

PCA decomposition decomposes Sigma into F @ F.T + D (k systematic factors,
diagonal idiosyncratic variances) for risk attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf


@dataclass
class RiskDecomposition:
    tickers: list[str]
    weights: np.ndarray                 # N portfolio weights
    total_variance: float
    systematic_variance: float          # explained by top-k PCA factors
    idiosyncratic_variance: float       # residual
    factor_contributions: np.ndarray    # N — per-ticker systematic contribution
    idio_contributions: np.ndarray      # N — per-ticker idiosyncratic contribution

    @property
    def systematic_pct(self) -> float:
        return self.systematic_variance / self.total_variance if self.total_variance > 0 else 0.0

    @property
    def idiosyncratic_pct(self) -> float:
        return self.idiosyncratic_variance / self.total_variance if self.total_variance > 0 else 0.0


@dataclass
class RiskModel:
    tickers: list[str]
    cov: np.ndarray                      # N×N shrunk covariance
    shrinkage: float                     # Ledoit-Wolf shrinkage coefficient
    factor_loadings: np.ndarray          # N×k — top PCA eigenvectors
    factor_variances: np.ndarray         # k — eigenvalues of systematic factors
    idio_variances: np.ndarray           # N — diagonal idiosyncratic variances


def estimate_covariance(
    returns: pl.DataFrame,
    tickers: list[str],
    n_factors: int = 10,
    min_periods: int = 63,
) -> RiskModel:
    """Estimate a Ledoit-Wolf shrunk covariance matrix from daily log returns.

    Args:
        returns: DataFrame with columns [date] + ticker columns, sorted by date.
            Each ticker column contains daily log returns.
        tickers: Ordered list of tickers (must match column names in returns).
        n_factors: Number of PCA factors to extract for risk decomposition.
        min_periods: Minimum non-null observations per ticker to include.

    Returns:
        RiskModel with full covariance, shrinkage coefficient, and PCA decomposition.
    """
    if not tickers:
        raise ValueError("tickers must be non-empty")

    # Extract return matrix; drop tickers with insufficient history.
    available = [t for t in tickers if t in returns.columns]
    if not available:
        raise ValueError("None of the requested tickers are in returns DataFrame")

    mat = returns.select(available).to_numpy()  # T×N

    # Drop tickers with too many nulls.
    valid_mask = np.sum(~np.isnan(mat), axis=0) >= min_periods
    valid_tickers = [t for t, ok in zip(available, valid_mask) if ok]
    mat = mat[:, valid_mask]

    if mat.shape[1] < 2:
        raise ValueError(f"Fewer than 2 tickers with >= {min_periods} returns observations")

    # Fill remaining NaNs with column means (for partial history tickers).
    col_means = np.nanmean(mat, axis=0)
    nan_mask = np.isnan(mat)
    mat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    lw = LedoitWolf().fit(mat)
    cov = lw.covariance_
    shrinkage = float(lw.shrinkage_)

    # PCA decomposition for risk attribution.
    # Sigma = Q @ diag(eigenvalues) @ Q.T
    # Systematic: top-k components; Idiosyncratic: residual diagonal.
    n_factors_actual = min(n_factors, mat.shape[1] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order; take the largest n_factors.
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    factor_variances = eigenvalues[:n_factors_actual]          # k
    factor_loadings = eigenvectors[:, :n_factors_actual]       # N×k

    # Idiosyncratic = diagonal of (Sigma - F @ diag(λ) @ F.T).
    systematic_cov = factor_loadings @ np.diag(factor_variances) @ factor_loadings.T
    residual = cov - systematic_cov
    idio_variances = np.maximum(np.diag(residual), 0.0)        # clip negatives from floating point

    return RiskModel(
        tickers=valid_tickers,
        cov=cov,
        shrinkage=shrinkage,
        factor_loadings=factor_loadings,
        factor_variances=factor_variances,
        idio_variances=idio_variances,
    )


def decompose_risk(weights: np.ndarray, risk_model: RiskModel) -> RiskDecomposition:
    """Decompose portfolio variance into systematic and idiosyncratic components.

    Systematic variance = w.T @ (F @ diag(λ) @ F.T) @ w
    Idiosyncratic variance = w.T @ diag(idio) @ w

    Per-ticker contributions follow the Euler decomposition:
      systematic_i   = w_i * (F @ diag(λ) @ F.T @ w)_i
      idio_i         = w_i * idio_i * w_i  (= w_i² * idio_i)
    """
    if len(weights) != len(risk_model.tickers):
        raise ValueError(
            f"weights length {len(weights)} != tickers length {len(risk_model.tickers)}"
        )

    F = risk_model.factor_loadings       # N×k
    lam = risk_model.factor_variances    # k

    total_cov_w = risk_model.cov @ weights                     # N  (Sigma @ w)
    systematic_cov_w = F @ (np.diag(lam) @ (F.T @ weights))   # N  (F diag(λ) F.T @ w)

    total_variance = float(weights @ total_cov_w)
    systematic_variance = float(weights @ systematic_cov_w)
    # Idiosyncratic is defined as the remainder so the decomposition is exact.
    idiosyncratic_variance = total_variance - systematic_variance

    # Per-ticker Euler contributions (w_i * (Sigma @ w)_i sums to total variance).
    factor_contributions = weights * systematic_cov_w
    idio_contributions = weights * total_cov_w - factor_contributions

    return RiskDecomposition(
        tickers=risk_model.tickers,
        weights=weights,
        total_variance=total_variance,
        systematic_variance=systematic_variance,
        idiosyncratic_variance=idiosyncratic_variance,
        factor_contributions=factor_contributions,
        idio_contributions=idio_contributions,
    )


def build_returns_wide(prices: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
    """Pivot a long prices DataFrame into a wide (date × ticker) returns matrix.

    Args:
        prices: Long DataFrame with columns [ticker, date, log_return_1d].
        tickers: Tickers to include (others are dropped).

    Returns:
        Wide DataFrame: columns = [date] + tickers, rows = trading days.
    """
    filtered = prices.filter(pl.col("ticker").is_in(tickers))
    wide = (
        filtered.select(["ticker", "date", "log_return_1d"])
        .pivot(index="date", on="ticker", values="log_return_1d")
        .sort("date")
    )
    return wide
