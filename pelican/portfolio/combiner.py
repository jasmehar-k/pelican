"""
Multi-signal combination and weighting.

Combines raw scores from multiple signals into a single composite alpha vector.
Strategies: equal weight, IC-weighted.
Each signal is cross-sectionally z-scored before combination so that no single
signal dominates due to scale differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl


@dataclass
class CombinerConfig:
    method: Literal["equal", "ic_weighted"] = "equal"
    # Minimum number of non-null scores a signal must have to participate.
    min_coverage: int = 10


def _zscore(s: pl.Series) -> pl.Series:
    """Cross-sectional z-score, returning nulls for tickers with no score."""
    non_null = s.drop_nulls()
    if len(non_null) < 2:
        return pl.Series(s.name, [None] * len(s), dtype=pl.Float64)
    mu = non_null.mean()
    sigma = non_null.std()
    if sigma is None or sigma == 0.0:
        return pl.Series(s.name, [0.0 if v is not None else None for v in s.to_list()], dtype=pl.Float64)
    return ((s - mu) / sigma).alias(s.name)


def combine(
    scores: dict[str, pl.Series],
    ic_weights: dict[str, float] | None = None,
    config: CombinerConfig | None = None,
) -> pl.Series:
    """Combine per-signal score Series into a single composite alpha.

    Args:
        scores: Mapping from signal name to a pl.Series of float scores,
            all with the same length and ticker ordering.
        ic_weights: Per-signal IC weights (used when config.method == "ic_weighted").
            If None and method is ic_weighted, falls back to equal weight.
        config: Combination settings.

    Returns:
        pl.Series named "alpha" with the composite score. Tickers that have no
        non-null score across any signal receive null.
    """
    if config is None:
        config = CombinerConfig()
    if not scores:
        raise ValueError("scores dict is empty")

    lengths = {name: len(s) for name, s in scores.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"All score Series must have the same length; got {lengths}")

    n = next(iter(lengths.values()))

    # Normalize each signal cross-sectionally.
    normed: dict[str, pl.Series] = {}
    for name, s in scores.items():
        non_null_count = s.drop_nulls().__len__()
        if non_null_count < config.min_coverage:
            continue
        normed[name] = _zscore(s.cast(pl.Float64))

    if not normed:
        return pl.Series("alpha", [None] * n, dtype=pl.Float64)

    # Determine weights.
    if config.method == "ic_weighted" and ic_weights:
        weights = {
            name: max(0.0, ic_weights.get(name, 0.0))
            for name in normed
        }
        total = sum(weights.values())
        if total == 0:
            weights = {name: 1.0 for name in normed}
            total = float(len(normed))
        weights = {name: w / total for name, w in weights.items()}
    else:
        eq = 1.0 / len(normed)
        weights = {name: eq for name in normed}

    # Weighted sum, treating null as 0 contribution (re-normalised by non-null count).
    composite = pl.Series("alpha", [0.0] * n, dtype=pl.Float64)
    weight_sum = pl.Series("_wsum", [0.0] * n, dtype=pl.Float64)

    for name, s in normed.items():
        w = weights[name]
        filled = s.fill_null(0.0)
        has_value = s.is_not_null().cast(pl.Float64) * w
        composite = composite + filled * w
        weight_sum = weight_sum + has_value

    # Where all signals were null, return null.
    result = (
        pl.DataFrame({"_c": composite, "_w": weight_sum})
        .select(
            pl.when(pl.col("_w") > 0)
            .then(pl.col("_c") / pl.col("_w"))
            .otherwise(None)
            .alias("alpha")
        )
        .to_series()
    )
    return result
