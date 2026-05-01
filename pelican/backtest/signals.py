"""
Signal base class and in-process registry.

Defines the Signal protocol that all generated and hand-coded signals must
implement. Maintains a registry of validated signals keyed by name and version.
Each Signal wraps a compute function, its metadata (SignalSpec), and its latest
BacktestResult. Persisted to DuckDB via the data store.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class SignalSpec:
    name: str
    description: str
    # Minimum lookback in calendar days needed to compute this signal.
    # The engine will fetch this many extra days of history before rebal_date.
    lookback_days: int = 504
    # Whether the engine should join the fundamentals panel into the cross-section.
    requires_fundamentals: bool = False
    # Which fundamental columns this signal reads (for validation).
    data_deps: tuple[str, ...] = ()
    # Expected IC range (lo, hi) as a rough sanity-check reference.
    expected_ic_range: tuple[float, float] = (-0.10, 0.10)
    # Rebalance frequency of the underlying data.
    data_frequency: str = "monthly"
    # Whether the engine should join the edgar_sentiment panel into the cross-section.
    requires_edgar: bool = False
    # Which edgar columns this signal reads (for documentation).
    edgar_data_deps: tuple[str, ...] = ()


# A signal function receives the cross-section panel (all tickers, all history
# up to and including rebal_date) and returns a score Series aligned with the
# input DataFrame's rows at `rebal_date`.  Higher score = more bullish.
# NaN is valid and is dropped before quintile ranking.
SignalFn = Callable[[pl.DataFrame], pl.Series]


@dataclass
class SignalDef:
    spec: SignalSpec
    fn: SignalFn


_REGISTRY: dict[str, SignalDef] = {}


def register(spec: SignalSpec) -> Callable[[SignalFn], SignalFn]:
    """Decorator that adds a signal function to the registry."""
    def decorator(fn: SignalFn) -> SignalFn:
        _REGISTRY[spec.name] = SignalDef(spec=spec, fn=fn)
        return fn
    return decorator


def get_signal(name: str) -> SignalDef:
    if name not in _REGISTRY:
        raise KeyError(f"Signal '{name}' not registered. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_signals() -> list[str]:
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Cross-section feature builder
# ---------------------------------------------------------------------------

def build_cross_section_features(
    panel: pl.DataFrame,
    rebal_date,
) -> pl.DataFrame:
    """Attach lagged-price and rolling-vol features to the panel.

    All lags and rolling windows are computed within each ticker using
    `.over("ticker")` so that no cross-ticker leakage occurs.  The result
    is then filtered to `date == rebal_date` so each ticker has exactly one row.
    """
    lags = {
        "close_21d": 21,
        "close_63d": 63,
        "close_126d": 126,
        "close_252d": 252,
        "close_504d": 504,
    }
    lag_exprs = [
        pl.col("close").shift(n).over("ticker").alias(name)
        for name, n in lags.items()
    ]
    vol_exprs = [
        (
            pl.col("log_return_1d")
            .rolling_std(window_size=w)
            .over("ticker")
            * math.sqrt(252)
        ).alias(f"vol_{w}d")
        for w in (21, 63)
    ]
    enriched = panel.sort(["ticker", "date"]).with_columns(lag_exprs + vol_exprs)
    return enriched.filter(pl.col("date") == rebal_date)


# ---------------------------------------------------------------------------
# Benchmark signals
# ---------------------------------------------------------------------------

@register(SignalSpec(
    name="HML_REVERSAL",
    description="2-year price reversal as a value proxy (De Bondt & Thaler 1985). "
                "Negated so high signal → cheap / value tilt.",
    lookback_days=504,
))
def _hml_reversal(cs: pl.DataFrame) -> pl.Series:
    return (-(cs["close"] / cs["close_504d"] - 1.0)).alias("HML_REVERSAL")
