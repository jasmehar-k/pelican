"""Vectorized cross-sectional backtesting engine."""

from pelican.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from pelican.backtest.metrics import (
    compute_ic_stats,
    compute_max_drawdown,
    compute_sharpe,
    spearman_ic,
)
from pelican.backtest.signals import SignalDef, SignalSpec, get_signal, list_signals, register
from pelican.backtest.universe import get_point_in_time_universe, get_rebalance_dates

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "compute_ic_stats",
    "compute_max_drawdown",
    "compute_sharpe",
    "spearman_ic",
    "SignalDef",
    "SignalSpec",
    "get_signal",
    "list_signals",
    "register",
    "get_point_in_time_universe",
    "get_rebalance_dates",
]
