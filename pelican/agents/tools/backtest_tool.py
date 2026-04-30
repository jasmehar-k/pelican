"""
Backtest runner for dynamically generated signal functions.

Temporarily registers a callable in the signal registry under a unique name,
runs the backtest engine, then removes the registration so the registry stays
clean.  The Critic node uses this to evaluate generated code against real data.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import replace

import polars as pl

from pelican.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from pelican.backtest.signals import SignalDef, SignalSpec, _REGISTRY


def run_backtest_with_fn(
    fn: Callable[[pl.DataFrame], pl.Series],
    spec: SignalSpec,
    config: BacktestConfig,
    store: object,
) -> BacktestResult:
    """Run a backtest for an arbitrary callable without permanently registering it.

    A temporary name is used so the registry is left clean regardless of outcome.
    Thread-safety note: concurrent calls with the same store are safe because each
    invocation uses a unique temp name.
    """
    temp_name = f"_agent_{uuid.uuid4().hex[:12]}"
    temp_spec = replace(spec, name=temp_name)
    _REGISTRY[temp_name] = SignalDef(spec=temp_spec, fn=fn)
    try:
        return run_backtest(temp_name, config, store)
    finally:
        _REGISTRY.pop(temp_name, None)
