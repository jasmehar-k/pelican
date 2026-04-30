"""
Critic agent node.

Runs a 1-year backtest on the generated signal function and rejects it if:
  - no code was produced
  - the backtest raises an error
  - IC t-stat < IC_TSTAT_THRESHOLD (1.5)
  - net Sharpe < SHARPE_THRESHOLD (0.3)

Acceptance does not mean the signal is great — it means it clears the minimum
statistical bar to be worth keeping.  The feedback string explains the decision.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import polars as pl

from pelican.agents.state import AgentState
from pelican.agents.tools.backtest_tool import run_backtest_with_fn
from pelican.agents.tools.code_exec import execute_signal_code, needs_fundamentals
from pelican.backtest.engine import BacktestConfig
from pelican.backtest.signals import SignalSpec
from pelican.utils.logging import get_logger

IC_TSTAT_THRESHOLD = 1.5
SHARPE_THRESHOLD = 0.3

log = get_logger(__name__)


def _make_critic_node(store: Any, config: BacktestConfig):
    """Return a LangGraph-compatible node function closed over store and config."""

    def critic_node(state: AgentState) -> AgentState:
        code = state.get("generated_code")
        if not code:
            return {
                **state,
                "decision": "reject",
                "feedback": "no code was generated after all retries",
                "ic_tstat": None,
                "sharpe_net": None,
            }

        success, error_msg, fn = execute_signal_code(code)
        if not success:
            return {
                **state,
                "decision": "reject",
                "feedback": f"code failed re-validation in critic: {error_msg}",
                "ic_tstat": None,
                "sharpe_net": None,
            }

        spec = SignalSpec(
            name="_critic_eval",
            description=state["theme"],
            requires_fundamentals=needs_fundamentals(code),
        )

        try:
            result = run_backtest_with_fn(fn, spec, config, store)
        except Exception as exc:
            log.warning("critic backtest failed", error=str(exc))
            return {
                **state,
                "decision": "reject",
                "feedback": f"backtest error: {exc}",
                "ic_tstat": None,
                "sharpe_net": None,
            }

        ic_tstat = result.ic_tstat
        sharpe = result.sharpe_net

        log.info(
            "critic evaluation",
            ic_tstat=ic_tstat,
            sharpe_net=sharpe,
            n_periods=result.n_periods,
        )

        if math.isnan(ic_tstat) or ic_tstat < IC_TSTAT_THRESHOLD:
            return {
                **state,
                "decision": "reject",
                "feedback": (
                    f"IC t-stat {ic_tstat:.2f} < threshold {IC_TSTAT_THRESHOLD} "
                    f"(n_periods={result.n_periods})"
                ),
                "ic_tstat": ic_tstat,
                "sharpe_net": sharpe,
            }

        if math.isnan(sharpe) or sharpe < SHARPE_THRESHOLD:
            return {
                **state,
                "decision": "reject",
                "feedback": (
                    f"net Sharpe {sharpe:.2f} < threshold {SHARPE_THRESHOLD} "
                    f"(IC t-stat={ic_tstat:.2f})"
                ),
                "ic_tstat": ic_tstat,
                "sharpe_net": sharpe,
            }

        return {
            **state,
            "decision": "accept",
            "feedback": (
                f"IC t-stat={ic_tstat:.2f}, net Sharpe={sharpe:.2f}, "
                f"IC mean={result.ic_mean:.4f}, periods={result.n_periods}"
            ),
            "ic_tstat": ic_tstat,
            "sharpe_net": sharpe,
        }

    return critic_node
