"""
Critic agent node.

Runs a 1-year backtest on the generated signal function and rejects it if:
  - no code was produced
  - the backtest raises an error
  - IC t-stat < IC_TSTAT_THRESHOLD (0.5)
  - net Sharpe < SHARPE_THRESHOLD (0.3)

Pre-flight check: if the signal requires fundamentals, the critic queries the DB
for the available date range and returns a clear error rather than silently
skipping all periods.  This prevents the misleading "no periods" error when
the backtest window predates the fundamentals data.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from typing import Any

from pelican.agents.state import AgentState
from pelican.agents.tools.backtest_tool import run_backtest_with_fn
from pelican.agents.tools.code_exec import execute_signal_code, needs_fundamentals
from pelican.backtest.engine import BacktestConfig
from pelican.backtest.signals import SignalSpec
from pelican.utils.logging import get_logger

IC_TSTAT_THRESHOLD = 1.5
SHARPE_THRESHOLD = 0.3

log = get_logger(__name__)


def _fundamentals_coverage(store: Any) -> tuple[date, date] | None:
    """Return (first_dense_date, latest_date) for the fundamentals table.

    first_dense_date: earliest available_date where ≥100 tickers have non-null roe.
    latest_date: the largest available_date in the table.

    Using non-null roe as the coverage proxy avoids misleading rows that exist
    with all-null values (yfinance creates empty rows for quarters it doesn't
    populate, so MIN(available_date) would be falsely early).
    """
    try:
        r = store.query(
            """
            SELECT
                (SELECT MIN(available_date)
                 FROM (
                     SELECT available_date
                     FROM fundamentals
                     WHERE roe IS NOT NULL
                     GROUP BY available_date
                     HAVING COUNT(DISTINCT ticker) >= 100
                 )) AS lo,
                MAX(available_date) AS hi
            FROM fundamentals
            """
        )
        if r.is_empty():
            return None
        lo, hi = r["lo"][0], r["hi"][0]
        if lo is None or hi is None:
            return None
        return lo, hi
    except Exception:
        return None


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

        requires_fund = needs_fundamentals(code)

        # Pre-flight: verify fundamentals coverage if the signal needs them.
        effective_config = config
        if requires_fund:
            coverage = _fundamentals_coverage(store)
            if coverage is None:
                return {
                    **state,
                    "decision": "reject",
                    "feedback": (
                        "signal uses fundamental columns (roe / debt_to_equity / pe_ratio / "
                        "pb_ratio / market_cap) but no fundamentals data found in the database. "
                        "Run: python scripts/seed_fundamentals.py"
                    ),
                    "ic_tstat": None,
                    "sharpe_net": None,
                }
            fund_lo, fund_hi = coverage
            if config.end < fund_lo or config.start > fund_hi:
                return {
                    **state,
                    "decision": "reject",
                    "feedback": (
                        f"backtest window {config.start} → {config.end} has no overlap with "
                        f"available fundamentals ({fund_lo} → {fund_hi}). "
                        f"Re-run with --start {fund_lo} or later."
                    ),
                    "ic_tstat": None,
                    "sharpe_net": None,
                }
            if config.start < fund_lo:
                # Trim start to first available period so we get real scores.
                effective_config = replace(config, start=fund_lo)
                log.info(
                    "adjusted backtest start for fundamentals coverage",
                    original=config.start,
                    adjusted=fund_lo,
                )

        spec = SignalSpec(
            name="_critic_eval",
            description=state["theme"],
            requires_fundamentals=requires_fund,
        )

        try:
            result = run_backtest_with_fn(fn, spec, effective_config, store)
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
            low_periods_hint = (
                " — fewer than 6 periods had sufficient coverage; "
                "try a longer window or verify fundamentals data"
                if result.n_periods < 6
                else ""
            )
            return {
                **state,
                "decision": "reject",
                "feedback": (
                    f"IC t-stat {ic_tstat:.2f} < threshold {IC_TSTAT_THRESHOLD} "
                    f"(n_periods={result.n_periods}){low_periods_hint}"
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
