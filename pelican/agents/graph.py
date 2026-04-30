"""
LangGraph graph for Stage 5: single Coder → Critic pipeline.

Topology: START → coder → critic → END

The Coder node generates and sandbox-validates the signal code (up to 3 retries
internally).  The Critic runs a real backtest and gates on IC t-stat ≥ 1.5 and
net Sharpe ≥ 0.3.  Both nodes return the full AgentState, so the final state
contains the generated code, decision, and backtest metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from langgraph.graph import END, START, StateGraph

from pelican.agents.coder import _make_coder_node
from pelican.agents.critic import _make_critic_node
from pelican.agents.state import AgentState
from pelican.backtest.engine import BacktestConfig


def build_graph(
    store: Any,
    backtest_config: BacktestConfig | None = None,
    model: str | None = None,
    on_token: Callable[[str], None] | None = None,
    on_attempt_start: Callable[[int], None] | None = None,
):
    """Build and compile the Coder → Critic graph.

    Args:
        store: DataStore instance (closed over by the Critic node).
        backtest_config: Backtest window for IC validation.  Defaults to the
            trailing 12 months ending yesterday so the graph is always usable
            without passing explicit dates.

    Returns:
        Compiled LangGraph graph.  Invoke with an AgentState dict containing at
        minimum {"theme": "..."}; the remaining keys default to None / [].
    """
    if backtest_config is None:
        today = date.today()
        backtest_config = BacktestConfig(
            start=today - timedelta(days=365),
            end=today - timedelta(days=1),
        )

    critic_node = _make_critic_node(store, backtest_config)

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("coder", _make_coder_node(
        model=model,
        on_token=on_token,
        on_attempt_start=on_attempt_start,
    ))
    builder.add_node("critic", critic_node)
    builder.add_edge(START, "coder")
    builder.add_edge("coder", "critic")
    builder.add_edge("critic", END)

    return builder.compile()


def initial_state(theme: str) -> AgentState:
    """Return a fresh AgentState for a given factor description."""
    return AgentState(
        theme=theme,
        generated_code=None,
        errors=[],
        decision=None,
        feedback=None,
        ic_tstat=None,
        sharpe_net=None,
    )
