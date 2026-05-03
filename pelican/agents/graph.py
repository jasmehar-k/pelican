"""
LangGraph graph for Stage 7: Researcher → Coder → Critic → Reporter pipeline.

Topology:
    START → [researcher →] coder → critic ──accept──→ reporter → END
                                         ╰─reject, retry < MAX─→ coder
                                         ╰─reject, retries exhausted─→ reporter → END

The Coder node generates and sandbox-validates the signal code (up to 3 code-gen
attempts internally). The Critic runs a real backtest and gates on IC t-stat ≥ 0.5
and net Sharpe ≥ 0.3. On rejection, the graph loops back to Coder (up to
MAX_GRAPH_RETRIES times) with the Critic's feedback attached. On accept, the
Reporter generates an investment memo and persists it to DuckDB.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from langgraph.graph import END, START, StateGraph

from pelican.agents.coder import _make_coder_node
from pelican.agents.critic import _make_critic_node
from pelican.agents.reporter import _make_reporter_node
from pelican.agents.researcher import _make_researcher_node
from pelican.agents.state import AgentState
from pelican.backtest.engine import BacktestConfig

MAX_GRAPH_RETRIES = 2  # graph-level coder→critic cycles before giving up


_HARD_RETRY_CAP = MAX_GRAPH_RETRIES * 4  # absolute ceiling regardless of any bug


def _route_after_critic(state: AgentState) -> str:
    if state.get("decision") == "accept":
        return "reporter"
    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_GRAPH_RETRIES and retry_count < _HARD_RETRY_CAP:
        return "coder"
    return "reporter"


def build_graph(
    store: Any,
    backtest_config: BacktestConfig | None = None,
    model: str | None = None,
    on_token: Callable[[str], None] | None = None,
    on_attempt_start: Callable[[int], None] | None = None,
    with_researcher: bool = True,
):
    """Build and compile the agent graph.

    Args:
        store: DataStore instance (used by Critic for backtest and Reporter for memos).
        backtest_config: Backtest window for IC validation.  Defaults to the
            trailing 12 months ending yesterday.

    Returns:
        Compiled LangGraph graph.  Invoke with an AgentState dict containing at
        minimum {"theme": "..."}; remaining keys default via initial_state().
    """
    if backtest_config is None:
        today = date.today()
        backtest_config = BacktestConfig(
            start=today - timedelta(days=365),
            end=today - timedelta(days=1),
        )

    builder: StateGraph = StateGraph(AgentState)

    if with_researcher:
        builder.add_node("researcher", _make_researcher_node(model=model))
    builder.add_node("coder", _make_coder_node(
        model=model,
        on_token=on_token,
        on_attempt_start=on_attempt_start,
    ))
    builder.add_node("critic", _make_critic_node(store, backtest_config))
    builder.add_node("reporter", _make_reporter_node(store, model=model))

    if with_researcher:
        builder.add_edge(START, "researcher")
        builder.add_edge("researcher", "coder")
    else:
        builder.add_edge(START, "coder")

    builder.add_edge("coder", "critic")
    builder.add_conditional_edges("critic", _route_after_critic, {
        "coder": "coder",
        "reporter": "reporter",
    })
    builder.add_edge("reporter", END)

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
        papers=[],
        signal_hypothesis=None,
        arxiv_ids=[],
        run_id=str(uuid.uuid4()),
        retry_count=0,
        memo=None,
        signal_name=None,
    )


_STATE_DEFAULTS: dict = {
    "generated_code": None,
    "errors": [],
    "decision": None,
    "feedback": None,
    "ic_tstat": None,
    "sharpe_net": None,
    "papers": [],
    "signal_hypothesis": None,
    "arxiv_ids": [],
    "run_id": "",
    "retry_count": 0,
    "memo": None,
    "signal_name": None,
}


def coerce_state(state: dict) -> AgentState:
    """Fill any missing AgentState fields with safe defaults.

    Allows callers that build state by hand (e.g. the API, direct test
    invocations) to omit fields that were added in later stages without
    causing KeyError inside nodes.
    """
    merged = {**_STATE_DEFAULTS, **state}
    if not merged["run_id"]:
        merged["run_id"] = str(uuid.uuid4())
    if not isinstance(merged["errors"], list):
        merged["errors"] = []
    return AgentState(**merged)
