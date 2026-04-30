"""LangGraph state type for the agent pipeline."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    theme: str                  # natural language factor description (input)
    generated_code: str | None  # Python source from Coder node
    errors: list[str]           # sandbox / LLM failures accumulated across retries
    decision: str | None        # "accept" | "reject" from Critic node
    feedback: str | None        # Critic's explanation
    ic_tstat: float | None      # backtest IC t-stat from Critic
    sharpe_net: float | None    # backtest net Sharpe from Critic
