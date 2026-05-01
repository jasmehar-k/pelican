"""Pydantic request and response models for the API layer."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, Field


def _default_start() -> date:
    return date.today() - timedelta(days=365)


def _default_end() -> date:
    return date.today() - timedelta(days=1)


class AgentRunRequest(BaseModel):
    theme: str
    model: str | None = None
    with_researcher: bool = True
    start: date = Field(default_factory=_default_start)
    end: date = Field(default_factory=_default_end)


class AgentRunSummary(BaseModel):
    run_id: str
    theme: str
    decision: str | None
    ic_tstat: float | None
    sharpe_net: float | None
    retry_count: int
    ts: str


class AgentStreamEvent(BaseModel):
    event: str  # node_start | node_complete | llm_token | run_complete | run_error
    node: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
