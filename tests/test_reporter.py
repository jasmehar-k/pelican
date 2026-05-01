"""Tests for the Reporter node and the graph-level critic→coder retry loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pelican.agents.graph import MAX_GRAPH_RETRIES, _route_after_critic, initial_state
from pelican.agents.reporter import _make_reporter_node
from pelican.agents.state import AgentState
from pelican.data.store import DataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store() -> DataStore:
    s = DataStore(":memory:")
    s.init_schema()
    return s


def _state(**overrides) -> AgentState:
    s = initial_state("test theme")
    s.update(overrides)
    return s


# ---------------------------------------------------------------------------
# TestGraphRouting
# ---------------------------------------------------------------------------

class TestGraphRouting:
    def test_accept_routes_to_reporter(self):
        state = _state(decision="accept", retry_count=1)
        assert _route_after_critic(state) == "reporter"

    def test_reject_below_max_routes_to_coder(self):
        state = _state(decision="reject", retry_count=1)
        assert _route_after_critic(state) == "coder"

    def test_reject_at_max_routes_to_reporter(self):
        state = _state(decision="reject", retry_count=MAX_GRAPH_RETRIES)
        assert _route_after_critic(state) == "reporter"

    def test_reject_above_max_routes_to_reporter(self):
        state = _state(decision="reject", retry_count=MAX_GRAPH_RETRIES + 1)
        assert _route_after_critic(state) == "reporter"

    def test_no_decision_routes_to_reporter(self):
        # No decision means critic couldn't run — bail out gracefully
        state = _state(decision=None, retry_count=MAX_GRAPH_RETRIES)
        assert _route_after_critic(state) == "reporter"

    def test_initial_retry_count_is_zero(self):
        state = initial_state("theme")
        assert state["retry_count"] == 0


# ---------------------------------------------------------------------------
# TestCoderFeedbackLoop
# ---------------------------------------------------------------------------

class TestCoderFeedbackLoop:
    def test_coder_increments_retry_count(self):
        from pelican.agents.coder import _make_coder_node

        good_code = (
            "import polars as pl\n"
            "def compute_signal(df: pl.DataFrame) -> pl.Series:\n"
            "    return df['close'].alias('s')\n"
        )
        llm_response = MagicMock()
        llm_response.content = f"```python\n{good_code}\n```"

        with patch("pelican.agents.coder._get_llm", return_value=MagicMock(invoke=lambda _: llm_response)):
            coder = _make_coder_node()
            result = coder(_state(retry_count=0))

        assert result["retry_count"] == 1

    def test_coder_includes_critic_feedback_on_retry(self):
        from pelican.agents.coder import _build_user_message

        msg = _build_user_message(
            "momentum factor",
            errors=[],
            critic_feedback="IC t-stat 0.3 < threshold 0.5",
        )
        assert "rejected" in msg.lower() or "REJECTED" in msg
        assert "IC t-stat" in msg

    def test_coder_no_feedback_on_first_attempt(self):
        from pelican.agents.coder import _build_user_message

        msg = _build_user_message("momentum factor", errors=[])
        assert "rejected" not in msg.lower()
        assert "REJECTED" not in msg


# ---------------------------------------------------------------------------
# TestReporterNode
# ---------------------------------------------------------------------------

class TestReporterNode:
    def test_accepted_signal_generates_memo(self):
        store = _store()
        memo_text = "The signal captures cross-sectional momentum over 12 months."
        mock_response = MagicMock()
        mock_response.content = memo_text

        with patch("pelican.agents.reporter._get_llm",
                   return_value=MagicMock(invoke=lambda _: mock_response)):
            reporter = _make_reporter_node(store)
            result = reporter(_state(
                decision="accept",
                ic_tstat=1.5,
                sharpe_net=0.8,
                generated_code="def compute_signal(df): ...",
                signal_hypothesis="12-month momentum.",
                arxiv_ids=["2401.00001"],
                retry_count=1,
            ))

        assert result["memo"] == memo_text

    def test_rejected_signal_has_no_memo(self):
        store = _store()
        reporter = _make_reporter_node(store)
        result = reporter(_state(decision="reject", retry_count=2))
        assert result["memo"] is None

    def test_memo_persisted_to_db(self):
        store = _store()
        memo_text = "The momentum signal demonstrated robust IC over 46 periods."
        mock_response = MagicMock()
        mock_response.content = memo_text

        with patch("pelican.agents.reporter._get_llm",
                   return_value=MagicMock(invoke=lambda _: mock_response)):
            reporter = _make_reporter_node(store)
            reporter(_state(
                decision="accept",
                ic_tstat=1.5,
                sharpe_net=0.8,
                generated_code="def compute_signal(df): ...",
                retry_count=1,
            ))

        rows = store.query("SELECT memo, decision FROM signal_memos")
        assert rows.shape[0] == 1
        assert rows["memo"][0] == memo_text
        assert rows["decision"][0] == "accept"

    def test_rejected_run_still_persisted(self):
        store = _store()
        reporter = _make_reporter_node(store)
        reporter(_state(decision="reject", retry_count=MAX_GRAPH_RETRIES))

        rows = store.query("SELECT decision, memo FROM signal_memos")
        assert rows.shape[0] == 1
        assert rows["decision"][0] == "reject"
        assert rows["memo"][0] is None

    def test_retry_count_stored_in_memo_row(self):
        store = _store()
        mock_response = MagicMock()
        mock_response.content = "Memo text."

        with patch("pelican.agents.reporter._get_llm",
                   return_value=MagicMock(invoke=lambda _: mock_response)):
            reporter = _make_reporter_node(store)
            reporter(_state(decision="accept", ic_tstat=1.2, sharpe_net=0.5, retry_count=2))

        rows = store.query("SELECT retry_count FROM signal_memos")
        assert rows["retry_count"][0] == 2

    def test_llm_failure_falls_back_to_plain_memo(self):
        store = _store()
        with patch("pelican.agents.reporter._get_llm",
                   return_value=MagicMock(invoke=MagicMock(side_effect=RuntimeError("API down")))):
            reporter = _make_reporter_node(store)
            result = reporter(_state(
                decision="accept",
                ic_tstat=1.5,
                sharpe_net=0.8,
                retry_count=1,
            ))

        assert result["memo"] is not None
        assert "1.500" in result["memo"] or "1.5" in result["memo"]

    def test_memo_state_field_initialises_none(self):
        state = initial_state("momentum")
        assert state["memo"] is None


# ---------------------------------------------------------------------------
# TestSignalMemosTable
# ---------------------------------------------------------------------------

class TestSignalMemosTable:
    def test_signal_memos_table_created(self):
        store = _store()
        tables = store.query("SHOW TABLES")
        assert "signal_memos" in tables["name"].to_list()

    def test_log_memo_inserts_row(self):
        store = _store()
        store.log_memo({
            "run_id": "run-1",
            "theme": "momentum",
            "decision": "accept",
            "ic_tstat": 1.5,
            "sharpe_net": 0.8,
            "retry_count": 1,
            "arxiv_ids": ["2401.00001"],
            "memo": "A brief memo.",
        })
        rows = store.query("SELECT * FROM signal_memos")
        assert rows.shape[0] == 1
        assert rows["run_id"][0] == "run-1"
        assert rows["memo"][0] == "A brief memo."
