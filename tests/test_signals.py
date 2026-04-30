"""
Tests for the Stage 5 agent: sandbox executor, Coder node retries, and Critic
IC gate.  No LLM calls — the Coder tests mock the LLM; Critic tests mock the
backtest runner.

Sandbox tests:    valid code, disallowed import, syntax error, runtime error,
                  wrong return type, wrong length, missing function definition.
Coder node tests: succeed on first attempt, succeed after two failures, exhaust
                  retries and return no code.
Critic tests:     no code → reject, backtest error → reject, IC too low → reject,
                  Sharpe too low → reject, both pass → accept.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from pelican.agents.coder import coder_node
from pelican.agents.critic import IC_TSTAT_THRESHOLD, SHARPE_THRESHOLD, _make_critic_node
from pelican.agents.state import AgentState
from pelican.agents.tools.code_exec import (
    execute_signal_code,
    make_mock_df,
    needs_fundamentals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CODE = """
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    return (df["close"] / df["close_252d"] - 1.0).alias("sig")
"""

VALID_CODE_NUMPY = """
import polars as pl
import numpy as np

def compute_signal(df: pl.DataFrame) -> pl.Series:
    arr = np.log(df["close"].to_numpy() / df["close_252d"].to_numpy())
    return pl.Series("sig", arr)
"""

DISALLOWED_IMPORT_CODE = """
import os
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    return df["close"].alias("sig")
"""

DISALLOWED_FROM_IMPORT_CODE = """
from pathlib import Path
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    return df["close"].alias("sig")
"""

SYNTAX_ERROR_CODE = """
def compute_signal(df
    return df["close"]
"""

RUNTIME_ERROR_CODE = """
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    return df["nonexistent_column_xyz"].alias("sig")
"""

WRONG_TYPE_CODE = """
def compute_signal(df):
    return 42
"""

WRONG_LENGTH_CODE = """
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    return pl.Series("sig", [1.0, 2.0])
"""

NO_FUNCTION_CODE = """
import polars as pl
x = 1 + 1
"""

FUNDAMENTAL_CODE = """
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    return (1.0 / df["pe_ratio"]).alias("sig")
"""


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "theme": "buy stocks with improving earnings revision breadth",
        "generated_code": None,
        "errors": [],
        "decision": None,
        "feedback": None,
        "ic_tstat": None,
        "sharpe_net": None,
    }
    return {**base, **overrides}


def _make_backtest_result(ic_tstat: float, sharpe_net: float, n_periods: int = 12):
    r = MagicMock()
    r.ic_tstat = ic_tstat
    r.sharpe_net = sharpe_net
    r.ic_mean = ic_tstat * 0.04
    r.n_periods = n_periods
    return r


# ---------------------------------------------------------------------------
# Sandbox: execute_signal_code
# ---------------------------------------------------------------------------

class TestSandbox:
    def test_valid_code_succeeds(self):
        ok, err, fn = execute_signal_code(VALID_CODE)
        assert ok, f"expected success, got: {err}"
        assert fn is not None
        assert err == ""

    def test_valid_numpy_code_succeeds(self):
        ok, err, fn = execute_signal_code(VALID_CODE_NUMPY)
        assert ok, err

    def test_disallowed_import_rejected(self):
        ok, err, fn = execute_signal_code(DISALLOWED_IMPORT_CODE)
        assert not ok
        assert "disallowed import" in err
        assert fn is None

    def test_disallowed_from_import_rejected(self):
        ok, err, fn = execute_signal_code(DISALLOWED_FROM_IMPORT_CODE)
        assert not ok
        assert "disallowed import" in err.lower()

    def test_syntax_error_rejected(self):
        ok, err, fn = execute_signal_code(SYNTAX_ERROR_CODE)
        assert not ok
        assert "SyntaxError" in err

    def test_runtime_error_caught(self):
        ok, err, fn = execute_signal_code(RUNTIME_ERROR_CODE)
        assert not ok
        assert "runtime error" in err.lower()
        assert fn is None

    def test_wrong_return_type_rejected(self):
        ok, err, fn = execute_signal_code(WRONG_TYPE_CODE)
        assert not ok
        assert "pl.Series" in err or "Series" in err

    def test_wrong_length_rejected(self):
        mock_df = make_mock_df(n_tickers=20)
        ok, err, fn = execute_signal_code(WRONG_LENGTH_CODE, mock_df=mock_df)
        assert not ok
        assert "length" in err.lower()

    def test_no_function_rejected(self):
        ok, err, fn = execute_signal_code(NO_FUNCTION_CODE)
        assert not ok
        assert "compute_signal" in err

    def test_valid_fn_produces_correct_shape(self):
        mock_df = make_mock_df(n_tickers=30)
        ok, _, fn = execute_signal_code(VALID_CODE, mock_df=mock_df)
        assert ok
        result = fn(mock_df)
        assert isinstance(result, pl.Series)
        assert len(result) == 30

    def test_valid_fn_output_is_finite_or_null(self):
        mock_df = make_mock_df()
        ok, _, fn = execute_signal_code(VALID_CODE, mock_df=mock_df)
        assert ok
        result = fn(mock_df)
        non_null = result.drop_nulls().to_numpy()
        assert np.all(np.isfinite(non_null))

    def test_fundamental_code_succeeds(self):
        # The mock_df includes pe_ratio, so fundamental-based code is valid.
        ok, err, fn = execute_signal_code(FUNDAMENTAL_CODE)
        assert ok, err

    def test_needs_fundamentals_detection(self):
        assert needs_fundamentals(FUNDAMENTAL_CODE)
        assert not needs_fundamentals(VALID_CODE)


# ---------------------------------------------------------------------------
# Coder node: retry logic (LLM mocked)
# ---------------------------------------------------------------------------

class TestCoderNode:
    def _mock_llm_response(self, code: str):
        """Return a mock LLM response wrapping code in a ```python``` block."""
        msg = MagicMock()
        msg.content = f"Here is the implementation:\n\n```python\n{code}\n```"
        return msg

    def test_succeeds_on_first_attempt(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response(VALID_CODE)

        with patch("pelican.agents.coder._get_llm", return_value=mock_llm):
            result = coder_node(_make_state())

        assert result["generated_code"] is not None
        assert result["generated_code"].strip() == VALID_CODE.strip()
        assert mock_llm.invoke.call_count == 1

    def test_retries_on_sandbox_failure_then_succeeds(self):
        # First two attempts return invalid code; third returns valid code.
        bad_response = self._mock_llm_response(WRONG_TYPE_CODE)
        good_response = self._mock_llm_response(VALID_CODE)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [bad_response, bad_response, good_response]

        with patch("pelican.agents.coder._get_llm", return_value=mock_llm):
            result = coder_node(_make_state())

        assert result["generated_code"] is not None
        assert mock_llm.invoke.call_count == 3
        assert len(result["errors"]) == 2

    def test_errors_passed_to_subsequent_attempts(self):
        # Verify that prior errors appear in the user message on retries.
        captured_messages = []

        def fake_invoke(messages):
            captured_messages.append(messages)
            return self._mock_llm_response(VALID_CODE)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = fake_invoke

        # Pre-seed one error to simulate a prior failure.
        state = _make_state(errors=["attempt 1: some prior error"])
        with patch("pelican.agents.coder._get_llm", return_value=mock_llm):
            coder_node(state)

        user_msg = captured_messages[0][1]["content"]
        assert "some prior error" in user_msg

    def test_exhausts_retries_returns_no_code(self):
        bad_response = self._mock_llm_response(WRONG_TYPE_CODE)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = bad_response

        with patch("pelican.agents.coder._get_llm", return_value=mock_llm):
            result = coder_node(_make_state())

        assert result["generated_code"] is None
        assert mock_llm.invoke.call_count == 3  # MAX_RETRIES
        assert len(result["errors"]) == 3

    def test_llm_failure_counts_as_retry(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("network error")

        with patch("pelican.agents.coder._get_llm", return_value=mock_llm):
            result = coder_node(_make_state())

        assert result["generated_code"] is None
        assert any("LLM call failed" in e for e in result["errors"])

    def test_no_code_block_counts_as_retry(self):
        msg = MagicMock()
        msg.content = "I cannot produce code for this task."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = msg

        with patch("pelican.agents.coder._get_llm", return_value=mock_llm):
            result = coder_node(_make_state())

        assert result["generated_code"] is None
        assert any("no ```python```" in e for e in result["errors"])

    def test_state_errors_accumulate_across_retries(self):
        bad_response = self._mock_llm_response(WRONG_TYPE_CODE)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = bad_response

        with patch("pelican.agents.coder._get_llm", return_value=mock_llm):
            result = coder_node(_make_state())

        assert len(result["errors"]) == 3


# ---------------------------------------------------------------------------
# Critic node: IC / Sharpe gate (backtest mocked)
# ---------------------------------------------------------------------------

class TestCriticNode:
    def _make_store_config(self):
        store = MagicMock()
        from datetime import date
        from pelican.backtest.engine import BacktestConfig
        config = BacktestConfig(start=date(2024, 1, 1), end=date(2024, 12, 31))
        return store, config

    def test_rejects_when_no_code(self):
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        result = critic(_make_state(generated_code=None))
        assert result["decision"] == "reject"
        assert "no code" in result["feedback"]

    def test_rejects_on_backtest_error(self):
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)

        with patch(
            "pelican.agents.critic.run_backtest_with_fn",
            side_effect=ValueError("no data"),
        ):
            result = critic(_make_state(generated_code=VALID_CODE))

        assert result["decision"] == "reject"
        assert "backtest error" in result["feedback"]

    def test_rejects_ic_tstat_below_threshold(self):
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        low_ic = _make_backtest_result(ic_tstat=0.8, sharpe_net=1.0)

        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=low_ic):
            result = critic(_make_state(generated_code=VALID_CODE))

        assert result["decision"] == "reject"
        assert "IC t-stat" in result["feedback"]
        assert result["ic_tstat"] == pytest.approx(0.8)

    def test_rejects_sharpe_below_threshold(self):
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        low_sharpe = _make_backtest_result(ic_tstat=2.5, sharpe_net=0.1)

        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=low_sharpe):
            result = critic(_make_state(generated_code=VALID_CODE))

        assert result["decision"] == "reject"
        assert "Sharpe" in result["feedback"]
        assert result["sharpe_net"] == pytest.approx(0.1)

    def test_accepts_when_both_thresholds_met(self):
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        good = _make_backtest_result(ic_tstat=2.0, sharpe_net=0.6)

        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=good):
            result = critic(_make_state(generated_code=VALID_CODE))

        assert result["decision"] == "accept"
        assert result["ic_tstat"] == pytest.approx(2.0)
        assert result["sharpe_net"] == pytest.approx(0.6)

    def test_rejects_at_exact_thresholds(self):
        # Values exactly at the boundary are below-threshold (strict <).
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        borderline = _make_backtest_result(
            ic_tstat=IC_TSTAT_THRESHOLD - 1e-9,
            sharpe_net=SHARPE_THRESHOLD + 1.0,
        )
        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=borderline):
            result = critic(_make_state(generated_code=VALID_CODE))
        assert result["decision"] == "reject"

    def test_accepts_above_thresholds(self):
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        above = _make_backtest_result(
            ic_tstat=IC_TSTAT_THRESHOLD + 0.01,
            sharpe_net=SHARPE_THRESHOLD + 0.01,
        )
        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=above):
            result = critic(_make_state(generated_code=VALID_CODE))
        assert result["decision"] == "accept"

    def test_rejects_nan_ic_tstat(self):
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        nan_result = _make_backtest_result(ic_tstat=float("nan"), sharpe_net=1.0)

        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=nan_result):
            result = critic(_make_state(generated_code=VALID_CODE))

        assert result["decision"] == "reject"

    def test_metrics_present_on_accept(self):
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        good = _make_backtest_result(ic_tstat=3.0, sharpe_net=1.2)

        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=good):
            result = critic(_make_state(generated_code=VALID_CODE))

        assert result["ic_tstat"] is not None
        assert result["sharpe_net"] is not None
        assert not math.isnan(result["ic_tstat"])
        assert not math.isnan(result["sharpe_net"])
