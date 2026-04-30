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

# --- failure-mode fixtures ---

LOOK_AHEAD_CODE = """
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    return df["forward_return_21d"].alias("sig")
"""

ALL_NULL_CODE = """
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    return pl.Series("sig", [None] * len(df), dtype=pl.Float64)
"""

SHIFT_OVER_TICKER_CODE = """
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    # Wrong: each ticker has 1 row, so .shift(1).over("ticker") returns all null.
    return df.with_columns(
        pl.col("close").shift(1).over("ticker").alias("sig")
    )["sig"]
"""

INF_OUTPUT_CODE = """
import polars as pl
import numpy as np

def compute_signal(df: pl.DataFrame) -> pl.Series:
    arr = np.full(len(df), np.inf)
    return pl.Series("sig", arr)
"""

CONSTANT_SIGNAL_CODE = """
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    return pl.Series("sig", [1.0] * len(df))
"""

FEW_NULL_CODE = """
import polars as pl

def compute_signal(df: pl.DataFrame) -> pl.Series:
    scores = (df["close"] / df["close_252d"] - 1.0).to_list()
    scores[0] = None
    scores[1] = None
    scores[2] = None
    return pl.Series("sig", scores)
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

    # --- future-data / look-ahead ---

    def test_look_ahead_forward_return_rejected(self):
        ok, err, fn = execute_signal_code(LOOK_AHEAD_CODE)
        assert not ok
        assert "look-ahead" in err
        assert "forward_return_21d" in err
        assert fn is None

    # --- all-null output ---

    def test_all_null_output_rejected(self):
        ok, err, fn = execute_signal_code(ALL_NULL_CODE)
        assert not ok
        assert "null" in err
        assert fn is None

    def test_shift_over_ticker_all_null_rejected(self):
        # Each ticker group has exactly 1 row; shift(1) within it gives null for every ticker.
        ok, err, fn = execute_signal_code(SHIFT_OVER_TICKER_CODE)
        assert not ok
        assert "null" in err
        assert fn is None

    # --- inf/nan output ---

    def test_inf_output_rejected(self):
        ok, err, fn = execute_signal_code(INF_OUTPUT_CODE)
        assert not ok
        assert "inf" in err.lower()
        assert fn is None

    # --- valid edge cases that sandbox should accept ---

    def test_constant_signal_passes_sandbox(self):
        # A constant score is semantically useless (IC ≈ 0) but not a sandbox violation.
        # The critic gates on IC threshold — that's its job.
        ok, err, fn = execute_signal_code(CONSTANT_SIGNAL_CODE)
        assert ok, f"constant signal should pass sandbox: {err}"

    def test_few_nulls_below_threshold_passes(self):
        # 3/20 = 15% null — well under the 80% rejection threshold.
        ok, err, fn = execute_signal_code(FEW_NULL_CODE)
        assert ok, f"15% null should pass sandbox: {err}"


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

    def test_rate_limit_error_recorded_in_errors(self):
        # A 429 exception must be labelled "rate limited (429)" so the user can distinguish
        # transient quota exhaustion from real code failures.
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("HTTP 429: rate limit exceeded")

        with patch("pelican.agents.coder._get_llm", return_value=mock_llm):
            result = coder_node(_make_state())

        assert result["generated_code"] is None
        assert any("rate limited (429)" in e for e in result["errors"])

    def test_null_heavy_output_triggers_coder_retry(self):
        # Code that produces >80% null (e.g. .shift().over()) must be rejected by sandbox
        # and cause the coder to retry with that error in context.
        null_response = self._mock_llm_response(SHIFT_OVER_TICKER_CODE)
        good_response = self._mock_llm_response(VALID_CODE)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [null_response, good_response]

        with patch("pelican.agents.coder._get_llm", return_value=mock_llm):
            result = coder_node(_make_state())

        assert result["generated_code"] is not None
        assert mock_llm.invoke.call_count == 2
        assert any("null" in e.lower() for e in result["errors"])


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

    def _make_fund_row(self, lo, hi):
        """Return a mock store.query() result for the coverage check."""
        row = MagicMock()
        row.is_empty.return_value = False
        row.__getitem__ = lambda self, key: {"lo": [lo], "hi": [hi]}[key]
        return row

    def test_rejects_when_fundamentals_not_in_db(self):
        store, config = self._make_store_config()
        # Coverage query returns empty (no non-null roe rows above threshold).
        row = MagicMock()
        row.is_empty.return_value = False
        row.__getitem__ = lambda self, key: {"lo": [None], "hi": [None]}[key]
        store.query.return_value = row
        critic = _make_critic_node(store, config)
        result = critic(_make_state(generated_code=FUNDAMENTAL_CODE))
        assert result["decision"] == "reject"
        assert "no fundamentals data" in result["feedback"]

    def test_rejects_when_window_outside_fundamentals_range(self):
        from datetime import date
        from pelican.backtest.engine import BacktestConfig

        store = MagicMock()
        # First dense fundamentals date is Feb 2025.
        store.query.return_value = self._make_fund_row(
            date(2025, 2, 14), date(2026, 5, 15)
        )
        # Backtest ends before fundamentals start.
        config = BacktestConfig(start=date(2023, 1, 1), end=date(2024, 1, 1))
        critic = _make_critic_node(store, config)
        result = critic(_make_state(generated_code=FUNDAMENTAL_CODE))
        assert result["decision"] == "reject"
        assert "no overlap" in result["feedback"]
        assert "2025-02-14" in result["feedback"]

    def test_trims_start_when_partially_outside_fundamentals_range(self):
        from datetime import date
        from pelican.backtest.engine import BacktestConfig

        fund_lo = date(2025, 2, 14)
        fund_hi = date(2026, 5, 15)
        store = MagicMock()
        store.query.return_value = self._make_fund_row(fund_lo, fund_hi)

        config = BacktestConfig(start=date(2024, 1, 1), end=date(2025, 6, 1))
        critic = _make_critic_node(store, config)

        good = _make_backtest_result(ic_tstat=2.0, sharpe_net=0.6)
        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=good) as mock_bt:
            result = critic(_make_state(generated_code=FUNDAMENTAL_CODE))

        called_config = mock_bt.call_args[0][2]
        assert called_config.start == fund_lo
        assert result["decision"] == "accept"

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

    def test_rejects_trivially_zero_ic_tstat(self):
        # A constant or random-noise signal produces IC ≈ 0, which must be rejected.
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        zero_ic = _make_backtest_result(ic_tstat=0.0, sharpe_net=0.0)

        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=zero_ic):
            result = critic(_make_state(generated_code=VALID_CODE))

        assert result["decision"] == "reject"
        assert "IC t-stat" in result["feedback"]
        assert result["ic_tstat"] == pytest.approx(0.0)

    def test_rejects_nan_sharpe(self):
        # NaN sharpe (e.g. zero-variance L/S spread) must be rejected even if IC is fine.
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        nan_sharpe = _make_backtest_result(ic_tstat=2.5, sharpe_net=float("nan"))

        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=nan_sharpe):
            result = critic(_make_state(generated_code=VALID_CODE))

        assert result["decision"] == "reject"
        assert "Sharpe" in result["feedback"]

    def test_low_period_hint_in_feedback(self):
        # When n_periods < 6 and IC is below threshold the feedback should explain why.
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        short_window = _make_backtest_result(ic_tstat=0.5, sharpe_net=0.5, n_periods=4)

        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=short_window):
            result = critic(_make_state(generated_code=VALID_CODE))

        assert result["decision"] == "reject"
        assert "fewer than 6 periods" in result["feedback"]

    def test_ic_tstat_present_in_state_when_sharpe_fails(self):
        # When IC passes but Sharpe fails, ic_tstat must still be returned in the state
        # so the caller can display it in the rejection panel.
        store, config = self._make_store_config()
        critic = _make_critic_node(store, config)
        low_sharpe = _make_backtest_result(ic_tstat=2.5, sharpe_net=0.1)

        with patch("pelican.agents.critic.run_backtest_with_fn", return_value=low_sharpe):
            result = critic(_make_state(generated_code=VALID_CODE))

        assert result["decision"] == "reject"
        assert result["ic_tstat"] == pytest.approx(2.5)
        assert result["sharpe_net"] == pytest.approx(0.1)
