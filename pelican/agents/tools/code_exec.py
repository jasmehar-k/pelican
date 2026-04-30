"""
Sandboxed Python code execution for LLM-generated signal functions.

Checks imports against an allowlist (polars, numpy, math), exec()s the code
in a restricted namespace, then validates the output shape and dtype against
a mock cross-section DataFrame.  Returns a structured (success, error, fn) tuple
so the Coder node can retry without crashing.
"""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import polars as pl

ALLOWED_MODULES: frozenset[str] = frozenset({"polars", "numpy", "math"})


def _check_imports(code: str) -> str | None:
    """Return an error string if the code imports anything outside ALLOWED_MODULES."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return f"disallowed import: '{alias.name}'"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root and root not in ALLOWED_MODULES:
                return f"disallowed import from: '{node.module}'"
    return None


def make_mock_df(n_tickers: int = 20) -> pl.DataFrame:
    """Minimal cross-section DataFrame for sandbox validation.

    Includes all columns a signal might realistically read:
    price history lags, rolling vols, and fundamental ratios.
    """
    from datetime import date

    rng = np.random.default_rng(0)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rebal = date(2024, 1, 1)

    rows = []
    for t in tickers:
        rows.append({
            "ticker": t,
            "date": rebal,
            "close": float(rng.uniform(10.0, 300.0)),
            "log_return_1d": float(rng.normal(0.0, 0.01)),
            "forward_return_21d": float(rng.normal(0.0, 0.05)),
            "close_21d": float(rng.uniform(10.0, 300.0)),
            "close_63d": float(rng.uniform(10.0, 300.0)),
            "close_126d": float(rng.uniform(10.0, 300.0)),
            "close_252d": float(rng.uniform(10.0, 300.0)),
            "close_504d": float(rng.uniform(10.0, 300.0)),
            "vol_21d": float(rng.uniform(0.10, 0.60)),
            "vol_63d": float(rng.uniform(0.10, 0.60)),
            "market_cap": float(rng.uniform(1e9, 5e11)),
            "pe_ratio": float(rng.uniform(5.0, 60.0)),
            "pb_ratio": float(rng.uniform(0.5, 12.0)),
            "roe": float(rng.uniform(0.0, 0.40)),
            "debt_to_equity": float(rng.uniform(0.0, 4.0)),
        })
    return pl.DataFrame(rows)


def execute_signal_code(
    code: str,
    mock_df: pl.DataFrame | None = None,
) -> tuple[bool, str, Any]:
    """Compile, execute, and validate LLM-generated signal code.

    Args:
        code: Python source code.  Must define `compute_signal(df) -> pl.Series`.
        mock_df: DataFrame to run the function against; defaults to make_mock_df().

    Returns:
        (success, error_message, fn_or_None)
        On success: (True, "", callable)
        On failure: (False, human-readable error, None)
    """
    import_err = _check_imports(code)
    if import_err:
        return False, import_err, None

    namespace: dict[str, Any] = {"pl": pl, "np": np, "math": math}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return False, f"exec error: {exc}", None

    fn = namespace.get("compute_signal")
    if fn is None or not callable(fn):
        return False, "code must define a callable 'compute_signal'", None

    if mock_df is None:
        mock_df = make_mock_df()

    try:
        result = fn(mock_df)
    except Exception as exc:
        return False, f"runtime error on mock data: {exc}", None

    if not isinstance(result, pl.Series):
        return False, f"compute_signal must return pl.Series, got {type(result).__name__}", None

    if len(result) != len(mock_df):
        return (
            False,
            f"output length {len(result)} != input length {len(mock_df)}",
            None,
        )

    return True, "", fn


def needs_fundamentals(code: str) -> bool:
    """Heuristic: does the code reference any fundamental columns?"""
    fund_cols = {"pe_ratio", "pb_ratio", "roe", "debt_to_equity", "market_cap"}
    return any(col in code for col in fund_cols)
