"""
CLI runner for the Stage 5 Coder → Critic agent pipeline.

Runs synchronously (no API server needed) and prints the decision, feedback,
and generated code to stdout.  Useful for development and spot-checking new
factor descriptions against the live database.

Usage:
    python scripts/run_agent.py --theme "buy stocks with improving earnings revision breadth"
    python scripts/run_agent.py --theme "low volatility anomaly" --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pelican.agents.graph import build_graph, initial_state
from pelican.backtest.engine import BacktestConfig
from pelican.data.store import DataStore
from pelican.utils.config import get_settings
from pelican.utils.logging import configure_logging


def parse_args(argv=None) -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="Run the Coder → Critic agent pipeline")
    p.add_argument("--theme", required=True, help="Natural language factor description")
    p.add_argument("--start", default="2025-03-01",
                   help="Backtest start date (YYYY-MM-DD). "
                        "Fundamental signals need ≥2025-03-01 (first dense quarter).")
    p.add_argument("--end", default="2025-11-01", help="Backtest end date (YYYY-MM-DD)")
    p.add_argument("--db-path", default=str(s.duckdb_path))
    p.add_argument(
        "--model",
        default=None,
        help="OpenRouter model ID (default: settings.openrouter_model). "
             "Example: deepseek/deepseek-chat:free",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    configure_logging(dev=True)
    args = parse_args(argv)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        sys.exit(1)

    store = DataStore(db_path)
    config = BacktestConfig(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )

    graph = build_graph(store, config, model=args.model)
    state = initial_state(args.theme)

    print(f"\nTheme: {args.theme}")
    print(f"Backtest window: {args.start} → {args.end}\n")
    print("=" * 60)

    result = graph.invoke(state)

    print("\n" + "=" * 60)
    print(f"Decision  : {result['decision']}")
    print(f"Feedback  : {result['feedback']}")
    if result.get("ic_tstat") is not None:
        print(f"IC t-stat : {result['ic_tstat']:.2f}")
    if result.get("sharpe_net") is not None:
        print(f"Net Sharpe: {result['sharpe_net']:.2f}")

    if result.get("errors"):
        print("\nRetry errors:")
        for e in result["errors"]:
            print(f"  {e}")

    if result["generated_code"]:
        print("\n--- Generated code ---")
        print(result["generated_code"])
    else:
        print("\n(no code generated)")

    store.close()
    sys.exit(0 if result["decision"] == "accept" else 1)


if __name__ == "__main__":
    main()
