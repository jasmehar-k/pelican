"""
Run all 8 classic factors end-to-end and produce a factor correlation matrix.

Outputs:
  - Console tearsheets (IC, ICIR, Sharpe, drawdown, turnover)
  - Summary comparison table
  - factor_correlation.png

Usage:
    python scripts/run_factor_library.py
    python scripts/run_factor_library.py --start 2018-01-01 --end 2023-01-01
    python scripts/run_factor_library.py --factors MOM_1_12 LOW_VOL VALUE_PE
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

# Import factors package first — registers all 8 signals into the engine registry.
import pelican.factors  # noqa: F401

from pelican.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from pelican.data.store import DataStore
from pelican.factors import ALL_FACTORS, build_factor_correlation_matrix, plot_correlation_heatmap
from pelican.utils.config import get_settings
from pelican.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="Run classic factor library")
    p.add_argument("--start", default=s.backtest_start.isoformat())
    p.add_argument("--end", default=s.backtest_end.isoformat())
    p.add_argument("--db-path", default=str(s.duckdb_path))
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--factors", nargs="+", default=ALL_FACTORS)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-corr", action="store_true", help="skip correlation matrix")
    return p.parse_args(argv)


def print_tearsheet(result: BacktestResult) -> None:
    w = 50
    print(f"\n{'='*w}")
    print(f"  {result.signal_name} — tearsheet")
    print(f"  {result.config.start}  →  {result.config.end}")
    print(f"{'='*w}")
    print(f"  Periods            : {result.n_periods}")
    print(f"  Avg universe size  : {result.avg_universe_size:.0f}")
    print(f"  IC mean            : {result.ic_mean:+.4f}")
    print(f"  ICIR               : {result.icir:+.3f}")
    print(f"  IC t-stat          : {result.ic_tstat:+.3f}")
    print(f"  Sharpe (gross)     : {result.sharpe_gross:+.3f}")
    print(f"  Sharpe (net)       : {result.sharpe_net:+.3f}")
    print(f"  Max drawdown(gross): {result.max_drawdown_gross:+.2%}")
    print(f"  Max drawdown(net)  : {result.max_drawdown_net:+.2%}")
    print(f"  Avg turnover       : {result.avg_turnover:.2%}")
    print(f"{'='*w}")


def main(argv=None) -> None:
    configure_logging(dev=True)
    args = parse_args(argv)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        print("Run:  python scripts/seed_data.py  and  python scripts/seed_fundamentals.py  first.")
        sys.exit(1)

    store = DataStore(db_path)
    config = BacktestConfig(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        cost_bps=args.cost_bps,
    )

    results: dict[str, BacktestResult] = {}
    for signal_name in args.factors:
        print(f"\nRunning {signal_name} ...")
        t0 = time.perf_counter()
        try:
            result = run_backtest(signal_name, config, store)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        elapsed = time.perf_counter() - t0
        results[signal_name] = result
        print_tearsheet(result)
        print(f"  (completed in {elapsed:.1f}s)")

    store.close()

    if not results:
        print("\nNo results produced.")
        sys.exit(1)

    # Summary comparison table.
    print("\n" + "=" * 75)
    print(f"{'Signal':<22} {'IC':>8} {'ICIR':>8} {'Sharpe(net)':>12} {'MaxDD':>10} {'Turn':>8}")
    print("-" * 75)
    for name, r in results.items():
        print(
            f"{name:<22} {r.ic_mean:>+8.4f} {r.icir:>+8.3f} "
            f"{r.sharpe_net:>+12.3f} {r.max_drawdown_net:>+10.2%} "
            f"{r.avg_turnover:>8.2%}"
        )
    print("=" * 75)

    # Factor correlation matrix.
    if not args.no_corr and len(results) >= 2:
        print("\nBuilding factor correlation matrix ...")
        # Re-open store for correlation runs.
        store2 = DataStore(db_path)
        try:
            corr_df = build_factor_correlation_matrix(list(results.keys()), config, store2)
        finally:
            store2.close()

        # Print matrix to console.
        names = corr_df["signal"].to_list()
        print(f"\n{'':22}" + "".join(f"{n:>12}" for n in names))
        for name in names:
            row = corr_df.filter(pl.col("signal") == name)
            vals = "".join(f"{row[col].item():>12.3f}" for col in names)
            print(f"{name:<22}{vals}")

        if not args.no_plot:
            plot_correlation_heatmap(corr_df, Path("factor_correlation.png"))


if __name__ == "__main__":
    main()
