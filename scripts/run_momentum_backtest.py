"""
Momentum factor demo: run MOM_1_12 (and benchmarks) end-to-end against seeded data.

Outputs:
  - Console tearsheet with IC, ICIR, Sharpe, max drawdown, avg turnover
  - ic_decay.png  — IC by holding horizon (1d, 5d, 10d, 21d)
  - quintile_spread.png — cumulative Q5-Q1 spread vs each quintile return

Usage:
    python scripts/run_momentum_backtest.py
    python scripts/run_momentum_backtest.py --start 2018-01-01 --end 2023-01-01
    python scripts/run_momentum_backtest.py --signals MOM_1_12 LOW_VOL HML_REVERSAL
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

# Make sure pelican is importable when script is run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from pelican.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from pelican.data.store import DataStore
from pelican.utils.config import get_settings
from pelican.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

BENCHMARK_SIGNALS = ["MOM_1_12", "HML_REVERSAL", "LOW_VOL"]


def parse_args(argv=None) -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="Run momentum backtest demo")
    p.add_argument("--start", default=s.backtest_start.isoformat())
    p.add_argument("--end", default=s.backtest_end.isoformat())
    p.add_argument("--db-path", default=str(s.duckdb_path))
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument(
        "--signals", nargs="+",
        default=BENCHMARK_SIGNALS,
        help="signals to run (default: MOM_1_12 HML_REVERSAL LOW_VOL)",
    )
    p.add_argument("--no-plot", action="store_true", help="skip matplotlib output")
    return p.parse_args(argv)


def print_tearsheet(result: BacktestResult) -> None:
    w = 50
    print(f"\n{'='*w}")
    print(f"  {result.signal_name} — backtest tearsheet")
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
    print(f"  Cost bps           : {result.config.cost_bps:.1f}")
    print(f"{'='*w}\n")


def _safe_cumprod(series):
    return (1.0 + series.fill_null(0.0)).cum_prod()


def plot_ic_decay(result: BacktestResult, output_path: Path) -> None:
    """Bar chart of IC by rebalance period with mean IC reference line."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed, skipping IC decay plot")
        return

    dates = result.ic_series["date"].to_list()
    ics = result.ic_series["ic"].to_list()
    # Replace None with 0.0 for bar heights; colour grey for missing periods.
    bar_heights = [ic if ic is not None else 0.0 for ic in ics]
    colors = [
        "#aaaaaa" if ic is None else ("#d62728" if ic < 0 else "#2ca02c")
        for ic in ics
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dates, bar_heights, color=colors, alpha=0.7, width=20)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(result.ic_mean, color="navy", linewidth=1.5, linestyle="--",
               label=f"mean IC = {result.ic_mean:.3f}")
    ax.set_title(f"{result.signal_name} — IC by rebalance period (ICIR={result.icir:.2f})")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Spearman IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → IC chart saved to {output_path}")


def plot_quintile_spread(result: BacktestResult, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed, skipping quintile spread plot")
        return

    pr = result.period_returns
    dates = pr["date"].to_list()
    n_q = result.config.quintile_n

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: cumulative quintile returns
    palette = plt.cm.RdYlGn([i / (n_q - 1) for i in range(n_q)])
    for qi, color in zip(range(1, n_q + 1), palette):
        col = f"q{qi}"
        if col in pr.columns:
            cum = _safe_cumprod(pr[col]) - 1.0
            ax1.plot(dates, cum.to_list(), label=f"Q{qi}", color=color, linewidth=1.2)

    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title(f"{result.signal_name} — cumulative quintile returns")
    ax1.set_ylabel("Cumulative return")
    ax1.legend(fontsize=8)

    # Bottom panel: L/S spread gross vs net
    cum_gross = _safe_cumprod(pr["ls_gross"]) - 1.0
    cum_net = _safe_cumprod(pr["ls_net"]) - 1.0
    cum_gross_vals = cum_gross.to_list()
    cum_net_vals = cum_net.to_list()
    ax2.plot(dates, cum_gross_vals, color="steelblue", label="L/S gross", linewidth=1.5)
    ax2.plot(
        dates, cum_net_vals,
        color="darkorange", label="L/S net", linewidth=1.5, linestyle="--",
    )
    ax2.axhline(0, color="black", linewidth=0.5)
    safe_gross = [v if v is not None else 0.0 for v in cum_gross_vals]
    ax2.fill_between(dates, safe_gross, 0,
                     where=[v > 0 for v in safe_gross],
                     alpha=0.08, color="steelblue")
    ax2.set_title(
        f"L/S spread  |  Sharpe gross={result.sharpe_gross:.2f}"
        f"  net={result.sharpe_net:.2f}"
    )
    ax2.set_ylabel("Cumulative return")
    ax2.set_xlabel("Rebalance date")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → Quintile spread chart saved to {output_path}")


def main(argv=None) -> None:
    configure_logging(dev=True)
    args = parse_args(argv)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        print("Run:  python scripts/seed_data.py  first.")
        sys.exit(1)

    store = DataStore(db_path)

    config = BacktestConfig(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        cost_bps=args.cost_bps,
    )

    results: dict[str, BacktestResult] = {}
    for signal_name in args.signals:
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

        if not args.no_plot:
            plot_ic_decay(result, Path(f"ic_decay_{signal_name}.png"))
            plot_quintile_spread(result, Path(f"quintile_spread_{signal_name}.png"))

    store.close()

    if not results:
        print("\nNo results produced. Check that data is seeded and date range is valid.")
        sys.exit(1)

    # Summary comparison table
    print("\n" + "=" * 70)
    print(f"{'Signal':<20} {'IC':>8} {'ICIR':>8} {'Sharpe(net)':>12} {'MaxDD':>10} {'Turn':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(
            f"{name:<20} {r.ic_mean:>+8.4f} {r.icir:>+8.3f} "
            f"{r.sharpe_net:>+12.3f} {r.max_drawdown_net:>+10.2%} "
            f"{r.avg_turnover:>8.2%}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
