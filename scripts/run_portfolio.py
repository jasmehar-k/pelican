"""
Stage 4 demo: combine MOM_1_12 + QUALITY_LEVERAGE + LOW_VOL into a portfolio.

Runs monthly rebalance over the backtest window, combines the three signals into
an IC-weighted alpha vector, optimizes long/short weights, and prints a risk
attribution table showing systematic vs idiosyncratic variance per position.

Usage:
    python scripts/run_portfolio.py
    python scripts/run_portfolio.py --start 2024-10-01 --end 2025-11-01
    python scripts/run_portfolio.py --objective hrp
    python scripts/run_portfolio.py --signals MOM_1_12 QUALITY_LEVERAGE
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl

import pelican.factors  # noqa: F401 — registers all 8 signals
from pelican.backtest.engine import BacktestConfig, run_backtest
from pelican.backtest.signals import build_cross_section_features, get_signal
from pelican.backtest.universe import get_rebalance_dates
from pelican.data.store import DataStore
from pelican.portfolio.combiner import CombinerConfig, combine
from pelican.portfolio.optimizer import PortfolioConfig, optimize
from pelican.portfolio.risk import build_returns_wide, decompose_risk, estimate_covariance
from pelican.utils.config import get_settings
from pelican.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

DEFAULT_SIGNALS = ["MOM_1_12", "QUALITY_LEVERAGE", "LOW_VOL"]


def parse_args(argv=None) -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="Portfolio construction demo")
    p.add_argument("--start", default="2024-10-01")
    p.add_argument("--end", default="2025-11-01")
    p.add_argument("--db-path", default=str(s.duckdb_path))
    p.add_argument("--signals", nargs="+", default=DEFAULT_SIGNALS)
    p.add_argument("--objective", choices=["max_sharpe", "min_variance", "hrp"],
                   default="max_sharpe")
    p.add_argument("--lambda-risk", type=float, default=0.5)
    p.add_argument("--max-weight", type=float, default=0.05)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--no-chart", action="store_true", help="skip matplotlib output")
    return p.parse_args(argv)


def _run_signal_backtest(name: str, config: BacktestConfig, store: DataStore):
    try:
        result = run_backtest(name, config, store)
        return result
    except Exception as exc:
        log.warning("backtest failed", signal=name, error=str(exc))
        return None


def _build_cross_section_at_date(
    rebal_date: date,
    store: DataStore,
    config: BacktestConfig,
    signal_names: list[str],
) -> tuple[pl.DataFrame, list[str]] | None:
    """Build a cross-section at a single rebalance date with all signals scored."""
    from pelican.data.universe import get_universe

    tickers = get_universe(rebal_date, store)
    if len(tickers) < config.min_universe_size:
        return None

    lookback_start = rebal_date - timedelta(days=config.lookback_calendar_days)
    prices = store.query(
        "SELECT ticker, date, open, high, low, close, volume, log_return_1d, forward_return_21d "
        "FROM prices "
        "WHERE ticker IN ({tickers}) AND date BETWEEN '{start}' AND '{end}' "
        "ORDER BY ticker, date".format(
            tickers=", ".join(f"'{t}'" for t in tickers),
            start=lookback_start,
            end=rebal_date,
        )
    )
    if prices.is_empty():
        return None

    cs = build_cross_section_features(prices, rebal_date)
    if cs.is_empty():
        return None

    # Check if any signal requires fundamentals.
    needs_fund = any(get_signal(n).spec.requires_fundamentals for n in signal_names)
    if needs_fund:
        try:
            fund = store.query(
                "SELECT f.* FROM (SELECT ticker, MAX(available_date) AS best_avail "
                "FROM fundamentals WHERE available_date <= '{d}' GROUP BY ticker) AS latest "
                "JOIN fundamentals AS f ON f.ticker = latest.ticker "
                "AND f.available_date = latest.best_avail".format(d=rebal_date)
            )
            if not fund.is_empty():
                cs = cs.join(fund.drop(["available_date", "period_end"]), on="ticker", how="left")
        except Exception as exc:
            log.warning("fundamentals join failed", date=rebal_date, error=str(exc))

    return cs, tickers


def main(argv=None) -> None:
    configure_logging(dev=True)
    args = parse_args(argv)
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        sys.exit(1)

    store = DataStore(db_path)
    backtest_config = BacktestConfig(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        cost_bps=args.cost_bps,
    )

    # ── 1. Backtest each signal to get IC weights ────────────────────────────
    print("\n" + "=" * 60)
    print("  Stage 1: individual signal backtests")
    print("=" * 60)
    ic_weights: dict[str, float] = {}
    backtest_results = {}
    for name in args.signals:
        r = _run_signal_backtest(name, backtest_config, store)
        if r is not None and r.n_periods > 0 and not np.isnan(r.ic_mean):
            ic_weights[name] = max(0.0, r.ic_mean)
            backtest_results[name] = r
            print(f"  {name:<22} IC={r.ic_mean:+.4f}  ICIR={r.icir:+.3f}  "
                  f"Sharpe(net)={r.sharpe_net:+.3f}  periods={r.n_periods}")
        else:
            print(f"  {name:<22} FAILED (no usable periods)")

    if not backtest_results:
        print("\nNo signals produced backtest results. Check data coverage.")
        store.close()
        sys.exit(1)

    active_signals = list(backtest_results.keys())
    total_ic = sum(ic_weights.values())
    if total_ic > 0:
        ic_weights = {k: v / total_ic for k, v in ic_weights.items()}
    else:
        ic_weights = {k: 1.0 / len(active_signals) for k in active_signals}

    print("\n  IC weights used for combination:")
    for name, w in ic_weights.items():
        print(f"    {name:<22} {w:.3f}")

    # ── 2. Build risk model ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Stage 2: risk model (Ledoit-Wolf covariance)")
    print("=" * 60)
    prices_panel = store.query(
        "SELECT ticker, date, log_return_1d FROM prices "
        "WHERE date BETWEEN '{start}' AND '{end}' "
        "ORDER BY date".format(
            start=(date.fromisoformat(args.start) - timedelta(days=365)),
            end=args.end,
        )
    )
    all_tickers = prices_panel["ticker"].unique().to_list()
    returns_wide = build_returns_wide(prices_panel, all_tickers)
    risk_model = estimate_covariance(returns_wide, all_tickers, n_factors=10)
    print(f"  Universe: {len(risk_model.tickers)} tickers")
    print(f"  Shrinkage: {risk_model.shrinkage:.4f}")
    print(f"  Cov shape: {risk_model.cov.shape}")

    # ── 3. Compute alpha at the latest rebalance date ────────────────────────
    print("\n" + "=" * 60)
    print("  Stage 3: alpha construction")
    print("=" * 60)
    rebal_dates = get_rebalance_dates(backtest_config.start, backtest_config.end, store)
    if not rebal_dates:
        print("  No rebalance dates found in the requested window.")
        store.close()
        sys.exit(1)

    rebal_date = rebal_dates[-1]
    built = _build_cross_section_at_date(rebal_date, store, backtest_config, active_signals)
    if built is None:
        print(f"  No cross-section could be built for rebalance date {rebal_date}.")
        store.close()
        sys.exit(1)

    cs, _ = built
    print(f"  Rebalance date: {rebal_date}")
    print(f"  Cross-section: {len(cs)} tickers")

    # Score each signal.
    signal_scores: dict[str, pl.Series] = {}
    for name in active_signals:
        try:
            scores = get_signal(name).fn(cs)
            signal_scores[name] = scores
            n_scored = scores.drop_nulls().len()
            print(f"  {name:<22} scored {n_scored}/{len(cs)} tickers")
        except Exception as exc:
            log.warning("signal scoring failed", signal=name, error=str(exc))

    if not signal_scores:
        print("  No signals could be scored. Exiting.")
        store.close()
        sys.exit(1)

    # Combine into alpha.
    alpha = combine(
        signal_scores,
        ic_weights=ic_weights,
        config=CombinerConfig(method="ic_weighted", min_coverage=10),
    )
    alpha_non_null = alpha.drop_nulls()
    n_alpha = alpha_non_null.len()
    print(f"\n  Composite alpha: {n_alpha} tickers scored")
    alpha_std = alpha_non_null.std()
    if alpha_std is None:
        print("  Alpha std: n/a (no non-null alpha values)")
    else:
        print(f"  Alpha std: {float(alpha_std):.4f}")

    if n_alpha == 0:
        print("  Composite alpha is empty. Check signal coverage at the selected rebalance date.")
        store.close()
        sys.exit(1)

    # ── 4. Align alpha and risk model ────────────────────────────────────────
    cs_tickers = cs["ticker"].to_list()
    rm_ticker_set = set(risk_model.tickers)
    overlap = [t for t in cs_tickers if t in rm_ticker_set]
    if len(overlap) < 40:
        print(f"\n  WARNING: only {len(overlap)} tickers overlap alpha × risk model")

    alpha_map = dict(zip(cs_tickers, alpha.to_list()))
    alpha_aligned = pl.Series(
        "alpha",
        [alpha_map.get(t, None) for t in risk_model.tickers],
        dtype=pl.Float64,
    )

    # ── 5. Optimize ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Stage 4: portfolio optimization ({args.objective})")
    print("=" * 60)
    port_config = PortfolioConfig(
        objective=args.objective,
        lambda_risk=args.lambda_risk,
        max_weight=args.max_weight,
    )
    result = optimize(alpha_aligned, risk_model, config=port_config)

    long_pos = [(t, w) for t, w in zip(result.tickers, result.weights) if w > 1e-4]
    short_pos = [(t, w) for t, w in zip(result.tickers, result.weights) if w < -1e-4]
    long_pos.sort(key=lambda x: -x[1])
    short_pos.sort(key=lambda x: x[1])

    print(f"  Status            : {result.status}")
    print(f"  Long positions    : {len(long_pos)}")
    print(f"  Short positions   : {len(short_pos)}")
    print(f"  Expected return   : {result.expected_return:+.4f}")
    print(f"  Expected vol (ann): {(result.expected_variance ** 0.5) * np.sqrt(252):.4f}")
    print(f"  Expected Sharpe   : {result.expected_sharpe:+.3f}")

    # ── 6. Risk attribution ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Stage 5: risk decomposition")
    print("=" * 60)
    rd = result.risk_decomposition
    if rd:
        port_vol = rd.total_variance ** 0.5 * np.sqrt(252)
        sys_vol = rd.systematic_variance ** 0.5 * np.sqrt(252)
        idio_vol = rd.idiosyncratic_variance ** 0.5 * np.sqrt(252)
        print(f"  Total ann. vol    : {port_vol:.4f}")
        print(f"  Systematic vol    : {sys_vol:.4f}  ({rd.systematic_pct:.1%} of variance)")
        print(f"  Idiosyncratic vol : {idio_vol:.4f}  ({rd.idiosyncratic_pct:.1%} of variance)")

        # Top 10 contributors to total risk.
        total_contribs = np.abs(rd.factor_contributions + rd.idio_contributions)
        top_idx = np.argsort(total_contribs)[::-1][:10]
        print("\n  Top 10 risk contributors:")
        print(f"  {'Ticker':<8} {'Weight':>8} {'Sys%':>8} {'Idio%':>8} {'Total%':>8}")
        print("  " + "-" * 44)
        for i in top_idx:
            t = rd.tickers[i]
            w = rd.weights[i]
            sys_pct = rd.factor_contributions[i] / rd.total_variance * 100 if rd.total_variance else 0
            idio_pct = rd.idio_contributions[i] / rd.total_variance * 100 if rd.total_variance else 0
            tot_pct = (rd.factor_contributions[i] + rd.idio_contributions[i]) / rd.total_variance * 100 if rd.total_variance else 0
            print(f"  {t:<8} {w:>+8.3f} {sys_pct:>8.1f} {idio_pct:>8.1f} {tot_pct:>8.1f}")

    # ── 7. Top positions ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Portfolio positions (top 10 long / top 10 short)")
    print("=" * 60)
    print(f"  {'LONG':<8} {'Weight':>8}        {'SHORT':<8} {'Weight':>8}")
    print("  " + "-" * 40)
    for (lt, lw), (st, sw) in zip(long_pos[:10], short_pos[:10]):
        print(f"  {lt:<8} {lw:>+8.3f}        {st:<8} {sw:>+8.3f}")

    # ── 8. Chart (optional) ─────────────────────────────────────────────────
    if not args.no_chart and rd:
        _plot_risk_attribution(rd, output_path="risk_attribution.png")

    store.close()


def _plot_risk_attribution(rd: object, output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.warning("matplotlib not installed — skipping chart")
        return

    tickers = rd.tickers
    weights = rd.weights
    sys_contribs = rd.factor_contributions
    idio_contribs = rd.idio_contributions
    total_var = rd.total_variance

    # Show top 20 by absolute total contribution.
    total_contribs = np.abs(sys_contribs + idio_contribs)
    top_idx = np.argsort(total_contribs)[::-1][:20]

    top_tickers = [tickers[i] for i in top_idx]
    top_sys = sys_contribs[top_idx] / total_var * 100
    top_idio = idio_contribs[top_idx] / total_var * 100
    top_w = weights[top_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: stacked bar chart of risk contributions.
    x = np.arange(len(top_tickers))
    colors_sys = ["#d62728" if w < 0 else "#1f77b4" for w in top_w]
    colors_idio = ["#ff7f0e" if w < 0 else "#aec7e8" for w in top_w]

    bars_sys = ax1.bar(x, top_sys, label="Systematic", color=colors_sys, alpha=0.85)
    bars_idio = ax1.bar(x, top_idio, bottom=top_sys, label="Idiosyncratic",
                         color=colors_idio, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_tickers, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("% of portfolio variance")
    ax1.set_title("Risk contribution by position\n(blue=long, red=short)")
    ax1.axhline(0, color="black", linewidth=0.5)
    long_patch = mpatches.Patch(color="#1f77b4", label="Long — systematic")
    short_patch = mpatches.Patch(color="#d62728", label="Short — systematic")
    idio_patch = mpatches.Patch(color="#aec7e8", label="Idiosyncratic")
    ax1.legend(handles=[long_patch, short_patch, idio_patch], fontsize=8)

    # Right: pie chart of systematic vs idiosyncratic.
    sys_pct = rd.systematic_pct * 100
    idio_pct = rd.idiosyncratic_pct * 100
    ax2.pie(
        [max(0, sys_pct), max(0, idio_pct)],
        labels=[f"Systematic\n{sys_pct:.1f}%", f"Idiosyncratic\n{idio_pct:.1f}%"],
        colors=["#1f77b4", "#aec7e8"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ann_vol = rd.total_variance ** 0.5 * np.sqrt(252)
    ax2.set_title(f"Variance decomposition\nPortfolio ann. vol = {ann_vol:.2%}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Risk attribution chart saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
