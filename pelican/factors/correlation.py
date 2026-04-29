"""
Factor correlation matrix.

Runs backtests for a list of signals and computes pairwise Spearman
correlations of their IC time series. Low cross-factor correlation
confirms orthogonality — a prerequisite for meaningful combination.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pelican.backtest.engine import BacktestConfig, run_backtest
from pelican.backtest.metrics import spearman_ic
from pelican.data.store import DataStore
from pelican.utils.logging import get_logger

log = get_logger(__name__)


def build_factor_correlation_matrix(
    signal_names: list[str],
    config: BacktestConfig,
    store: DataStore,
) -> pl.DataFrame:
    """Run each signal's backtest and return a pairwise Spearman IC correlation matrix.

    IC series are aligned on date before computing correlations.
    Returns a DataFrame with columns: [signal] + one column per signal name.
    Signals that fail to backtest are omitted with a warning.
    """
    ic_series: dict[str, pl.DataFrame] = {}
    for name in signal_names:
        try:
            result = run_backtest(name, config, store)
            ic_series[name] = result.ic_series  # columns: date, ic
            log.info("backtest complete for correlation", signal=name)
        except Exception as exc:
            log.warning("skipping signal for correlation matrix", signal=name, error=str(exc))

    if not ic_series:
        raise ValueError("No signals produced IC series — cannot build correlation matrix.")

    names = list(ic_series.keys())

    # All signals are run over the same config so they share the same rebalance dates.
    # Left join preserves all dates from the first series and avoids duplicate date columns.
    aligned = ic_series[names[0]].rename({"ic": names[0]})
    for name in names[1:]:
        aligned = aligned.join(
            ic_series[name].rename({"ic": name}),
            on="date",
            how="left",
        )
    aligned = aligned.sort("date")

    # Compute pairwise Spearman correlations using the existing spearman_ic helper.
    n = len(names)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                matrix[i][j] = 1.0
            elif j > i:
                si = aligned[ni].fill_nan(None).drop_nulls()
                sj = aligned[nj].fill_nan(None).drop_nulls()
                # Align pairwise by dropping positions where either is null.
                pair = pl.DataFrame({"a": aligned[ni], "b": aligned[nj]}).drop_nulls()
                if len(pair) < 3:
                    corr = float("nan")
                else:
                    corr = spearman_ic(pair["a"], pair["b"])
                matrix[i][j] = corr
                matrix[j][i] = corr

    rows = [{"signal": names[i], **{names[j]: matrix[i][j] for j in range(n)}} for i in range(n)]
    return pl.DataFrame(rows)


def plot_correlation_heatmap(corr_df: pl.DataFrame, output_path: Path) -> None:
    """Save an annotated heatmap of the factor correlation matrix."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        log.warning("matplotlib not installed, skipping correlation heatmap")
        return

    signal_names = corr_df["signal"].to_list()
    n = len(signal_names)
    matrix = [
        [corr_df.filter(pl.col("signal") == name)[col].item() for col in signal_names]
        for name in signal_names
    ]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(signal_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(signal_names, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_title("Factor IC Correlation Matrix", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → Correlation heatmap saved to {output_path}")
