"""Shared API service helpers for signals, tearsheets, and portfolio optimization."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

import pelican.factors  # noqa: F401 - register factor signals
from pelican.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from pelican.backtest.signals import get_signal, list_signals
from pelican.backtest.universe import get_point_in_time_universe, get_rebalance_dates
from pelican.portfolio.combiner import CombinerConfig, combine
from pelican.portfolio.optimizer import PortfolioConfig, optimize
from pelican.portfolio.risk import build_returns_wide, estimate_covariance


def signal_spec_payload(signal_name: str) -> dict[str, Any]:
    signal = get_signal(signal_name)
    spec = signal.spec
    return {
        "name": spec.name,
        "description": spec.description,
        "lookback_days": spec.lookback_days,
        "requires_fundamentals": spec.requires_fundamentals,
        "requires_edgar": spec.requires_edgar,
        "requires_news": spec.requires_news,
        "requires_earnings": spec.requires_earnings,
        "data_deps": list(spec.data_deps),
        "edgar_data_deps": list(spec.edgar_data_deps),
        "expected_ic_range": list(spec.expected_ic_range),
        "data_frequency": spec.data_frequency,
        "min_score_coverage": spec.min_score_coverage,
        "source": spec.source,
    }


def _backtest_config(settings: Any, start: date | None, end: date | None, cost_bps: float = 2.0, impact_bps: float = 5.0) -> BacktestConfig:
    return BacktestConfig(
        start=start or settings.backtest_start,
        end=end or settings.backtest_end,
        cost_bps=cost_bps,
        impact_bps=impact_bps,
    )


def serialize_backtest_result(result: BacktestResult) -> dict[str, Any]:
    return {
        "signal_name": result.signal_name,
        "config": {
            "start": result.config.start,
            "end": result.config.end,
            "cost_bps": result.config.cost_bps,
            "impact_bps": result.config.impact_bps,
            "min_universe_size": result.config.min_universe_size,
            "min_score_coverage": result.config.min_score_coverage,
            "lookback_calendar_days": result.config.lookback_calendar_days,
            "quintile_n": result.config.quintile_n,
        },
        "metrics": {
            "ic_mean": result.ic_mean,
            "icir": result.icir,
            "ic_tstat": result.ic_tstat,
            "sharpe_gross": result.sharpe_gross,
            "sharpe_net": result.sharpe_net,
            "max_drawdown_gross": result.max_drawdown_gross,
            "max_drawdown_net": result.max_drawdown_net,
            "avg_turnover": result.avg_turnover,
            "n_periods": result.n_periods,
            "avg_universe_size": result.avg_universe_size,
        },
        "period_returns": result.period_returns.to_dicts(),
        "ic_series": result.ic_series.to_dicts(),
    }


def signal_summary_payload(settings: Any, store: Any, signal_name: str, start: date | None = None, end: date | None = None) -> dict[str, Any]:
    base = signal_spec_payload(signal_name)
    cfg = _backtest_config(settings, start, end)
    try:
        result = run_backtest(signal_name, cfg, store)
        if result.avg_universe_size < 20:
            base["stats"] = None
            base["error"] = (
                f"Insufficient coverage: {result.avg_universe_size:.0f} avg tickers/period "
                f"(need ≥20 for reliable IC estimates — seed more data)"
            )
        else:
            base["stats"] = {
                "ic_mean": result.ic_mean,
                "icir": result.icir,
                "ic_tstat": result.ic_tstat,
                "sharpe_gross": result.sharpe_gross,
                "sharpe_net": result.sharpe_net,
                "max_drawdown_gross": result.max_drawdown_gross,
                "max_drawdown_net": result.max_drawdown_net,
                "avg_turnover": result.avg_turnover,
                "n_periods": result.n_periods,
                "avg_universe_size": result.avg_universe_size,
            }
            base["error"] = None
    except Exception as exc:
        base["stats"] = None
        base["error"] = str(exc)
    return base


def build_tearsheet(settings: Any, store: Any, signal_name: str, start: date | None = None, end: date | None = None) -> dict[str, Any]:
    cfg = _backtest_config(settings, start, end)
    result = run_backtest(signal_name, cfg, store)
    # Build summary inline from the result we already have — avoids a second backtest run.
    spec_payload = signal_spec_payload(signal_name)
    if result.avg_universe_size < 20:
        spec_payload["stats"] = None
        spec_payload["error"] = (
            f"Insufficient coverage: {result.avg_universe_size:.0f} avg tickers/period "
            f"(need ≥20 for reliable IC estimates — seed more data)"
        )
    else:
        spec_payload["stats"] = {
            "ic_mean": result.ic_mean,
            "icir": result.icir,
            "ic_tstat": result.ic_tstat,
            "sharpe_gross": result.sharpe_gross,
            "sharpe_net": result.sharpe_net,
            "max_drawdown_gross": result.max_drawdown_gross,
            "max_drawdown_net": result.max_drawdown_net,
            "avg_turnover": result.avg_turnover,
            "n_periods": result.n_periods,
            "avg_universe_size": result.avg_universe_size,
        }
        spec_payload["error"] = None
    return {
        "summary": spec_payload,
        "config": {
            "start": cfg.start,
            "end": cfg.end,
            "cost_bps": cfg.cost_bps,
            "impact_bps": cfg.impact_bps,
            "min_universe_size": cfg.min_universe_size,
            "min_score_coverage": cfg.min_score_coverage,
            "lookback_calendar_days": cfg.lookback_calendar_days,
            "quintile_n": cfg.quintile_n,
        },
        "period_returns": result.period_returns.to_dicts(),
        "ic_series": result.ic_series.to_dicts(),
    }


def _build_cross_section_at_date(
    rebal_date: date,
    store: Any,
    config: BacktestConfig,
    signal_names: list[str],
) -> tuple[pl.DataFrame, list[str]] | None:
    tickers = get_point_in_time_universe(rebal_date, store)
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

    from pelican.backtest.signals import build_cross_section_features

    cs = build_cross_section_features(prices, rebal_date)
    if cs.is_empty():
        return None

    needs_fund = any(get_signal(name).spec.requires_fundamentals for name in signal_names)
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
        except Exception:
            pass

    if any(get_signal(name).spec.requires_edgar for name in signal_names):
        try:
            edgar = store.query(
                "SELECT ticker, filing_date, tone_score, tone_delta FROM edgar_sentiment "
                "WHERE filing_date <= '{d}' ORDER BY ticker, filing_date".format(d=rebal_date)
            )
            if not edgar.is_empty():
                pit_edgar = (
                    edgar.filter(pl.col("filing_date") <= rebal_date)
                    .sort("filing_date")
                    .group_by("ticker")
                    .last()
                    .drop("filing_date")
                )
                cs = cs.join(pit_edgar, on="ticker", how="left")
        except Exception:
            pass

    if any(get_signal(name).spec.requires_earnings for name in signal_names):
        try:
            earnings = store.query(
                "SELECT ticker, date, surprise_pct, reported_eps, estimated_eps "
                "FROM earnings_surprises WHERE date <= '{d}' ORDER BY ticker, date".format(d=rebal_date)
            )
            if not earnings.is_empty():
                pit_earnings = (
                    earnings.filter(pl.col("date") <= rebal_date)
                    .sort("date")
                    .group_by("ticker")
                    .last()
                    .drop("date")
                )
                cs = cs.join(pit_earnings, on="ticker", how="left")
        except Exception:
            pass

    return cs, tickers


def optimize_portfolio(settings: Any, store: Any, request: Any) -> dict[str, Any]:
    signal_names = list(request.signals)
    if not signal_names:
        raise ValueError("signals must be non-empty")

    config = BacktestConfig(
        start=request.start,
        end=request.end,
        cost_bps=request.cost_bps,
        impact_bps=getattr(request, "impact_bps", 5.0),
        min_universe_size=request.min_universe_size,
        min_score_coverage=request.min_score_coverage,
        lookback_calendar_days=request.lookback_calendar_days,
    )

    rebal_dates = get_rebalance_dates(config.start, config.end, store)
    if not rebal_dates:
        raise ValueError("No rebalance dates found in the requested window")
    rebal_date = request.rebalance_date or rebal_dates[-1]

    built = _build_cross_section_at_date(rebal_date, store, config, signal_names)
    if built is None:
        raise ValueError(f"No cross-section available for {rebal_date}")
    cs, _ = built

    # Validate all signal names exist before doing any compute.
    for name in signal_names:
        try:
            get_signal(name)
        except KeyError:
            raise ValueError(f"unknown signal: {name!r}")

    signal_scores: dict[str, pl.Series] = {}
    ic_weights: dict[str, float] = {}
    for name in signal_names:
        scores = get_signal(name).fn(cs)
        signal_scores[name] = scores
        if request.method == "ic_weighted":
            try:
                r = run_backtest(name, config, store)
                if r.n_periods > 0 and not np.isnan(r.ic_mean):
                    ic_weights[name] = max(0.0, float(r.ic_mean))
            except Exception:
                pass

    if request.method == "ic_weighted":
        total = sum(ic_weights.values())
        if total > 0:
            ic_weights = {name: weight / total for name, weight in ic_weights.items()}
        else:
            ic_weights = {name: 1.0 / len(signal_scores) for name in signal_scores}
    else:
        ic_weights = {name: 1.0 / len(signal_scores) for name in signal_scores}

    alpha = combine(
        signal_scores,
        ic_weights=ic_weights,
        config=CombinerConfig(method=request.method, min_coverage=10),
    )

    prices_panel = store.query(
        "SELECT ticker, date, log_return_1d FROM prices "
        "WHERE date BETWEEN '{start}' AND '{end}' ORDER BY date".format(
            start=rebal_date - timedelta(days=365),
            end=rebal_date,
        )
    )
    if prices_panel.is_empty():
        raise ValueError("No price data available for portfolio optimization")

    all_tickers = prices_panel["ticker"].unique().to_list()
    returns_wide = build_returns_wide(prices_panel, all_tickers)
    risk_model = estimate_covariance(returns_wide, all_tickers, n_factors=10)

    alpha_map = dict(zip(cs["ticker"].to_list(), alpha.to_list()))
    alpha_aligned = pl.Series(
        "alpha",
        [alpha_map.get(t, None) for t in risk_model.tickers],
        dtype=pl.Float64,
    )

    result = optimize(
        alpha_aligned,
        risk_model,
        config=PortfolioConfig(
            objective=request.objective,
            lambda_risk=request.lambda_risk,
            max_weight=request.max_weight,
            turnover_limit=request.turnover_limit,
            cost_bps=config.cost_bps,
        ),
    )

    risk = result.risk_decomposition
    risk_payload = None
    if risk is not None:
        risk_payload = {
            "tickers": risk.tickers,
            "weights": risk.weights.tolist(),
            "total_variance": risk.total_variance,
            "systematic_variance": risk.systematic_variance,
            "idiosyncratic_variance": risk.idiosyncratic_variance,
            "systematic_pct": risk.systematic_pct,
            "idiosyncratic_pct": risk.idiosyncratic_pct,
            "factor_contributions": risk.factor_contributions.tolist(),
            "idio_contributions": risk.idio_contributions.tolist(),
        }

    return {
        "signals": signal_names,
        "rebalance_date": rebal_date,
        "objective": request.objective,
        "method": request.method,
        "status": result.status,
        "expected_return": result.expected_return,
        "expected_variance": result.expected_variance,
        "expected_sharpe": result.expected_sharpe,
        "positions": [
            {"ticker": ticker, "weight": float(weight)}
            for ticker, weight in zip(result.tickers, result.weights)
            if abs(weight) > 1e-8
        ],
        "risk_decomposition": risk_payload,
        "ic_weights": ic_weights,
        "alpha_coverage": int(alpha.drop_nulls().len()),
    }


def signal_names() -> list[str]:
    return list_signals()


def run_portfolio_backtest(settings: Any, store: Any, request: Any) -> dict[str, Any]:
    """Walk-forward IC-weighted portfolio backtest over the requested date range.

    Walk-forward optimizer-based portfolio backtest.

    At each rebalance date: builds the cross-section, combines signal scores into
    an alpha vector (IC-weighted), solves the CVXPy mean-variance QP, then
    measures actual portfolio P&L as w @ forward_return_21d minus Almgren-Chriss
    transaction costs.  Tracks the previous period's weights for turnover continuity.
    """
    import math as _math

    from pelican.backtest.metrics import compute_max_drawdown, compute_sharpe

    signal_names_list = list(request.signals)
    if not signal_names_list:
        raise ValueError("signals must be non-empty")

    cfg = _backtest_config(
        settings,
        getattr(request, "start", None),
        getattr(request, "end", None),
        cost_bps=getattr(request, "cost_bps", 2.0),
        impact_bps=getattr(request, "impact_bps", 5.0),
    )

    rebal_dates = get_rebalance_dates(cfg.start, cfg.end, store)
    if not rebal_dates:
        raise ValueError("No rebalance dates found in the requested window")

    # Pre-compute IC weights from per-signal historical backtests (reused every period).
    ic_raw: dict[str, float] = {}
    for name in signal_names_list:
        try:
            r = run_backtest(name, cfg, store)
            if r.n_periods > 0 and not _math.isnan(r.ic_mean):
                ic_raw[name] = max(0.0, r.ic_mean)
        except Exception:
            pass
    total_ic = sum(ic_raw.values())
    ic_weights: dict[str, float] = (
        {n: w / total_ic for n, w in ic_raw.items()} if total_ic > 0
        else {n: 1.0 / len(signal_names_list) for n in signal_names_list}
    )

    equity_curve = []
    equity = 1.0
    net_returns: list[float] = []
    prev_weights: np.ndarray | None = None
    prev_tickers: list[str] = []

    for rebal_date in rebal_dates:
        built = _build_cross_section_at_date(rebal_date, store, cfg, signal_names_list)
        if built is None:
            continue
        cs, tickers = built

        # Compute alpha from combined signal scores.
        signal_scores: dict[str, pl.Series] = {}
        for name in signal_names_list:
            try:
                signal_scores[name] = get_signal(name).fn(cs)
            except Exception:
                pass
        if not signal_scores:
            continue

        alpha = combine(
            signal_scores,
            ic_weights={n: ic_weights.get(n, 0.0) for n in signal_scores},
            config=CombinerConfig(method="ic_weighted", min_coverage=10),
        )

        # Build risk model from 252 trading days up to rebal_date.
        lookback_start = rebal_date - timedelta(days=400)
        prices_panel = store.query(
            "SELECT ticker, date, log_return_1d FROM prices "
            "WHERE ticker IN ({t}) AND date BETWEEN '{s}' AND '{e}' ORDER BY date".format(
                t=", ".join(f"'{tk}'" for tk in tickers),
                s=lookback_start,
                e=rebal_date,
            )
        )
        if prices_panel.is_empty():
            continue
        try:
            risk_model = estimate_covariance(build_returns_wide(prices_panel, tickers), tickers, n_factors=10)
        except Exception:
            continue

        # Align alpha to risk model ticker order.
        alpha_map = dict(zip(cs["ticker"].to_list(), alpha.to_list()))
        alpha_aligned = pl.Series(
            "alpha",
            [alpha_map.get(t) for t in risk_model.tickers],
            dtype=pl.Float64,
        )

        # Align previous weights to current tickers (universe changes each period).
        aligned_prev: np.ndarray | None = None
        if prev_weights is not None and prev_tickers:
            prev_map = dict(zip(prev_tickers, prev_weights))
            aligned_prev = np.array([prev_map.get(t, 0.0) for t in risk_model.tickers])

        try:
            result = optimize(
                alpha_aligned,
                risk_model,
                config=PortfolioConfig(
                    objective=getattr(request, "objective", "max_sharpe"),
                    lambda_risk=getattr(request, "lambda_risk", 1.0),
                    max_weight=getattr(request, "max_weight", 0.05),
                    cost_bps=cfg.cost_bps,
                ),
                prev_weights=aligned_prev,
            )
        except Exception:
            continue

        prev_weights = result.weights
        prev_tickers = risk_model.tickers

        # Fetch forward returns for this rebalance date.
        if not risk_model.tickers:
            continue
        fwd = store.query(
            "SELECT ticker, forward_return_21d FROM prices "
            "WHERE date = '{d}' AND ticker IN ({t})".format(
                d=rebal_date,
                t=", ".join(f"'{tk}'" for tk in risk_model.tickers),
            )
        )
        if fwd.is_empty():
            continue
        fwd_map = dict(zip(fwd["ticker"].to_list(), fwd["forward_return_21d"].to_list()))
        fwd_returns = np.array([fwd_map.get(t, 0.0) or 0.0 for t in risk_model.tickers])

        # Almgren-Chriss transaction cost on actual portfolio turnover.
        if aligned_prev is not None:
            turnover = float(np.abs(result.weights - aligned_prev).sum()) / 2.0
        else:
            turnover = float(np.abs(result.weights).sum()) / 2.0
        cost = (turnover * cfg.cost_bps + turnover ** 1.5 * cfg.impact_bps) / 10_000

        period_return = float(result.weights @ fwd_returns) - cost
        equity *= 1.0 + period_return
        net_returns.append(period_return)
        equity_curve.append({
            "date": rebal_date,
            "portfolio_return": period_return,
            "cumulative_return": equity - 1.0,
        })

    series = pl.Series(net_returns) if net_returns else pl.Series([], dtype=pl.Float64)
    return {
        "signals": signal_names_list,
        "start": cfg.start,
        "end": cfg.end,
        "n_periods": len(equity_curve),
        "sharpe_net": float(compute_sharpe(series)) if net_returns else None,
        "max_drawdown": float(compute_max_drawdown(series)) if net_returns else None,
        "total_return": equity - 1.0 if equity_curve else None,
        "ic_weights": ic_weights,
        "equity_curve": equity_curve,
    }