"""
Backtest API router.

Endpoints:
  POST /backtest/run       — run the engine for a given signal name and date range
  GET  /backtest/{run_id}  — poll a running backtest job for status/results
  GET  /backtest/compare   — side-by-side metric comparison across multiple signals
"""
