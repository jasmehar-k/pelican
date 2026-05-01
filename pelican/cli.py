"""Typer CLI entry point for the ``pelican`` command."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer

from pelican.utils.config import get_settings
from pelican.utils.logging import configure_logging

app = typer.Typer(help="Pelican command line interface.")
data_app = typer.Typer(help="Data ingestion and validation commands.")
app.add_typer(data_app, name="data")


def _run_seed(
    start: str,
    end: str,
    batch_size: int,
    db_path: str,
    seed_main: Callable[[list[str] | None], None] | None = None,
) -> None:
    configure_logging(dev=True)
    if seed_main is None:
        from scripts.seed_data import main as seed_main

    seed_main(
        [
            "--start",
            start,
            "--end",
            end,
            "--batch-size",
            str(batch_size),
            "--db-path",
            db_path,
        ]
    )


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Host interface to bind."),
    port: int | None = typer.Option(None, help="Port to bind."),
    reload: bool = typer.Option(False, help="Enable auto-reload for local development."),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    configure_logging(dev=reload)
    settings = get_settings()
    uvicorn.run(
        "pelican.api.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


@data_app.command("seed")
def data_seed(
    start: str | None = typer.Option(None, help="Start date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, help="End date (YYYY-MM-DD)."),
    batch_size: int = typer.Option(50, help="Tickers per yfinance batch."),
    db_path: Path | None = typer.Option(None, help="DuckDB database path."),
) -> None:
    """Download and persist the market data foundation."""
    settings = get_settings()
    _run_seed(
        start=start or settings.backtest_start.isoformat(),
        end=end or settings.backtest_end.isoformat(),
        batch_size=batch_size,
        db_path=str(db_path or settings.duckdb_path),
    )
