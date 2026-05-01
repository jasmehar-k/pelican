from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from pelican.utils.config import get_settings
from scripts import seed_data


def test_parse_args_overrides_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    args = seed_data.parse_args(
        [
            "--start",
            "2020-01-01",
            "--end",
            "2021-01-01",
            "--batch-size",
            "25",
            "--db-path",
            "./alt.duckdb",
        ]
    )
    assert args.start == "2020-01-01"
    assert args.end == "2021-01-01"
    assert args.batch_size == 25
    assert args.db_path == "./alt.duckdb"


def test_seed_main_loads_ever_members_and_is_rerunnable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    events: list[tuple] = []

    class FakeLogger:
        def info(self, event: str, **kwargs) -> None:
            events.append(("log", event, kwargs))

    class FakeStore:
        def __init__(self, db_path: Path) -> None:
            self.db_path = Path(db_path)
            events.append(("store_init", str(self.db_path)))

        def init_schema(self) -> None:
            events.append(("init_schema",))

        def query(self, sql: str, params=None) -> pl.DataFrame:
            events.append(("query", sql.strip(), params))
            if "SELECT DISTINCT ticker FROM sp500_universe" in sql:
                return pl.DataFrame({"ticker": ["AAPL", "DEAD"]})
            if "SELECT count(*) AS n FROM prices" in sql:
                return pl.DataFrame({"n": [60]})
            raise AssertionError(f"Unexpected SQL: {sql}")

        def __enter__(self) -> "FakeStore":
            events.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append(("exit",))

    def fake_load_universe(store: FakeStore) -> None:
        events.append(("load_universe", store.db_path.name))

    def fake_get_universe(query_date: date, store: FakeStore) -> list[str]:
        events.append(("get_universe", query_date.isoformat()))
        return ["AAPL"]

    def fake_load_prices(
        store: FakeStore,
        tickers: list[str],
        start: date,
        end: date,
        batch_size: int,
    ) -> None:
        events.append(("load_prices", tickers, start.isoformat(), end.isoformat(), batch_size))

    monkeypatch.setattr("pelican.utils.logging.configure_logging", lambda dev=True: events.append(("configure_logging", dev)))
    monkeypatch.setattr("pelican.utils.logging.get_logger", lambda name="seed": FakeLogger())
    monkeypatch.setattr("pelican.data.store.DataStore", FakeStore)
    monkeypatch.setattr("pelican.data.universe.load_universe", fake_load_universe)
    monkeypatch.setattr("pelican.data.universe.get_universe", fake_get_universe)
    monkeypatch.setattr("pelican.data.prices.load_prices", fake_load_prices)

    argv = [
        "--start",
        "2020-01-01",
        "--end",
        "2021-01-01",
        "--batch-size",
        "25",
        "--db-path",
        "./data/test.duckdb",
    ]
    seed_data.main(argv)
    seed_data.main(argv)

    load_prices_events = [event for event in events if event[0] == "load_prices"]
    assert load_prices_events == [
        ("load_prices", ["AAPL", "DEAD"], "2020-01-01", "2021-01-01", 25),
        ("load_prices", ["AAPL", "DEAD"], "2020-01-01", "2021-01-01", 25),
    ]

    first_universe = events.index(("load_universe", "test.duckdb"))
    first_prices = events.index(("load_prices", ["AAPL", "DEAD"], "2020-01-01", "2021-01-01", 25))
    assert first_universe < first_prices
