"""DuckDB-backed data store. Single access point for all persisted data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import polars as pl

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sp500_universe (
    ticker      VARCHAR NOT NULL,
    entry_date  DATE    NOT NULL,
    exit_date   DATE,
    company     VARCHAR,
    PRIMARY KEY (ticker, entry_date)
);

CREATE TABLE IF NOT EXISTS prices (
    ticker              VARCHAR NOT NULL,
    date                DATE    NOT NULL,
    open                DOUBLE,
    high                DOUBLE,
    low                 DOUBLE,
    close               DOUBLE,
    volume              BIGINT,
    log_return_1d       DOUBLE,
    forward_return_21d  DOUBLE,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker          VARCHAR NOT NULL,
    available_date  DATE    NOT NULL,
    period_end      DATE    NOT NULL,
    market_cap      DOUBLE,
    pe_ratio        DOUBLE,
    pb_ratio        DOUBLE,
    roe             DOUBLE,
    debt_to_equity  DOUBLE,
    PRIMARY KEY (ticker, period_end)
);
"""


class DataStore:
    """Wraps a DuckDB connection with schema management and Polars I/O."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self._path = str(db_path)
        self._conn = duckdb.connect(self._path)

    def init_schema(self) -> None:
        self._conn.execute(_SCHEMA_SQL)

    def write(self, df: pl.DataFrame, table: str) -> int:
        """Insert-or-replace rows from a Polars DataFrame into `table`.

        Returns the number of rows written.
        """
        # Register the Polars DataFrame as a DuckDB view then upsert.
        # DuckDB can read Polars DataFrames directly via the Arrow protocol.
        self._conn.register("_write_tmp", df.to_arrow())
        try:
            self._conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _write_tmp")
        finally:
            self._conn.unregister("_write_tmp")
        return len(df)

    def query(self, sql: str, params: Any = None) -> pl.DataFrame:
        """Run a SQL query and return results as a Polars DataFrame."""
        rel = self._conn.execute(sql, params or [])
        # rel.arrow() returns a RecordBatchReader; .read_all() materialises an Arrow Table
        # which carries the schema even when there are zero rows.
        arrow_table = rel.arrow().read_all()
        return pl.from_arrow(arrow_table)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DataStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
