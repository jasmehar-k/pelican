"""DuckDB-backed data store. Single access point for all persisted data."""

from __future__ import annotations

from pathlib import Path
import json
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

CREATE TABLE IF NOT EXISTS research_log (
    run_id             VARCHAR,
    ts                 TIMESTAMPTZ DEFAULT current_timestamp,
    theme              VARCHAR,
    arxiv_ids          VARCHAR[],
    papers             JSON,
    signal_hypothesis  TEXT,
    generated_code     TEXT,
    decision           VARCHAR,
    ic_tstat           DOUBLE,
    sharpe_net         DOUBLE,
    feedback           TEXT,
    retry_count        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS signal_memos (
    run_id      VARCHAR,
    ts          TIMESTAMPTZ DEFAULT current_timestamp,
    theme       VARCHAR,
    decision    VARCHAR,
    ic_tstat    DOUBLE,
    sharpe_net  DOUBLE,
    retry_count INTEGER,
    arxiv_ids   VARCHAR[],
    memo        TEXT
);

CREATE TABLE IF NOT EXISTS edgar_sentiment (
    ticker       VARCHAR NOT NULL,
    filing_date  DATE    NOT NULL,
    period_end   DATE    NOT NULL,
    filing_type  VARCHAR NOT NULL,
    tone_score   DOUBLE,
    tone_delta   DOUBLE,
    model        VARCHAR,
    PRIMARY KEY (ticker, period_end, filing_type)
);
"""


class DataStore:
    """Wraps a DuckDB connection with schema management and Polars I/O."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self._path = str(db_path)
        self._conn = duckdb.connect(self._path)

    def init_schema(self) -> None:
        self._conn.execute(_SCHEMA_SQL)
        self._conn.execute("ALTER TABLE research_log ADD COLUMN IF NOT EXISTS papers JSON")

    def log_run(self, state: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO research_log (
                run_id, theme, arxiv_ids, papers, signal_hypothesis, generated_code,
                decision, ic_tstat, sharpe_net, feedback, retry_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                state.get("run_id"),
                state.get("theme"),
                state.get("arxiv_ids") or [],
                json.dumps(state.get("papers") or []),
                state.get("signal_hypothesis"),
                state.get("generated_code"),
                state.get("decision"),
                state.get("ic_tstat"),
                state.get("sharpe_net"),
                state.get("feedback"),
                state.get("retry_count", 0),
            ],
        )

    def get_recent_research_log(self, limit: int = 50) -> pl.DataFrame:
        return self.query(
            """
            SELECT run_id, ts, theme, arxiv_ids, signal_hypothesis,
                   generated_code, decision, ic_tstat, sharpe_net,
                   feedback, retry_count
            FROM research_log
            ORDER BY ts DESC
            LIMIT ?
            """,
            [limit],
        )

    def get_research_log_entry(self, run_id: str) -> pl.DataFrame:
        return self.query(
            """
            SELECT run_id, ts, theme, arxiv_ids, signal_hypothesis,
                   generated_code, decision, ic_tstat, sharpe_net,
                   feedback, retry_count
            FROM research_log
            WHERE run_id = ?
            LIMIT 1
            """,
            [run_id],
        )

    def log_memo(self, state: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO signal_memos (
                run_id, theme, decision, ic_tstat, sharpe_net, retry_count, arxiv_ids, memo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                state.get("run_id"),
                state.get("theme"),
                state.get("decision"),
                state.get("ic_tstat"),
                state.get("sharpe_net"),
                state.get("retry_count", 0),
                state.get("arxiv_ids") or [],
                state.get("memo"),
            ],
        )

    def store_edgar_scores(self, df: pl.DataFrame) -> int:
        """Upsert rows into edgar_sentiment.  Returns number of rows written."""
        return self.write(df, "edgar_sentiment")

    def get_edgar_coverage(self) -> pl.DataFrame:
        """Return per-ticker filing counts and date range from edgar_sentiment."""
        return self.query(
            """
            SELECT ticker,
                   COUNT(*)           AS n_filings,
                   MIN(filing_date)   AS first_filing,
                   MAX(filing_date)   AS last_filing
            FROM edgar_sentiment
            GROUP BY ticker
            ORDER BY ticker
            """
        )

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
