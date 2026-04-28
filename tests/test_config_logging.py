from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from pelican.utils.config import get_settings
from pelican.utils.logging import configure_logging, get_logger


def test_get_settings_loads_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "DUCKDB_PATH=./custom/pelican.duckdb",
                "DATA_DIR=./cache-dir",
                "API_HOST=127.0.0.1",
                "API_PORT=9001",
                "BACKTEST_START=2020-01-01",
                "BACKTEST_END=2021-01-01",
            ]
        ),
        encoding="utf-8",
    )
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.duckdb_path == Path("./custom/pelican.duckdb")
    assert settings.data_dir == Path("./cache-dir")
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 9001
    assert settings.backtest_start.isoformat() == "2020-01-01"
    assert settings.backtest_end.isoformat() == "2021-01-01"


def test_get_settings_cache_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    monkeypatch.setenv("API_PORT", "8001")
    first = get_settings()
    monkeypatch.setenv("API_PORT", "8002")
    second = get_settings()
    assert first is second
    assert second.api_port == 8001

    get_settings.cache_clear()
    refreshed = get_settings()
    assert refreshed.api_port == 8002


def test_seed_defaults_follow_settings_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "DUCKDB_PATH=./env.duckdb\nBACKTEST_START=2019-01-01\nBACKTEST_END=2019-12-31\n",
        encoding="utf-8",
    )
    get_settings.cache_clear()

    from scripts import seed_data

    args = seed_data.parse_args([])
    assert args.db_path == "env.duckdb"  # Path('./env.duckdb') normalises to 'env.duckdb'
    assert args.start == "2019-01-01"
    assert args.end == "2019-12-31"


def test_configure_logging_dev_writes_console_output(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    configure_logging(dev=True)
    get_logger("test.dev").info("dev message", answer=42)
    output = stdout.getvalue()
    assert "dev message" in output
    assert "answer" in output


def test_configure_logging_prod_writes_json(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    configure_logging(dev=False)
    get_logger("test.json").info("json message", answer=42)
    payload = json.loads(stdout.getvalue().strip())
    assert payload["event"] == "json message"
    assert payload["answer"] == 42
    assert payload["level"] == "info"


def test_configure_logging_can_be_called_repeatedly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    configure_logging(dev=True)
    configure_logging(dev=False)
    get_logger("test.repeat").info("repeat-safe")
    payload = json.loads(stdout.getvalue().strip())
    assert payload["event"] == "repeat-safe"
