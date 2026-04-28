"""Application settings via pydantic-settings, read from environment / .env file."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Storage
    duckdb_path: Path = Path("./data/pelican.duckdb")
    data_dir: Path = Path("./data/cache")

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Backtest defaults
    backtest_start: date = date(2014, 1, 1)
    backtest_end: date = date(2024, 1, 1)

    # arXiv rate limit
    arxiv_rate_limit_seconds: float = 3.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
