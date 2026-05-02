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
    backtest_end: date = Field(default_factory=lambda: date.today())

    # arXiv rate limit
    arxiv_rate_limit_seconds: float = 3.0

    # EDGAR rate limit (SEC policy: ≤10 req/s)
    edgar_rate_limit_seconds: float = 0.11
    edgar_user_agent: str = Field(
        default="pelican-research jasmehar.kr@gmail.com",
        alias="EDGAR_USER_AGENT",
    )
    # Dedicated model for batch EDGAR tone scoring — choose one that is not
    # heavily rate-limited when called many times in succession.
    # Override via EDGAR_TONE_MODEL env var if the default is throttled.
    edgar_tone_model: str = Field(
        default="minimax/minimax-m2.5:free",
        alias="EDGAR_TONE_MODEL",
    )
    # Sleep between LLM tone-scoring calls to avoid free-tier rate limits.
    edgar_llm_rate_limit_seconds: float = 1.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
