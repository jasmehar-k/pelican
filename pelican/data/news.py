"""
News sentiment data ingestion.

Fetches and preprocesses news articles and earnings call transcripts.
Produces per-ticker daily sentiment scores (positive/negative/neutral)
using an LLM or lightweight classifier. Scores are stored with the
publication timestamp as the point-in-time anchor.

Pipeline:
  1. Fetch recent news headlines for each ticker via yfinance (free, no API key).
  2. Score each headline in [-1, +1] using an LLM via OpenRouter.
  3. Aggregate per-ticker per-day: avg_score and n_articles.
  4. Write to the `news_sentiment` DuckDB table.

Seeding is incremental: run seed_news.py periodically to accumulate history.
The backtest engine joins this table point-in-time when
SignalSpec.requires_news=True.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any

import polars as pl

from pelican.utils.config import get_settings
from pelican.utils.logging import get_logger

log = get_logger(__name__)

_RETRY_DELAYS = (15, 45, 90)

_HEADLINE_SYSTEM = """\
Rate the sentiment of this financial news headline on a scale from -1.0 to 1.0.

Return ONLY a JSON object:
{"sentiment": <float from -1.0 to 1.0>}

Scale:
-1.0  very negative: earnings miss, scandal, bankruptcy, lawsuit, downgrade
-0.5  moderately negative
 0.0  neutral / factual / mixed
+0.5  moderately positive
+1.0  very positive: earnings beat, upgrade, record revenue, positive guidance
"""

_SENTIMENT_RE = re.compile(r'"sentiment"\s*:\s*(-?\d+(?:\.\d+)?)')


def _get_llm(model: str | None = None):
    from langchain_openai import ChatOpenAI
    s = get_settings()
    return ChatOpenAI(
        model=model or s.edgar_tone_model,
        base_url=s.openrouter_base_url,
        api_key=s.openrouter_api_key,
        temperature=0.0,
        max_tokens=32,
    )


def _score_headline(text: str, model: str | None = None) -> float | None:
    """Score a single news headline in [-1, +1].  Returns None on failure."""
    if not text or not text.strip():
        return None
    for _attempt, backoff in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            llm = _get_llm(model)
            resp = llm.invoke([
                {"role": "system", "content": _HEADLINE_SYSTEM},
                {"role": "user", "content": text[:500]},
            ])
            raw = resp.content.strip()
            m = _SENTIMENT_RE.search(raw)
            if m:
                return max(-1.0, min(1.0, float(m.group(1))))
            log.warning("news: could not parse sentiment JSON", raw=raw[:80])
            return None
        except Exception as exc:
            err = str(exc)
            if "429" in err and backoff is not None:
                log.warning("news: rate limited, retrying", backoff=backoff, attempt=_attempt)
                time.sleep(backoff)
                continue
            log.warning("news: headline scoring failed", error=err[:200])
            return None
    return None


def fetch_ticker_news(ticker: str) -> list[dict]:
    """Return recent news items for `ticker` from yfinance.

    Each item is a dict with at minimum:
        title (str), providerPublishTime (int, unix epoch UTC)
    """
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
        return [item for item in items if item.get("title") and item.get("providerPublishTime")]
    except Exception as exc:
        log.warning("news: yfinance fetch failed", ticker=ticker, error=str(exc))
        return []


def seed_news_sentiment(
    store: Any,
    tickers: list[str],
    *,
    model: str | None = None,
    on_progress: Any | None = None,
) -> int:
    """Fetch, score, and store news_sentiment rows for `tickers`.

    Args:
        store: DataStore instance.
        tickers: List of tickers to process.
        model: OpenRouter model ID for sentiment scoring.
        on_progress: Optional callable(ticker: str) called after each ticker.

    Returns:
        Number of rows written to news_sentiment.
    """
    s = get_settings()
    model_name = model or s.edgar_tone_model

    # Skip (ticker, date) pairs already successfully scored.
    try:
        existing_df = store.query(
            "SELECT ticker, date FROM news_sentiment WHERE avg_score IS NOT NULL"
        )
        existing_keys: set[tuple] = {
            (row["ticker"], row["date"])
            for row in existing_df.to_dicts()
        }
    except Exception:
        existing_keys = set()

    all_rows: list[dict] = []

    for ticker in tickers:
        items = fetch_ticker_news(ticker)
        if not items:
            if on_progress:
                on_progress(ticker)
            continue

        # Group headlines by publication date.
        by_date: dict[date, list[str]] = defaultdict(list)
        for item in items:
            pub_ts = item["providerPublishTime"]
            pub_date = datetime.fromtimestamp(pub_ts, tz=timezone.utc).date()
            by_date[pub_date].append(item["title"])

        for pub_date, headlines in sorted(by_date.items()):
            if (ticker, pub_date) in existing_keys:
                log.debug("news: skipping already-scored", ticker=ticker, date=pub_date)
                continue

            scores: list[float] = []
            for headline in headlines:
                score = _score_headline(headline, model=model_name)
                if score is not None:
                    scores.append(score)
                time.sleep(s.edgar_llm_rate_limit_seconds)

            if not scores:
                continue

            avg = sum(scores) / len(scores)
            all_rows.append({
                "ticker": ticker,
                "date": pub_date,
                "avg_score": avg,
                "n_articles": len(scores),
                "model": model_name,
            })
            log.info("news: scored", ticker=ticker, date=pub_date,
                     n=len(scores), avg_score=round(avg, 3))

        if on_progress:
            on_progress(ticker)

    if not all_rows:
        return 0

    df = pl.DataFrame(all_rows).select([
        pl.col("ticker").cast(pl.Utf8),
        pl.col("date").cast(pl.Date),
        pl.col("avg_score").cast(pl.Float64),
        pl.col("n_articles").cast(pl.Int32),
        pl.col("model").cast(pl.Utf8),
    ])
    rows_written = store.store_news_scores(df)
    log.info("news: wrote rows", n=rows_written)
    return rows_written
