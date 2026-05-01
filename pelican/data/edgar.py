"""
SEC EDGAR filing ingestion — download, MD&A extraction, LLM tone scoring.

Three-step pipeline for each S&P 500 ticker:
  1. Fetch filing metadata from EDGAR submissions API (no API key required,
     but SEC policy requires a User-Agent header with name + email).
  2. Download the primary HTML document; cache locally so re-runs skip
     already-fetched filings.
  3. Extract the MD&A section, score its tone with an LLM in [-1, +1].

YoY tone delta is computed after scoring:
    delta(t) = tone_score(t) - tone_score(same filing_type, prior year ±45 days)

Rows are written to the `edgar_sentiment` DuckDB table keyed on
(ticker, period_end, filing_type).  The backtest engine joins this table
as a point-in-time alternative data signal when SignalSpec.requires_edgar=True.
"""

from __future__ import annotations

import re
import time
from datetime import date, timedelta
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, TypedDict

import httpx
import polars as pl

from pelican.utils.config import get_settings
from pelican.utils.logging import get_logger

log = get_logger(__name__)

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_DOCUMENT_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/{doc}"
)

_RETRY_DELAYS = (5, 15, 45)
_MDA_MAX_CHARS = 4_000


class FilingMeta(TypedDict):
    ticker: str
    cik: str
    accession: str
    filing_date: date
    period_end: date
    filing_type: str
    primary_doc: str


class EdgarScore(TypedDict):
    ticker: str
    filing_date: date
    period_end: date
    filing_type: str
    tone_score: float | None
    tone_delta: float | None
    model: str | None


# ---------------------------------------------------------------------------
# EDGAR API helpers
# ---------------------------------------------------------------------------

def _edgar_get(url: str, user_agent: str) -> httpx.Response:
    """HTTP GET with User-Agent header and retry on 429 / 5xx."""
    headers = {"User-Agent": user_agent, "Accept": "application/json, text/html"}
    last_exc: Exception | None = None
    for _attempt, backoff in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
            if resp.status_code == 429 and backoff is not None:
                log.warning("edgar 429, backing off", backoff=backoff)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            if backoff is not None:
                time.sleep(backoff)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500 and backoff is not None:
                last_exc = exc
                time.sleep(backoff)
            else:
                raise
    raise RuntimeError(f"EDGAR request failed after retries: {url}") from last_exc


@lru_cache(maxsize=1)
def _load_cik_map(user_agent: str) -> dict[str, str]:
    """Return {TICKER_UPPER: zero_padded_10_digit_cik} from SEC company_tickers.json."""
    resp = _edgar_get(_COMPANY_TICKERS_URL, user_agent)
    data = resp.json()
    return {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in data.values()
    }


def get_cik(ticker: str, user_agent: str | None = None) -> str | None:
    """Return the zero-padded 10-digit CIK for `ticker`, or None if not found."""
    s = get_settings()
    ua = user_agent or s.edgar_user_agent
    mapping = _load_cik_map(ua)
    return mapping.get(ticker.upper())


def fetch_filing_metadata(
    cik: str,
    filing_type: str,
    user_agent: str,
    after: date | None = None,
    before: date | None = None,
    limit: int = 8,
) -> list[FilingMeta]:
    """Return recent filings of `filing_type` for `cik` from EDGAR submissions API.

    The API returns filings in reverse chronological order.  We take up to
    `limit` entries whose filing_date falls within [after, before].
    """
    url = _SUBMISSIONS_URL.format(cik=cik)
    s = get_settings()
    time.sleep(s.edgar_rate_limit_seconds)
    resp = _edgar_get(url, user_agent)
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    results: list[FilingMeta] = []
    for form, fd, rd, acc, doc in zip(
        forms, filing_dates, report_dates, accessions, primary_docs
    ):
        if form.upper() != filing_type.upper():
            continue
        if not fd or not acc or not doc:
            continue
        try:
            filing_date = date.fromisoformat(fd)
            period_end = date.fromisoformat(rd) if rd else filing_date - timedelta(days=1)
        except ValueError:
            continue
        if after and filing_date < after:
            continue
        if before and filing_date > before:
            continue
        results.append(FilingMeta(
            ticker="",  # filled in by caller
            cik=cik,
            accession=acc,
            filing_date=filing_date,
            period_end=period_end,
            filing_type=filing_type.upper(),
            primary_doc=doc,
        ))
        if len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# HTML stripping + MD&A extraction
# ---------------------------------------------------------------------------

class _StripHTML(HTMLParser):
    """Minimal HTML-to-plaintext converter for SEC filing documents."""

    _SKIP_TAGS = frozenset({"script", "style", "head"})
    _BLOCK_TAGS = frozenset({"p", "div", "br", "li", "td", "tr", "h1", "h2", "h3", "h4"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP_TAGS:
            self._skip = True
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines = [ln.strip() for ln in raw.splitlines()]
        cleaned: list[str] = []
        blanks = 0
        for ln in lines:
            if ln:
                blanks = 0
                cleaned.append(ln)
            else:
                blanks += 1
                if blanks <= 2:
                    cleaned.append("")
        return "\n".join(cleaned)


def _strip_html(html: str) -> str:
    parser = _StripHTML()
    try:
        parser.feed(html)
        return parser.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)


# Match "Item 7." or "Item 7 " at the start of a line (section header, not a reference).
_MDA_START = re.compile(r"(?:^|\n)\s*item\s+7[^a-z0-9]", re.IGNORECASE)
# Item 7A or Item 8 at the start of a line marks the end of MD&A.
_MDA_END = re.compile(r"(?:^|\n)\s*item\s+(?:7a\b|8\b)", re.IGNORECASE)


_IXBRL_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*:[a-zA-Z][^>]*?>", re.DOTALL)


_MDA_MIN_SECTION = 150  # chars; shorter = likely a table-of-contents entry


def extract_mda(html_text: str) -> str:
    """Extract MD&A section from filing HTML.

    Returns up to _MDA_MAX_CHARS of the MD&A section, or the first
    _MDA_MAX_CHARS of body text if no Item 7 marker is found.
    """
    # Strip inline XBRL namespace tags (e.g. <ix:nonfraction>, <dei:...>)
    # while preserving their text content so MD&A prose survives.
    html_text = _IXBRL_TAG.sub("", html_text)
    plain = _strip_html(html_text)

    # Scan all Item 7 matches; skip table-of-contents entries (< _MDA_MIN_SECTION chars).
    best: str = ""
    pos = 0
    while True:
        start_m = _MDA_START.search(plain, pos)
        if start_m is None:
            break
        start_idx = start_m.start()
        end_m = _MDA_END.search(plain, start_idx + 20)
        end_idx = end_m.start() if end_m else start_idx + _MDA_MAX_CHARS * 2
        section = plain[start_idx:end_idx][:_MDA_MAX_CHARS]
        if len(section) >= _MDA_MIN_SECTION:
            best = section
            break  # take the first substantial hit
        pos = start_idx + 1

    return best or plain[:_MDA_MAX_CHARS]


# ---------------------------------------------------------------------------
# Local document cache + download
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, ticker: str, filing_type: str, accession: str) -> Path:
    return cache_dir / ticker / filing_type / f"{accession.replace('-', '')}.htm"


def fetch_primary_document(
    cik: str,
    accession: str,
    primary_doc: str,
    cache_dir: Path,
    ticker: str,
    filing_type: str,
    user_agent: str,
) -> str | None:
    """Download primary filing document and return raw HTML.  Results are cached."""
    dest = _cache_path(cache_dir, ticker, filing_type, accession)
    if dest.exists():
        return dest.read_text(encoding="utf-8", errors="replace")

    accession_nodashes = accession.replace("-", "")
    url = _DOCUMENT_URL.format(
        cik=cik, accession_nodashes=accession_nodashes, doc=primary_doc
    )
    s = get_settings()
    time.sleep(s.edgar_rate_limit_seconds)
    try:
        resp = _edgar_get(url, user_agent)
    except Exception as exc:
        log.warning("edgar document fetch failed", url=url, error=str(exc))
        return None

    html = resp.text
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    return html


# ---------------------------------------------------------------------------
# LLM tone scoring
# ---------------------------------------------------------------------------

_TONE_SYSTEM = """\
You are a financial analyst specialising in SEC filings.
Read the MD&A excerpt and rate management's overall tone and outlook.

Return ONLY a JSON object:
{"tone_score": <float from -1.0 to 1.0>}

Scale:
-1.0 very negative: severe warnings, major risks, deteriorating outlook
-0.5 moderately negative
 0.0 neutral / balanced
+0.5 moderately positive
+1.0 very positive: strong growth, confident guidance, improving fundamentals
"""

_TONE_RE = re.compile(r'"tone_score"\s*:\s*(-?\d+(?:\.\d+)?)')
_TONE_RETRY_DELAYS = (15, 45, 90)


def _get_llm(model: str | None = None):
    from langchain_openai import ChatOpenAI
    s = get_settings()
    # Use the dedicated edgar tone model when no override is given — it is less
    # rate-limited for batch use than the general openrouter_model.
    return ChatOpenAI(
        model=model or s.edgar_tone_model,
        base_url=s.openrouter_base_url,
        api_key=s.openrouter_api_key,
        temperature=0.0,
        max_tokens=48,
    )


def score_tone(mda_text: str, model: str | None = None) -> float | None:
    """Score MD&A tone with an LLM.  Returns a float in [-1, +1] or None on failure.

    Retries up to 3 times on 429 rate-limit responses (free-tier models are
    frequently throttled when called in rapid succession).
    """
    if not mda_text or not mda_text.strip():
        return None
    truncated = mda_text[:3000]
    for _attempt, backoff in enumerate((*_TONE_RETRY_DELAYS, None), start=1):
        try:
            llm = _get_llm(model)
            resp = llm.invoke([
                {"role": "system", "content": _TONE_SYSTEM},
                {"role": "user", "content": truncated},
            ])
            raw = resp.content.strip()
            m = _TONE_RE.search(raw)
            if m:
                return max(-1.0, min(1.0, float(m.group(1))))
            log.warning("tone scoring: could not parse JSON", raw=raw[:120])
            return None
        except Exception as exc:
            err = str(exc)
            if "429" in err and backoff is not None:
                log.warning("tone scoring rate limited, retrying",
                            backoff=backoff, attempt=_attempt)
                time.sleep(backoff)
                continue
            log.warning("tone scoring failed", error=err[:200])
            return None
    return None


# ---------------------------------------------------------------------------
# Tone delta computation
# ---------------------------------------------------------------------------

def _compute_tone_deltas(records: list[dict]) -> list[dict]:
    """Attach `tone_delta` = tone_score(current) - tone_score(prior year ±45 days).

    Looks for a prior-year filing within ±45 days of period_end - 365 days.
    Sets tone_delta = None when no prior-year match is found.
    """
    lookup: dict[tuple, float] = {
        (r["ticker"], r["filing_type"], r["period_end"]): r["tone_score"]
        for r in records
        if r.get("tone_score") is not None
    }
    updated = []
    for rec in records:
        tone_delta: float | None = None
        if rec.get("tone_score") is not None:
            target = rec["period_end"] - timedelta(days=365)
            for delta_days in range(-45, 46):
                candidate = target + timedelta(days=delta_days)
                key = (rec["ticker"], rec["filing_type"], candidate)
                if key in lookup:
                    tone_delta = rec["tone_score"] - lookup[key]
                    break
        updated.append({**rec, "tone_delta": tone_delta})
    return updated


# ---------------------------------------------------------------------------
# Main seeding entry point
# ---------------------------------------------------------------------------

def seed_edgar_sentiment(
    store: Any,
    tickers: list[str],
    filing_types: tuple[str, ...] = ("10-K", "10-Q"),
    *,
    model: str | None = None,
    after: date | None = None,
    before: date | None = None,
    limit: int = 8,
    cache_dir: Path | None = None,
    user_agent: str | None = None,
    on_progress: Any | None = None,
) -> int:
    """Download, score, and store edgar_sentiment rows for `tickers`.

    Args:
        store: DataStore instance.
        tickers: List of S&P 500 tickers to process.
        filing_types: Tuple of filing types to download (default: 10-K and 10-Q).
        model: OpenRouter model ID for tone scoring.
        after: Only include filings on or after this date.  Default: 3 years ago.
        before: Only include filings on or before this date.  Default: today.
        limit: Max filings per (ticker, filing_type) pair.
        cache_dir: Local cache directory.  Defaults to DATA_DIR/edgar.
        user_agent: SEC User-Agent header.  Defaults to settings.edgar_user_agent.
        on_progress: Optional callable(ticker: str) called after each ticker.

    Returns:
        Number of rows written to edgar_sentiment.
    """
    s = get_settings()
    ua = user_agent or s.edgar_user_agent
    if cache_dir is None:
        cache_dir = Path(s.data_dir) / "edgar"
    if before is None:
        before = date.today()
    if after is None:
        after = before - timedelta(days=3 * 365)

    # Only skip filings that were already successfully scored (tone_score IS NOT NULL).
    # Rows with null tone_score (e.g. from a prior rate-limited run) will be re-scored.
    try:
        existing_df = store.query(
            "SELECT ticker, period_end, filing_type FROM edgar_sentiment "
            "WHERE tone_score IS NOT NULL"
        )
        existing_keys: set[tuple] = {
            (row["ticker"], row["period_end"], row["filing_type"])
            for row in existing_df.to_dicts()
        }
    except Exception:
        existing_keys = set()

    all_records: list[dict] = []

    for ticker in tickers:
        cik = get_cik(ticker, ua)
        if cik is None:
            log.warning("edgar: CIK not found", ticker=ticker)
            if on_progress:
                on_progress(ticker)
            continue

        ticker_records: list[dict] = []

        for filing_type in filing_types:
            try:
                filings = fetch_filing_metadata(
                    cik, filing_type, ua, after=after, before=before, limit=limit,
                )
            except Exception as exc:
                log.warning("edgar: metadata fetch failed", ticker=ticker,
                            filing_type=filing_type, error=str(exc))
                continue

            for fm in filings:
                key = (ticker, fm["period_end"], filing_type)
                if key in existing_keys:
                    log.debug("edgar: skipping already-scored", ticker=ticker,
                              period_end=fm["period_end"])
                    continue

                html = fetch_primary_document(
                    cik=cik,
                    accession=fm["accession"],
                    primary_doc=fm["primary_doc"],
                    cache_dir=cache_dir,
                    ticker=ticker,
                    filing_type=filing_type,
                    user_agent=ua,
                )
                if html is None:
                    continue

                mda = extract_mda(html)
                tone = score_tone(mda, model)
                # Space out LLM calls to stay within free-tier rate limits.
                time.sleep(get_settings().edgar_llm_rate_limit_seconds)
                ticker_records.append({
                    "ticker": ticker,
                    "filing_date": fm["filing_date"],
                    "period_end": fm["period_end"],
                    "filing_type": filing_type,
                    "tone_score": tone,
                    "tone_delta": None,
                    "model": model or s.openrouter_model,
                })
                log.info("edgar: scored filing", ticker=ticker,
                         period_end=fm["period_end"], tone_score=tone)

        # Pull prior-year filings from DB so tone_delta can reference them.
        try:
            prior_df = store.query(
                "SELECT ticker, filing_date, period_end, filing_type, tone_score "
                "FROM edgar_sentiment WHERE ticker = ?",
                [ticker],
            )
            for row in prior_df.to_dicts():
                ticker_records.append({**row, "tone_delta": None, "model": None})
        except Exception:
            pass

        all_records.extend(ticker_records)
        if on_progress:
            on_progress(ticker)

    if not all_records:
        return 0

    with_deltas = _compute_tone_deltas(all_records)
    # Only write newly-scored rows (model != None).
    new_rows = [r for r in with_deltas if r.get("model") is not None]
    if not new_rows:
        return 0

    df = pl.DataFrame(new_rows).select([
        pl.col("ticker").cast(pl.Utf8),
        pl.col("filing_date").cast(pl.Date),
        pl.col("period_end").cast(pl.Date),
        pl.col("filing_type").cast(pl.Utf8),
        pl.col("tone_score").cast(pl.Float64),
        pl.col("tone_delta").cast(pl.Float64),
        pl.col("model").cast(pl.Utf8),
    ])
    rows_written = store.store_edgar_scores(df)
    log.info("edgar: wrote rows", n=rows_written)
    return rows_written
