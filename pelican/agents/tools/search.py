"""arXiv search tool for the Researcher agent.

Queries the arXiv Atom export API and returns a compact list of paper summaries.
The helper rate-limits requests according to the configured arxiv_rate_limit_seconds
setting so repeated agent runs do not violate the free API's terms.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from typing import TypedDict

import httpx

from pelican.utils.config import get_settings

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_CATEGORIES = "cat:q-fin.PM OR cat:q-fin.ST OR cat:econ.GN OR cat:q-fin.TR"

_last_req_time = 0.0


class SearchResult(TypedDict):
    title: str
    authors: list[str]
    abstract: str
    arxiv_id: str
    url: str


def _rate_limit() -> None:
    global _last_req_time
    settings = get_settings()
    now = time.monotonic()
    elapsed = now - _last_req_time
    wait = settings.arxiv_rate_limit_seconds - elapsed
    if _last_req_time and wait > 0:
        time.sleep(wait)
    _last_req_time = time.monotonic()


def _normalize_arxiv_id(raw_id: str) -> str:
    arxiv_id = raw_id.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", arxiv_id)


def _parse_entry(entry: ET.Element, namespace: dict[str, str]) -> SearchResult:
    title = (entry.findtext("atom:title", default="", namespaces=namespace) or "").strip()
    abstract = (entry.findtext("atom:summary", default="", namespaces=namespace) or "").strip()
    authors = [
        (author.findtext("atom:name", default="", namespaces=namespace) or "").strip()
        for author in entry.findall("atom:author", namespace)
    ]
    authors = [author for author in authors if author]
    arxiv_id = _normalize_arxiv_id(
        entry.findtext("atom:id", default="", namespaces=namespace) or ""
    )
    return {
        "title": title,
        "authors": authors,
        "abstract": abstract[:800],
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


_RETRY_DELAYS = (5, 15, 30)   # seconds to wait before each retry attempt


def _build_query(query: str) -> str:
    """Scope each word of the user query to abstract/title fields.

    Plain keyword search matches anywhere in arXiv metadata, which pulls in
    papers where e.g. "quality" appears in a CS engineering context.  Using
    abs:/ti: field selectors keeps results anchored to the paper's content.
    """
    words = [w for w in re.split(r"\s+", query.strip()) if len(w) > 2]
    if not words:
        words = [query]
    field_clauses = " AND ".join(f"(abs:{w} OR ti:{w})" for w in words)
    return f"({field_clauses}) AND ({ARXIV_CATEGORIES})"


def search_arxiv(query: str, max_results: int = 10) -> list[SearchResult]:
    _rate_limit()
    search_query = _build_query(query)
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    last_exc: Exception | None = None
    for attempt, backoff in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            response = httpx.get(ARXIV_API_URL, params=params, timeout=60)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", namespace)
            return [_parse_entry(e, namespace) for e in entries] if entries else []
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            if backoff is not None:
                time.sleep(backoff)
        except Exception:
            raise

    raise httpx.ReadTimeout(
        f"arXiv search timed out after {len(_RETRY_DELAYS) + 1} attempts"
    ) from last_exc
