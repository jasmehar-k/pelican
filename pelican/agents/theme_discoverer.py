"""Theme discoverer agent node — auto-proposes a factor theme from recent literature.

Fetches the most recently submitted q-fin arXiv papers (no keyword filter, sorted
by date) and asks the LLM to propose distinct factor themes grounded in those papers
while avoiding mechanisms already covered by signals in the registry.

Falls back to a hardcoded list of classic themes if arXiv is unreachable or the
LLM call fails, so the pipeline always gets a non-empty theme to proceed with.
"""

from __future__ import annotations

import re
from pathlib import Path

from pelican.agents.state import AgentState
from pelican.agents.tools.search import SearchResult, search_arxiv_recent
from pelican.backtest.signals import list_signals
from pelican.utils.config import get_settings
from pelican.utils.logging import get_logger

log = get_logger(__name__)

_FALLBACK_THEMES = [
    "short-term reversal",
    "low volatility anomaly",
    "quality factor: return on equity",
]


def _get_llm(model: str | None = None):
    from langchain_openai import ChatOpenAI

    s = get_settings()
    return ChatOpenAI(
        model=model or s.openrouter_model,
        base_url=s.openrouter_base_url,
        api_key=s.openrouter_api_key,
        temperature=0.7,
        max_tokens=512,
    )


def _load_system_prompt() -> str:
    return (Path(__file__).parent / "prompts" / "theme_discoverer.md").read_text()


def _format_papers(papers: list[SearchResult]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:3]) if p["authors"] else "Unknown"
        lines.append(
            f"{i}. {p['title']}\n"
            f"   arXiv: {p['arxiv_id']}  Authors: {authors}\n"
            f"   {p['abstract']}"
        )
    return "\n\n".join(lines)


def _build_user_message(papers: list[SearchResult], existing: list[str]) -> str:
    parts = [
        "Recent q-fin papers (newest first):",
        "",
        _format_papers(papers),
        "",
        "Signals already in registry — do NOT reproduce these mechanisms:",
        ", ".join(existing) if existing else "(none)",
        "",
        "Propose exactly 3 candidate factor themes. Return ONLY:",
        "THEME_1: <one sentence: economic rationale + paper citation + data columns>",
        "THEME_2: <one sentence: different mechanism, different paper, different columns>",
        "THEME_3: <one sentence: different mechanism, different paper, different columns>",
    ]
    return "\n".join(parts)


def _parse_themes(text: str) -> list[str]:
    themes = []
    for i in range(1, 4):
        m = re.search(rf"^THEME_{i}:\s*(.+)", text, re.MULTILINE | re.IGNORECASE)
        if m:
            t = m.group(1).strip()
            if len(t) > 10:
                themes.append(t)
    return themes


def discover_theme(n_papers: int = 15, model: str | None = None) -> str:
    """Fetch recent arXiv papers, ask LLM to propose themes, return the first."""
    try:
        papers = search_arxiv_recent(n=n_papers)
    except Exception as exc:
        log.warning("theme_discoverer: arXiv fetch failed", error=str(exc))
        return _FALLBACK_THEMES[0]

    if not papers:
        log.warning("theme_discoverer: no recent papers returned, using fallback")
        return _FALLBACK_THEMES[0]

    existing = list_signals()

    try:
        response = _get_llm(model).invoke([
            {"role": "system", "content": _load_system_prompt()},
            {"role": "user",   "content": _build_user_message(papers, existing)},
        ])
        themes = _parse_themes(response.content)
    except Exception as exc:
        log.warning("theme_discoverer: LLM call failed", error=str(exc))
        return _FALLBACK_THEMES[0]

    if not themes:
        log.warning("theme_discoverer: no themes parsed", raw=response.content[:400])
        return _FALLBACK_THEMES[0]

    log.info("theme_discoverer: selected theme", theme=themes[0], candidates=themes)
    return themes[0]


def _make_theme_discoverer_node(model: str | None = None):
    def theme_discoverer_node(state: AgentState) -> AgentState:
        return {**state, "theme": discover_theme(model=model)}

    return theme_discoverer_node
