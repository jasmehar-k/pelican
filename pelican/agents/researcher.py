"""Researcher agent node for literature-grounded factor discovery."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from pelican.agents.state import AgentState
from pelican.agents.tools.search import SearchResult, search_arxiv
from pelican.agents.tools.vector_store import find_similar, has_paper, retrieve_for_theme, store_paper
from pelican.utils.config import get_settings
from pelican.utils.logging import get_logger

log = get_logger(__name__)


def _get_llm(
    model: str | None = None,
    streaming: bool = False,
    callbacks: list | None = None,
):
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=0.2,
        max_tokens=1024,
        streaming=streaming,
        callbacks=callbacks or [],
    )


def _load_system_prompt() -> str:
    path = Path(__file__).parent / "prompts" / "researcher.md"
    return path.read_text()


def _format_papers(papers: list[SearchResult]) -> str:
    lines: list[str] = []
    for index, paper in enumerate(papers, start=1):
        authors = ", ".join(paper["authors"]) if paper["authors"] else "Unknown"
        lines.append(
            f"{index}. {paper['title']}\n"
            f"   arXiv: {paper['arxiv_id']}\n"
            f"   Authors: {authors}\n"
            f"   Abstract: {paper['abstract']}"
        )
    return "\n\n".join(lines)


def _parse_flag(text: str, key: str) -> str | None:
    match = re.search(rf"^{key}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _build_multi_user_message(
    theme: str,
    papers: list[SearchResult],
    n: int,
    existing_signals: list[str] | None = None,
) -> str:
    numbered = "\n\n".join(
        f"HYPOTHESIS_{i}: <2-3 sentences: economic rationale + which columns to use>\n"
        f"DATA_FIELDS_{i}: comma-separated exact column names from the available list\n"
        f"SIGNAL_NAME_{i}: short_snake_case"
        for i in range(1, n + 1)
    )
    parts = [
        f"Research theme: {theme}",
        "",
        "Paper summaries:",
        _format_papers(papers) if papers else "No papers were found.",
        "",
    ]
    if existing_signals:
        parts += [
            "Signals already in the registry — DO NOT reproduce these "
            "(different formula, different columns, or different mechanism required):",
            ", ".join(existing_signals),
            "",
        ]
    parts += [
        f"Generate exactly {n} DISTINCT signal hypotheses grounded in these papers.",
        "Each must use a different economic mechanism or different data fields.",
        "Return this structure — one block per signal:",
        "",
        numbered,
    ]
    return "\n".join(parts)


def _parse_multi_response(text: str, n: int) -> list[dict]:
    results = []
    for i in range(1, n + 1):
        h = _parse_flag(text, f"HYPOTHESIS_{i}")
        df = _parse_flag(text, f"DATA_FIELDS_{i}") or ""
        sn = _parse_flag(text, f"SIGNAL_NAME_{i}")
        # Reject unfilled template markers, fragments, and stub signal names
        is_template = h and ("<" in h or h.startswith("..."))
        is_fragment = h and (
            len(h.split()) < 10                          # fewer than 10 words
            or not any(c in h for c in ".!?")           # no sentence-ending punctuation
        )
        is_stub_name = sn in (None, "short_snake_case", f"signal_{i}_name")
        if h and not is_template and not is_fragment:
            results.append({
                "hypothesis": h,
                "data_fields": [f.strip() for f in df.split(",") if f.strip()],
                "signal_name": (f"signal_{i}" if is_stub_name else sn),
            })
    return results


def get_hypotheses(
    theme: str,
    n: int = 3,
    model: str | None = None,
    existing_signals: list[str] | None = None,
) -> tuple[list[SearchResult], list[dict]]:
    """Search arXiv once and return up to *n* distinct signal hypotheses.

    Returns (papers, hypotheses).  Each hypothesis is a dict with keys:
        hypothesis   — 2-3 sentence economic rationale
        data_fields  — list of column names to use
        signal_name  — short snake_case identifier

    Falls back to an empty hypotheses list if no papers are found.
    """
    papers = search_arxiv(theme, max_results=10)
    fresh = [p for p in papers if not find_similar(p["abstract"], threshold=0.92)]
    context = (fresh or papers)[:5]

    # When arXiv returns fewer than 3 usable papers, supplement with previously
    # stored papers retrieved from the vector store.
    if len(context) < 3:
        stored = retrieve_for_theme(theme, n_results=5)
        stored_ids = {p["arxiv_id"] for p in context}
        for match in stored:
            if match["arxiv_id"] not in stored_ids and len(context) < 5:
                m = match["metadata"]
                context.append({
                    "title": m.get("title", ""),
                    "authors": [a.strip() for a in m.get("authors", "").split(",") if a.strip()],
                    "abstract": match["abstract"],
                    "arxiv_id": match["arxiv_id"],
                    "url": m.get("url", f"https://arxiv.org/abs/{match['arxiv_id']}"),
                })
                stored_ids.add(match["arxiv_id"])
        if len(context) >= 3:
            log.info("researcher: supplemented with stored papers", total=len(context), theme=theme)

    if not context:
        log.warning("researcher: no papers found", theme=theme)
        return papers, []

    llm = _get_llm(model)
    system_prompt = _load_system_prompt()
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_multi_user_message(theme, context, n, existing_signals)},
    ])
    hypotheses = _parse_multi_response(response.content, n)
    if not hypotheses:
        log.warning(
            "researcher: no hypotheses parsed — raw LLM response below",
            theme=theme,
            response=response.content[:800],
        )

    for paper in context:
        if not has_paper(paper["arxiv_id"]):
            store_paper(
                paper["arxiv_id"],
                paper["title"],
                paper["abstract"],
                {"url": paper["url"], "authors": paper["authors"]},
            )

    log.info("researcher: hypotheses extracted", n=len(hypotheses), theme=theme)
    return papers, hypotheses



def _make_researcher_node(model: str | None = None):
    def researcher_node(state: AgentState) -> AgentState:
        theme = state["theme"]
        run_id = state.get("run_id") or str(uuid.uuid4())

        from pelican.backtest.signals import list_signals
        existing = list_signals()

        try:
            papers, hypotheses = get_hypotheses(theme, n=3, model=model, existing_signals=existing)
        except Exception as exc:
            log.warning("researcher: get_hypotheses failed", error=str(exc), theme=theme)
            papers, hypotheses = [], []

        arxiv_ids: list[str] = [p["arxiv_id"] for p in papers[:5]]
        # Hypothesis 0 is the default for the first coder attempt; later retries
        # will pick subsequent hypotheses via retry_count (see coder node).
        signal_hypothesis: str | None = hypotheses[0]["hypothesis"] if hypotheses else None
        signal_name: str | None = hypotheses[0]["signal_name"] if hypotheses else None

        return {
            **state,
            "papers": papers,
            "hypotheses": hypotheses,
            "signal_hypothesis": signal_hypothesis,
            "arxiv_ids": arxiv_ids,
            "run_id": run_id,
            "signal_name": signal_name,
        }

    return researcher_node
