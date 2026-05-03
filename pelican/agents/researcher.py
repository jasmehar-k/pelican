"""Researcher agent node for literature-grounded factor discovery."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from pelican.agents.state import AgentState
from pelican.agents.tools.pdf_extract import fetch_pdf_text
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


def _build_multi_user_message(theme: str, papers: list[SearchResult], n: int) -> str:
    numbered = "\n\n".join(
        f"HYPOTHESIS_{i}: <2-3 sentences: economic rationale + which columns to use>\n"
        f"DATA_FIELDS_{i}: comma-separated exact column names from the available list\n"
        f"SIGNAL_NAME_{i}: short_snake_case"
        for i in range(1, n + 1)
    )
    return "\n".join([
        f"Research theme: {theme}",
        "",
        "Paper summaries:",
        _format_papers(papers) if papers else "No papers were found.",
        "",
        f"Generate exactly {n} DISTINCT signal hypotheses grounded in these papers.",
        "Each must use a different economic mechanism or different data fields.",
        "Return this structure — one block per signal:",
        "",
        numbered,
    ])


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
        {"role": "user", "content": _build_multi_user_message(theme, context, n)},
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


def _parse_bool(text: str, key: str) -> bool:
    value = _parse_flag(text, key)
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "y"}


def _parse_response(text: str) -> dict[str, object]:
    hypothesis = _parse_flag(text, "HYPOTHESIS")
    data_fields = _parse_flag(text, "DATA_FIELDS") or ""
    signal_name = _parse_flag(text, "SIGNAL_NAME")
    return {
        "hypothesis": hypothesis,
        "data_fields": [field.strip() for field in data_fields.split(",") if field.strip()],
        "signal_name": signal_name,
        "need_more_detail": _parse_bool(text, "NEED_MORE_DETAIL"),
    }


def _build_user_message(theme: str, papers: list[SearchResult], pdf_text: str | None = None) -> str:
    parts = [
        f"Research theme: {theme}",
        "",
        "Paper summaries:",
        _format_papers(papers) if papers else "No papers were found.",
        "",
        "Return exactly these fields:",
        "HYPOTHESIS: 2-3 sentences with the economic rationale and relevant data fields",
        "DATA_FIELDS: comma-separated exact column names",
        "SIGNAL_NAME: short snake_case name",
        "NEED_MORE_DETAIL: true or false",
    ]
    if pdf_text:
        parts.extend([
            "",
            "Additional PDF text for the top paper:",
            pdf_text,
        ])
    return "\n".join(parts)


def _make_researcher_node(model: str | None = None):
    def researcher_node(state: AgentState) -> AgentState:
        theme = state["theme"]
        run_id = state.get("run_id") or str(uuid.uuid4())
        try:
            papers = search_arxiv(theme, max_results=10)
        except Exception as exc:
            log.warning("researcher: arXiv search failed, proceeding without papers", error=str(exc))
            papers = []

        fresh = [paper for paper in papers if not find_similar(paper["abstract"], threshold=0.92)]
        context = (fresh or papers)[:5]

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

        signal_hypothesis: str | None = None
        arxiv_ids: list[str] = [paper["arxiv_id"] for paper in context]

        if not context:
            return {
                **state,
                "papers": papers,
                "signal_hypothesis": None,
                "arxiv_ids": [],
                "run_id": run_id,
            }

        llm = _get_llm(model=model)
        system_prompt = _load_system_prompt()

        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_message(theme, context)},
        ])
        parsed = _parse_response(response.content)

        if parsed["need_more_detail"] and context and len(context[0]["abstract"]) < 300:
            pdf_text = fetch_pdf_text(context[0]["arxiv_id"], max_pages=5)
            if pdf_text:
                response = llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _build_user_message(theme, context, pdf_text)},
                ])
                parsed = _parse_response(response.content)

        signal_hypothesis = parsed["hypothesis"] if parsed["hypothesis"] else None

        for paper in context:
            if not has_paper(paper["arxiv_id"]):
                store_paper(
                    paper["arxiv_id"],
                    paper["title"],
                    paper["abstract"],
                    {"url": paper["url"], "authors": paper["authors"]},
                )

        log.info("research completed", theme=theme, papers=len(papers), context=len(context))
        return {
            **state,
            "papers": papers,
            "signal_hypothesis": signal_hypothesis,
            "arxiv_ids": arxiv_ids,
            "run_id": run_id,
        }

    return researcher_node
