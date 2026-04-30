"""Researcher agent node for literature-grounded factor discovery."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from pelican.agents.state import AgentState
from pelican.agents.tools.pdf_extract import fetch_pdf_text
from pelican.agents.tools.search import SearchResult, search_arxiv
from pelican.agents.tools.vector_store import find_similar, has_paper, store_paper
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
        papers = search_arxiv(theme, max_results=10)

        fresh = [paper for paper in papers if not find_similar(paper["abstract"], threshold=0.92)]
        context = (fresh or papers)[:5]

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
