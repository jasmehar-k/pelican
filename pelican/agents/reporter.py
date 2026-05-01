"""Reporter agent node — generates a structured memo for accepted signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pelican.agents.state import AgentState
from pelican.utils.config import get_settings
from pelican.utils.logging import get_logger

log = get_logger(__name__)


def _get_llm(model: str | None = None):
    from langchain_openai import ChatOpenAI

    s = get_settings()
    return ChatOpenAI(
        model=model or s.openrouter_model,
        base_url=s.openrouter_base_url,
        api_key=s.openrouter_api_key,
        temperature=0.3,
        max_tokens=512,
    )


def _load_system_prompt() -> str:
    return (Path(__file__).parent / "prompts" / "reporter.md").read_text()


def _extract_memo(raw: str) -> str:
    """Extract the final memo paragraph from LLM output.

    Some models prefix the memo with chain-of-thought (word counts, drafts,
    reasoning).  We look for the first paragraph that reads like a memo:
    starts with 'The signal' or is a long prose sentence (>= 60 chars, no
    colon at the start).  Fallback: return the longest paragraph.
    """
    paragraphs = [p.strip() for p in raw.strip().split("\n\n") if p.strip()]
    for para in paragraphs:
        first_line = para.splitlines()[0]
        if first_line.startswith(("The signal", "This signal", "The factor", "This factor")):
            return para
    # Fallback: return the longest paragraph (likely the actual memo)
    return max(paragraphs, key=len, default=raw.strip())


def _build_memo_prompt(state: AgentState) -> str:
    arxiv_refs = ", ".join(state.get("arxiv_ids") or []) or "none"
    return "\n".join([
        f"Signal hypothesis: {state.get('signal_hypothesis') or state['theme']}",
        "",
        "Signal code:",
        "```python",
        state.get("generated_code") or "",
        "```",
        "",
        f"Backtest results: IC t-stat={state.get('ic_tstat', 0):.3f}, "
        f"net Sharpe={state.get('sharpe_net', 0):.3f}",
        f"arXiv citations: {arxiv_refs}",
        "",
        "Write the investment memo.",
    ])


def _make_reporter_node(store: Any, model: str | None = None):
    def reporter_node(state: AgentState) -> AgentState:
        decision = state.get("decision")
        memo: str | None = None

        if decision == "accept":
            try:
                llm = _get_llm(model)
                system_prompt = _load_system_prompt()
                response = llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _build_memo_prompt(state)},
                ])
                memo = _extract_memo(response.content)
                log.info("reporter: memo generated", theme=state["theme"])
            except Exception as exc:
                log.warning("reporter: memo generation failed", error=str(exc))
                memo = (
                    f"Signal accepted. IC t-stat={state.get('ic_tstat', 0):.3f}, "
                    f"net Sharpe={state.get('sharpe_net', 0):.3f}."
                )

        try:
            store.log_memo({**state, "memo": memo})
        except Exception as exc:
            log.warning("reporter: failed to persist memo", error=str(exc))

        log.info(
            "reporter: run complete",
            decision=decision,
            retry_count=state.get("retry_count", 0),
            theme=state["theme"],
        )
        return {**state, "memo": memo}

    return reporter_node
