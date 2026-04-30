"""
Coder agent node.

Calls the LLM (OpenRouter) with the factor description and coder system prompt,
extracts a ```python``` block, validates it in the sandbox, and retries up to
MAX_RETRIES times on failure.  Accumulated errors are fed back to the LLM so
each retry has more context than the last.

Rate-limit (429) handling: exponential backoff (10s, 20s, 40s) before retrying
so the free-tier rate window has time to reset.  Non-429 failures retry immediately.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from pelican.agents.state import AgentState
from pelican.agents.tools.code_exec import execute_signal_code
from pelican.utils.config import get_settings
from pelican.utils.logging import get_logger

MAX_RETRIES = 3
# Seconds to wait before each retry after a 429; index = attempt number (0-based).
_BACKOFF_SECONDS = (10, 20, 40)

log = get_logger(__name__)


def _get_llm(model: str | None = None):
    from langchain_openai import ChatOpenAI

    s = get_settings()
    return ChatOpenAI(
        model=model or s.openrouter_model,
        base_url=s.openrouter_base_url,
        api_key=s.openrouter_api_key,
        temperature=0.2,
        max_tokens=1024,
    )


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc)


def _load_system_prompt() -> str:
    path = Path(__file__).parent / "prompts" / "coder.md"
    return path.read_text()


def _extract_code(text: str) -> str | None:
    """Pull the first ```python ... ``` block from the LLM response."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _build_user_message(theme: str, errors: list[str]) -> str:
    parts = [f"Factor description: {theme}"]
    if errors:
        recent = errors[-3:]
        error_block = "\n".join(f"  - {e}" for e in recent)
        parts.append(
            f"\nPrevious attempt(s) failed with these errors:\n{error_block}"
            "\n\nPlease fix all listed issues in your next implementation."
        )
    return "\n".join(parts)


def _make_coder_node(model: str | None = None):
    """Return a LangGraph-compatible coder node using the given model.

    Args:
        model: OpenRouter model ID override.  Defaults to settings.openrouter_model.
    """
    def coder_node(state: AgentState) -> AgentState:
        llm = _get_llm(model)
        system_prompt = _load_system_prompt()
        errors: list[str] = list(state.get("errors") or [])

        for attempt in range(1, MAX_RETRIES + 1):
            log.info("coder attempt", attempt=attempt, theme=state["theme"])
            user_msg = _build_user_message(state["theme"], errors)

            try:
                response = llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ])
                raw = response.content
            except Exception as exc:
                if _is_rate_limit(exc) and attempt < MAX_RETRIES:
                    wait = _BACKOFF_SECONDS[attempt - 1]
                    log.warning(
                        "rate limited, backing off",
                        attempt=attempt,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                err = f"attempt {attempt}: {'rate limited (429)' if _is_rate_limit(exc) else 'LLM call failed'}: {exc}"
                log.warning("llm call failed", attempt=attempt, error=str(exc))
                errors.append(err)
                continue

            code = _extract_code(raw)
            if code is None:
                err = f"attempt {attempt}: response contained no ```python``` block"
                log.warning("no code block", attempt=attempt)
                errors.append(err)
                continue

            success, error_msg, _ = execute_signal_code(code)
            if success:
                log.info("coder succeeded", attempt=attempt)
                return {**state, "generated_code": code, "errors": errors}

            err = f"attempt {attempt}: {error_msg}"
            log.warning("sandbox rejected code", attempt=attempt, error=error_msg)
            errors.append(err)

        log.warning("coder exhausted retries", retries=MAX_RETRIES, theme=state["theme"])
        return {**state, "generated_code": None, "errors": errors}

    return coder_node


# Default instance — used directly in graph.py and tests.
coder_node = _make_coder_node()
