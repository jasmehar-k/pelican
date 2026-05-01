"""Agent pipeline API router with SSE streaming."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from pelican.agents.graph import build_graph, coerce_state, initial_state
from pelican.api.models import AgentRunRequest, AgentRunSummary
from pelican.backtest.engine import BacktestConfig

router = APIRouter(prefix="/agents", tags=["agents"])

# run_id → asyncio.Queue (active or recently completed runs)
_ACTIVE_RUNS: dict[str, asyncio.Queue] = {}
# run_id → final state dict (populated once graph finishes)
_RUN_RESULTS: dict[str, dict[str, Any]] = {}


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_state(state: dict) -> dict:
    """Return a JSON-serialisable subset of a state dict."""
    safe_keys = {
        "decision", "feedback", "ic_tstat", "sharpe_net",
        "retry_count", "signal_hypothesis", "arxiv_ids", "theme",
        "errors", "memo",
    }
    out: dict[str, Any] = {}
    for k in safe_keys:
        v = state.get(k)
        if isinstance(v, (str, int, float, bool, list, type(None))):
            out[k] = v
    return out


def _run_graph_thread(
    run_id: str,
    req: AgentRunRequest,
    store: Any,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run the LangGraph graph synchronously in a thread executor."""

    def _put(event: dict | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def on_token(token: str) -> None:
        _put({"event": "llm_token", "node": "coder",
              "data": {"token": token}, "timestamp": _ts()})

    def on_attempt_start(n: int) -> None:
        _put({"event": "node_start", "node": "coder",
              "data": {"attempt": n}, "timestamp": _ts()})

    config = BacktestConfig(start=req.start, end=req.end)
    graph = build_graph(
        store, config,
        model=req.model,
        on_token=on_token,
        on_attempt_start=on_attempt_start,
        with_researcher=req.with_researcher,
    )
    state = coerce_state({"theme": req.theme})
    final_state: dict = dict(state)

    try:
        if req.with_researcher:
            _put({"event": "node_start", "node": "researcher",
                  "data": {}, "timestamp": _ts()})

        for node_event in graph.stream(state, stream_mode="updates"):
            for node_name, delta in node_event.items():
                final_state.update(delta)
                _put({"event": "node_complete", "node": node_name,
                      "data": _safe_state(final_state), "timestamp": _ts()})

        _RUN_RESULTS[run_id] = final_state
        try:
            store.log_run(final_state)
        except Exception:
            pass

        _put({"event": "run_complete", "node": None,
              "data": _safe_state(final_state), "timestamp": _ts()})

    except Exception as exc:
        _put({"event": "run_error", "node": None,
              "data": {"error": str(exc)}, "timestamp": _ts()})
    finally:
        _put(None)  # end-of-stream sentinel


async def _sse_generator(run_id: str, queue: asyncio.Queue):
    """Drain queue and yield SSE-formatted chunks.  Sends a heartbeat every 30s."""
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield ":heartbeat\n\n"
                continue

            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
    finally:
        _ACTIVE_RUNS.pop(run_id, None)


@router.post("/run")
async def start_run(req: AgentRunRequest, request: Request) -> dict:
    """Start a pipeline run. Returns run_id immediately; stream via /runs/{id}/stream."""
    run_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _ACTIVE_RUNS[run_id] = queue

    store = request.app.state.store
    loop = asyncio.get_running_loop()

    asyncio.get_event_loop().run_in_executor(
        None,
        _run_graph_thread,
        run_id, req, store, queue, loop,
    )

    return {"run_id": run_id}


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    """SSE endpoint — streams events for a running or recently started pipeline."""
    queue = _ACTIVE_RUNS.get(run_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="run not found or already complete")

    return StreamingResponse(
        _sse_generator(run_id, queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/runs")
async def list_runs(request: Request) -> list[AgentRunSummary]:
    """List completed runs from the research log (most recent first)."""
    store = request.app.state.store
    try:
        df = store.query(
            """
            SELECT run_id, theme, decision, ic_tstat, sharpe_net,
                   retry_count, CAST(ts AS VARCHAR) AS ts
            FROM research_log
            ORDER BY ts DESC
            LIMIT 50
            """
        )
        return [
            AgentRunSummary(
                run_id=row["run_id"] or "",
                theme=row["theme"] or "",
                decision=row["decision"],
                ic_tstat=row["ic_tstat"],
                sharpe_net=row["sharpe_net"],
                retry_count=int(row["retry_count"] or 0),
                ts=str(row["ts"] or ""),
            )
            for row in df.to_dicts()
        ]
    except Exception:
        return []


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    """Return cached final state for a completed run (in-memory only, lost on restart)."""
    result = _RUN_RESULTS.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _safe_state(result)


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    """Send end-of-stream sentinel to abort the SSE stream for a run."""
    queue = _ACTIVE_RUNS.get(run_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="run not found or already complete")
    await queue.put(None)
    _ACTIVE_RUNS.pop(run_id, None)
    return {"status": "cancelled"}
