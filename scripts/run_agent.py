"""
Live streaming demo for the Coder → Critic agent pipeline.

LLM tokens appear character-by-character inside a Rich panel as the model writes.
After the code is validated, a spinner shows while the backtest engine runs.
The final result appears as a styled decision panel.

Usage:
    python scripts/run_agent.py --theme "12-month momentum, skip last month"
    python scripts/run_agent.py --theme "low debt-to-equity" --model deepseek/deepseek-chat:free
    python scripts/run_agent.py --theme "low volatility" --start 2025-03-01 --end 2025-11-01
"""

from __future__ import annotations

import argparse
import sys
import threading
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pelican.agents.graph import build_graph, initial_state
from pelican.backtest.engine import BacktestConfig
from pelican.data.store import DataStore
from pelican.utils.config import get_settings
from pelican.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="Live streaming agent demo")
    p.add_argument("--theme", required=True, help="Natural language factor description")
    p.add_argument("--start", default="2025-03-01",
                   help="Backtest start (fundamental signals need ≥ 2025-03-01)")
    p.add_argument("--end",   default="2025-11-01", help="Backtest end")
    p.add_argument("--db-path", default=str(s.duckdb_path))
    p.add_argument("--model", default=None,
                   help="OpenRouter model ID, e.g. deepseek/deepseek-chat:free")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable token streaming (plain output)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Rich streaming display
# ---------------------------------------------------------------------------

def _run_streaming(args, store, config) -> dict:
    from rich.columns import Columns
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # Shared mutable state accessed from callbacks + main thread.
    _buf: list[str] = []
    _attempt: list[int] = [0]
    _phase: list[str] = ["generating"]  # generating | validating | backtesting | done
    _lock = threading.Lock()

    def on_attempt_start(attempt_num: int) -> None:
        with _lock:
            _buf.clear()
            _attempt[0] = attempt_num
            _phase[0] = "generating"

    def on_token(token: str) -> None:
        with _lock:
            _buf.append(token)

    # Custom renderable — Rich calls __rich_console__ on every refresh tick.
    class LiveDisplay:
        def __rich__(self):
            phase = _phase[0]
            attempt = _attempt[0]
            with _lock:
                code = "".join(_buf)

            if phase == "generating":
                title = (
                    f"[bold cyan]Generating signal code[/] "
                    f"[dim](attempt {attempt}/{3})[/]"
                )
                body = Syntax(code or " ", "python", theme="monokai",
                              line_numbers=False, word_wrap=True)
                return Panel(body, title=title, border_style="cyan", padding=(0, 1))

            if phase == "validating":
                return Panel(
                    Syntax(code, "python", theme="monokai",
                           line_numbers=False, word_wrap=True),
                    title="[bold yellow]Validating in sandbox…[/]",
                    border_style="yellow",
                    padding=(0, 1),
                )

            if phase == "backtesting":
                spinner = Spinner("dots", text=Text(" Running backtest engine…", style="bold"))
                return Panel(spinner, title="[bold blue]Critic: backtesting[/]",
                             border_style="blue", padding=(0, 1))

            return Panel("[dim]Done[/]", border_style="dim")

    graph = build_graph(
        store, config,
        model=args.model,
        on_token=on_token,
        on_attempt_start=on_attempt_start,
    )
    state = initial_state(args.theme)

    result: dict = {}

    with Live(LiveDisplay(), console=console, refresh_per_second=15,
              vertical_overflow="visible") as live:

        def _invoke():
            nonlocal result
            r = graph.invoke(state)
            result.update(r)
            with _lock:
                _phase[0] = "done"

        # Monkey-patch critic's backtest call to flip the phase indicator.
        import pelican.agents.critic as _critic_mod
        _orig_run_bt = _critic_mod.run_backtest_with_fn

        def _hooked_run_bt(fn, spec, cfg, st):
            with _lock:
                _phase[0] = "backtesting"
            return _orig_run_bt(fn, spec, cfg, st)

        _critic_mod.run_backtest_with_fn = _hooked_run_bt

        # Phase: code validated — flip before graph returns from coder node.
        # We detect this by watching _phase after the graph finishes the coder step;
        # the simplest hook is on_attempt_start not firing anymore.
        # Instead we set "validating" after on_token stops and before backtest starts.
        # We do this by watching the thread: start graph.invoke in a worker.
        worker = threading.Thread(target=_invoke, daemon=True)
        worker.start()

        # While worker runs, show live display and update phase hints.
        prev_len = 0
        ticks_since_last_token = 0
        while worker.is_alive():
            with _lock:
                cur_len = len(_buf)
                phase = _phase[0]
            if phase == "generating":
                if cur_len > prev_len:
                    ticks_since_last_token = 0
                    prev_len = cur_len
                elif cur_len > 0:
                    ticks_since_last_token += 1
                    # If tokens stopped arriving but backtest hasn't started,
                    # we're probably in sandbox validation.
                    if ticks_since_last_token > 5:
                        with _lock:
                            _phase[0] = "validating"
            import time
            time.sleep(0.07)

        worker.join()
        _critic_mod.run_backtest_with_fn = _orig_run_bt

    # --- Final result panel ---
    accepted = result.get("decision") == "accept"
    colour = "bold green" if accepted else "bold red"
    label = "ACCEPTED ✓" if accepted else "REJECTED ✗"

    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold", justify="right")
    t.add_column()
    t.add_row("Decision", f"[{colour}]{label}[/]")
    t.add_row("Feedback", result.get("feedback") or "—")
    if result.get("ic_tstat") is not None:
        t.add_row("IC t-stat", f"{result['ic_tstat']:.3f}")
    if result.get("sharpe_net") is not None:
        t.add_row("Net Sharpe", f"{result['sharpe_net']:.3f}")

    console.print()
    console.print(Panel(t, title=f"[{colour}]Critic decision[/]",
                        border_style="green" if accepted else "red"))

    if result.get("errors"):
        console.print("\n[dim]Retry log:[/]")
        for e in result["errors"]:
            console.print(f"  [dim]{e}[/]")

    return result


# ---------------------------------------------------------------------------
# Plain (no-stream) fallback
# ---------------------------------------------------------------------------

def _run_plain(args, store, config) -> dict:
    configure_logging(dev=True)
    graph = build_graph(store, config, model=args.model)
    state = initial_state(args.theme)

    print(f"\nTheme: {args.theme}")
    print(f"Backtest: {args.start} → {args.end}\n")

    result = graph.invoke(state)

    print(f"\nDecision  : {result['decision']}")
    print(f"Feedback  : {result['feedback']}")
    if result.get("ic_tstat") is not None:
        print(f"IC t-stat : {result['ic_tstat']:.2f}")
    if result.get("sharpe_net") is not None:
        print(f"Net Sharpe: {result['sharpe_net']:.2f}")
    if result.get("errors"):
        print("\nRetry errors:")
        for e in result["errors"]:
            print(f"  {e}")
    if result["generated_code"]:
        print("\n--- Generated code ---")
        print(result["generated_code"])
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        sys.exit(1)

    store = DataStore(db_path)
    config = BacktestConfig(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )

    if args.no_stream:
        result = _run_plain(args, store, config)
    else:
        result = _run_streaming(args, store, config)

    store.close()
    sys.exit(0 if result.get("decision") == "accept" else 1)


if __name__ == "__main__":
    main()
