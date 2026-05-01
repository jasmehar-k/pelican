"""
Live streaming demo for the agent pipeline.

Single-signal mode (default):
    python scripts/run_agent.py --theme "12-month momentum, skip last month"

Multi-signal autonomous demo:
    python scripts/run_agent.py --theme "earnings quality factors" --signals 3

The researcher searches arXiv, extracts N distinct hypotheses, then streams each
one through Coder → Critic and shows a final comparison table.

Other options:
    --model deepseek/deepseek-chat:free
    --start 2025-03-01 --end 2025-11-01
    --no-research   (skip arXiv, code directly from theme)
    --no-stream     (plain text output)
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
    p = argparse.ArgumentParser(description="Pelican agent demo")
    p.add_argument("--theme", required=True, help="Natural language factor description")
    p.add_argument("--start", default="2025-03-01",
                   help="Backtest start (fundamental signals need ≥ 2025-03-01)")
    p.add_argument("--end",   default="2025-11-01", help="Backtest end")
    p.add_argument("--db-path", default=str(s.duckdb_path))
    p.add_argument("--model", default=None,
                   help="OpenRouter model ID, e.g. deepseek/deepseek-chat:free")
    p.add_argument("--signals", type=int, default=1, choices=[1, 2, 3],
                   help="Number of signals to generate autonomously from one research run (1-3)")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable token streaming (plain output)")
    p.add_argument("--no-research", action="store_true",
                   help="Skip the researcher node and run Coder → Critic directly")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Single-signal streaming display
# ---------------------------------------------------------------------------

def _run_one_signal_streaming(
    hypothesis: str | None,
    signal_label: str,
    args,
    store,
    config,
    console,
    arxiv_ids: list[str] | None = None,
) -> dict:
    """Stream one coder→critic run.  Returns the final state dict."""
    import time

    from rich.live import Live
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    _buf: list[str] = []
    _attempt: list[int] = [0]
    _phase: list[str] = ["generating"]
    _lock = threading.Lock()

    def on_attempt_start(attempt_num: int) -> None:
        with _lock:
            _buf.clear()
            _attempt[0] = attempt_num
            _phase[0] = "generating"

    def on_token(token: str) -> None:
        with _lock:
            _buf.append(token)

    class LiveDisplay:
        def __rich__(self):
            phase = _phase[0]
            attempt = _attempt[0]
            with _lock:
                code = "".join(_buf)

            if phase == "generating":
                title = (
                    f"[bold cyan]Generating: {signal_label}[/] "
                    f"[dim](attempt {attempt}/3)[/]"
                )
                body = Syntax(code or " ", "python", theme="monokai",
                              line_numbers=False, word_wrap=True)
                return Panel(body, title=title, border_style="cyan", padding=(0, 1))

            if phase == "validating":
                return Panel(
                    Syntax(code, "python", theme="monokai",
                           line_numbers=False, word_wrap=True),
                    title="[bold yellow]Validating in sandbox…[/]",
                    border_style="yellow", padding=(0, 1),
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
        with_researcher=False,
    )
    state = initial_state(args.theme)
    state["signal_hypothesis"] = hypothesis
    if arxiv_ids:
        state["arxiv_ids"] = arxiv_ids

    result: dict = {}

    with Live(LiveDisplay(), console=console, refresh_per_second=15,
              vertical_overflow="visible"):

        def _invoke():
            nonlocal result
            r = graph.invoke(state)
            result.update(r)
            with _lock:
                _phase[0] = "done"

        import pelican.agents.critic as _critic_mod
        _orig_run_bt = _critic_mod.run_backtest_with_fn

        def _hooked_run_bt(fn, spec, cfg, st):
            with _lock:
                _phase[0] = "backtesting"
            return _orig_run_bt(fn, spec, cfg, st)

        _critic_mod.run_backtest_with_fn = _hooked_run_bt

        worker = threading.Thread(target=_invoke, daemon=True)
        worker.start()

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
                    if ticks_since_last_token > 5:
                        with _lock:
                            _phase[0] = "validating"
            time.sleep(0.07)

        worker.join()
        _critic_mod.run_backtest_with_fn = _orig_run_bt

    # Per-signal result panel
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
    console.print(Panel(t, title=f"[{colour}]{signal_label}[/]",
                        border_style="green" if accepted else "red"))

    if result.get("errors"):
        console.print("\n[dim]Retry log:[/]")
        for e in result["errors"]:
            console.print(f"  [dim]{e}[/]")

    return result


# ---------------------------------------------------------------------------
# Multi-signal autonomous demo
# ---------------------------------------------------------------------------

def _run_multi_signal(args, store, config) -> list[dict]:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    console = Console()

    # ---- Research phase ----
    from pelican.agents.researcher import get_hypotheses

    n = args.signals
    console.print()
    with console.status("[bold magenta]Searching arXiv for relevant papers…[/]"):
        papers, hypotheses = get_hypotheses(args.theme, n=n, model=args.model)

    if papers:
        pt = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
        pt.add_column("arXiv ID", style="dim", width=15)
        pt.add_column("Title")
        for p in papers[:5]:
            pt.add_row(p["arxiv_id"], p["title"])
        console.print(Panel(pt, title=f"[bold magenta]Papers found ({len(papers)})[/]",
                            border_style="magenta"))
    else:
        console.print("[dim]No papers found — proceeding with theme only[/]")

    if not hypotheses:
        hypotheses = [{"hypothesis": None, "signal_name": "signal_1", "data_fields": []}]

    arxiv_ids = [p["arxiv_id"] for p in papers[:5]]

    # ---- Per-signal loop ----
    results: list[dict] = []
    for i, hyp in enumerate(hypotheses, start=1):
        name = hyp.get("signal_name") or f"signal_{i}"
        hypothesis_text = hyp.get("hypothesis")

        console.print()
        console.print(Rule(f"[bold]Signal {i}/{len(hypotheses)} — {name}[/]"))
        if hypothesis_text:
            console.print(f"[italic dim]{hypothesis_text}[/]\n")

        result = _run_one_signal_streaming(
            hypothesis_text, name, args, store, config, console, arxiv_ids=arxiv_ids
        )
        result["_signal_name"] = name
        result["_hypothesis"] = hypothesis_text
        results.append(result)

    # ---- Summary table ----
    console.print()
    summary = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    summary.add_column("Signal", style="bold")
    summary.add_column("Decision", justify="center")
    summary.add_column("IC t-stat", justify="right")
    summary.add_column("Net Sharpe", justify="right")
    summary.add_column("Hypothesis", max_width=55)

    for r in results:
        accepted = r.get("decision") == "accept"
        dec = "[bold green]ACCEPT ✓[/]" if accepted else "[bold red]REJECT ✗[/]"
        ic = f"{r['ic_tstat']:.3f}" if r.get("ic_tstat") is not None else "—"
        sh = f"{r['sharpe_net']:.3f}" if r.get("sharpe_net") is not None else "—"
        hyp = (r.get("_hypothesis") or "—")[:80]
        summary.add_row(r.get("_signal_name", "?"), dec, ic, sh, hyp)

    console.print(Panel(summary, title="[bold]Research run summary[/]", border_style="dim"))

    return results


# ---------------------------------------------------------------------------
# Single-signal full pipeline (with optional researcher node)
# ---------------------------------------------------------------------------

def _run_streaming(args, store, config, with_researcher: bool = True) -> dict:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    console = Console()

    _buf: list[str] = []
    _attempt: list[int] = [0]
    _phase: list[str] = ["searching" if with_researcher else "generating"]
    _lock = threading.Lock()

    def on_attempt_start(attempt_num: int) -> None:
        with _lock:
            _buf.clear()
            _attempt[0] = attempt_num
            _phase[0] = "generating"

    def on_token(token: str) -> None:
        with _lock:
            _buf.append(token)

    class LiveDisplay:
        def __rich__(self):
            phase = _phase[0]
            attempt = _attempt[0]
            with _lock:
                code = "".join(_buf)

            if phase == "generating":
                title = (
                    f"[bold cyan]Generating signal code[/] "
                    f"[dim](attempt {attempt}/3)[/]"
                )
                body = Syntax(code or " ", "python", theme="monokai",
                              line_numbers=False, word_wrap=True)
                return Panel(body, title=title, border_style="cyan", padding=(0, 1))

            if phase == "searching":
                spinner = Spinner("dots", text=Text(" Searching arXiv…", style="bold"))
                return Panel(spinner, title="[bold magenta]Researcher: searching[/]",
                             border_style="magenta", padding=(0, 1))

            if phase == "validating":
                return Panel(
                    Syntax(code, "python", theme="monokai",
                           line_numbers=False, word_wrap=True),
                    title="[bold yellow]Validating in sandbox…[/]",
                    border_style="yellow", padding=(0, 1),
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
        with_researcher=with_researcher,
    )
    state = initial_state(args.theme)
    result: dict = {}

    with Live(LiveDisplay(), console=console, refresh_per_second=15,
              vertical_overflow="visible"):

        def _invoke():
            nonlocal result
            r = graph.invoke(state)
            result.update(r)
            with _lock:
                _phase[0] = "done"

        import pelican.agents.critic as _critic_mod
        _orig_run_bt = _critic_mod.run_backtest_with_fn

        def _hooked_run_bt(fn, spec, cfg, st):
            with _lock:
                _phase[0] = "backtesting"
            return _orig_run_bt(fn, spec, cfg, st)

        _critic_mod.run_backtest_with_fn = _hooked_run_bt

        worker = threading.Thread(target=_invoke, daemon=True)
        worker.start()

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
                    if ticks_since_last_token > 5:
                        with _lock:
                            _phase[0] = "validating"
            import time
            time.sleep(0.07)

        worker.join()
        _critic_mod.run_backtest_with_fn = _orig_run_bt

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

def _run_plain(args, store, config, with_researcher: bool = True) -> dict:
    configure_logging(dev=True)
    graph = build_graph(store, config, model=args.model, with_researcher=with_researcher)
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
    store.init_schema()
    config = BacktestConfig(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )

    # Multi-signal autonomous demo
    if args.signals > 1:
        results = _run_multi_signal(args, store, config)
        for r in results:
            store.log_run(r)
        store.close()
        sys.exit(0 if any(r.get("decision") == "accept" for r in results) else 1)

    # Single-signal run
    if args.no_stream:
        result = _run_plain(args, store, config, with_researcher=not args.no_research)
    else:
        result = _run_streaming(args, store, config, with_researcher=not args.no_research)

    store.log_run(result)
    store.close()
    sys.exit(0 if result.get("decision") == "accept" else 1)


if __name__ == "__main__":
    main()
