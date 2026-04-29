# Pelican

An agentic factor research platform. LLM agents autonomously discover, implement,
and backtest quantitative alpha signals from academic literature and alternative data.
Surviving signals feed a risk-aware portfolio optimizer. A FastAPI + React dashboard
lets you browse signals, inspect backtest results, and monitor the agent pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Pipeline                           │
│                                                                 │
│  Research Theme ──► Researcher ──► Coder ──► Critic ──► Signal  │
│                         ▲                       │    Registry   │
│                         └─────── reject ────────┘               │
│                                                                 │
│                    (LangGraph state machine)                    │
└─────────────────────────────────────────────────────────────────┘
          │ accepted signal
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backtest Engine                              │
│  Polars panel DataFrame ──► cross-sectional rank ──► quintile   │
│  portfolios ──► forward returns ──► IC / Sharpe / drawdown      │
│                    (point-in-time correct, vectorized)          │
└─────────────────────────────────────────────────────────────────┘
          │ signal scores + metrics
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Portfolio Construction                          │
│  Signal combiner (IC-weighted z-scores) ──► CVXPy optimizer     │
│  ──► weights (long-only or L/S, sector-neutral, risk-capped)    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI Backend + React Frontend                   │
│  /signals   /backtest   /portfolio   /agents                    │
│  Signal browser · Equity curves · Agent activity log            │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. `scripts/seed_data.py` downloads S&P 500 prices (yfinance) and SEC filings
   (sec-edgar-downloader) into a local DuckDB database.
2. The agent pipeline reads from DuckDB, generates signal code, and runs backtests.
3. Accepted signals are persisted back to DuckDB.
4. The FastAPI server reads from DuckDB and exposes JSON APIs consumed by React.

---

## Agent Pipeline Loop

```
1. RESEARCHER  — given a theme, searches academic papers and web, proposes
                 a SignalSpec (hypothesis + required data fields + citations)

2. CODER       — translates the SignalSpec into a Python/Polars compute_signal()
                 function; sandboxed execution validates it runs without error

3. CRITIC      — reviews code for look-ahead bias, runs the backtest, checks
                 IC t-stat ≥ 1.5 and Sharpe ≥ 0.3; returns accept or reject
                 with written feedback

4. LOOP        — on reject, feedback is sent back to Researcher (up to N retries)
                 on accept, signal is written to the registry
```

Orchestrated by LangGraph. Each node is a pure function `(State) -> State`.
The graph is compiled once at startup and reused across runs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Agent orchestration | LangGraph + LangChain |
| LLM | OpenRouter — `meta-llama/llama-3.3-70b-instruct:free` |
| Dataframe engine | Polars |
| Storage | DuckDB |
| Price data | yfinance |
| Filing data | sec-edgar-downloader |
| Portfolio optimization | CVXPy (CLARABEL/OSQP solver) |
| Hyperparameter tuning | Optuna |
| API server | FastAPI + Uvicorn |
| Frontend | React 18, Vite, TypeScript, Recharts |
| Testing | pytest, pytest-asyncio |

---

## Project Structure

```
pelican/
├── pelican/
│   ├── data/           # ingestion: prices, filings, news, DuckDB store
│   ├── agents/         # LangGraph graph, researcher/coder/critic nodes, tools, prompts
│   ├── backtest/       # vectorized engine, signal registry, metrics, universe
│   ├── portfolio/      # CVXPy optimizer, risk model, signal combiner
│   ├── api/            # FastAPI app, routers, Pydantic models, deps
│   ├── utils/          # config (pydantic-settings), structured logging
│   └── cli.py          # `pelican` CLI entry point (Typer)
├── frontend/           # React + Vite dashboard
├── scripts/            # seed_data.py, run_agent.py
├── tests/              # pytest test suite
├── notebooks/          # research notebooks (gitignored outputs)
├── data/               # local DuckDB + file cache (gitignored)
├── pyproject.toml
└── .env.example
```

---

## Quickstart

```bash
# 1. Install Python deps
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# edit .env: add OPENROUTER_API_KEY, etc.

# 3. Seed data (~30 min, ~10 GB)
python scripts/seed_data.py

# 4. Start API server
pelican serve
# or: uvicorn pelican.api.main:app --reload

# 5. Start frontend
cd frontend && npm install && npm run dev
# open http://localhost:5173

# 6. Run agent pipeline
pelican agent run --theme "earnings quality factors"
# or: python scripts/run_agent.py --theme "earnings quality factors"

# 7. Run tests
pytest
```

---

## Confirmed Scope

| Decision | Choice |
|---|---|
| Universe | S&P 500 (survivorship-bias-free historical membership) |
| Portfolio mode | Dollar-neutral long/short (long Q5, short Q1) |
| Rebalance frequency | Monthly (21 trading days) |
| Price data | Daily OHLCV via yfinance |
| Research search | arXiv API only (free, no key, 1 req/3s) |
| Agent progress | SSE streaming — real-time node-by-node events |

## Key Design Decisions

**Point-in-time correctness** — every join in the backtest engine uses the
filing/price date as the anchor, not the current calendar date. Universe
membership is also tracked historically. Delistings are modeled as NaN
forward returns, not dropped rows, to avoid survivorship bias.

**Sandboxed code execution** — LLM-generated signal code runs in a restricted
Python environment (allowlisted imports: polars, numpy, math only) before
touching the live backtest engine.

**Signal acceptance gate** — the Critic enforces hard thresholds (IC t-stat ≥ 1.5,
L/S spread Sharpe ≥ 0.3) so the registry only accumulates genuine signals.

**SSE streaming** — the agent pipeline streams node events (node_start, llm_token,
node_complete, run_complete) over a Server-Sent Events connection so the React
dashboard can render progress in real time without polling.

**Polars over Pandas** — all cross-sectional operations (rank, z-score, lag,
forward-fill) are vectorized Polars expressions; 10–50× faster on the 500-stock
monthly panel than equivalent Pandas code.
