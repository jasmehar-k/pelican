# Pelican

An agentic factor research platform. LLM agents autonomously discover, implement, and backtest quantitative alpha signals from academic literature. Accepted signals feed a risk-aware portfolio optimizer. A FastAPI + React dashboard lets you browse signals, inspect backtest tearsheets, and monitor the agent pipeline live.

---

## Screenshots

### Dashboard
Overview of registered signals, agent runs, top performers by IC t-stat, and a quick-launch agent panel.

![Dashboard](docs/screenshot_dashboard.png)

### Factor Library
Full signal table sortable by Sharpe, IC, or turnover. Click any row to open its tearsheet with equity curves and quintile spreads. Agent-discovered signals appear here automatically alongside the hand-coded factor library.

![Signals](docs/screenshot_signals.png)

### Research Log
Full lineage for every agent run — papers fetched from arXiv, the generated hypothesis, the produced signal code, and the backtest verdict. Click a row to inspect the full run detail on the right.

![Research Log](docs/screenshot_research.png)

### Portfolio Construction
Select signals, run the CVXPy optimizer, and view the composite basket equity curve and weights.

![Portfolio](docs/screenshot_portfolio.png)

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-org/pelican
cd pelican
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Required: OPENROUTER_API_KEY, DUCKDB_PATH, DATA_DIR
```

Get a free API key at [openrouter.ai](https://openrouter.ai). The default model (`meta-llama/llama-3.3-70b-instruct:free`) is free with no rate-limit fees.

```bash
# 3. Seed data  (~30 min, ~10 GB — prices first, then fundamentals)
python scripts/seed_data.py
python scripts/seed_fundamentals.py

# 4. Start the API server
uvicorn pelican.api.main:app --reload        # http://localhost:8000

# 5. Start the frontend (separate terminal)
cd frontend && npm install && npm run dev    # http://localhost:5173
```

---

## Running the Agent

### From the dashboard (recommended)

1. Open `http://localhost:5173`
2. Type a research theme in the **Run a research pipeline** panel — e.g. `earnings quality factors` or `low volatility anomaly`
3. Click **Run Agent**
4. Watch the researcher, coder, and critic nodes stream live in the right panel
5. On accept, the signal is immediately added to the **Factor Library** and visible on the Signals page

### From the terminal

```bash
python scripts/run_agent.py --theme "earnings quality factors"

# Skip the arXiv researcher (uses theme directly as signal description)
python scripts/run_agent.py --theme "12-1 month momentum" --no-research
```

### Good themes to try

| Theme | What the agent typically discovers |
|---|---|
| `earnings quality factors` | Accruals, cash-flow-to-price, earnings persistence |
| `low volatility anomaly` | Negative realized vol, min-variance tilt |
| `value investing pe ratio` | Earnings yield, negative P/E rank |
| `price momentum 12 month` | 12-1 momentum, intermediate-horizon continuation |
| `short-term reversal` | 1-month reversal, liquidity provision premium |
| `quality profitability` | ROE, gross profit margin, asset turnover |
| `52-week high momentum` | Distance from 52-week high as an anchor signal |
| `sentiment earnings announcements` | Post-earnings drift, analyst revision breadth |

---

## How It Works

```
Research Theme
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  RESEARCHER — searches arXiv for relevant papers, extracts a    │
│  grounded signal hypothesis with economic rationale and the     │
│  specific data columns to use                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ hypothesis + citations
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  CODER — translates hypothesis into a validated                 │
│  compute_signal(df: pl.DataFrame) -> pl.Series function;        │
│  sandboxed exec rejects disallowed imports and look-ahead bias  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ generated code
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  CRITIC — runs a real point-in-time backtest, checks            │
│  IC t-stat ≥ 0.5 and net Sharpe ≥ 0.3; returns accept or       │
│  reject with written feedback                                   │
└────────┬──────────────────┴──────────────────────┬─────────────┘
         │ reject (up to 2 retries)                │ accept
         └──────────► CODER (retry)                ▼
                                         Signal added to registry
                                         + investment memo written
                                         + persisted to DuckDB
```

The graph is a LangGraph state machine. Each node is a pure `(State) -> State` function; the Critic's conditional edge drives the retry loop.

---

## Architecture

```
pelican/
├── pelican/
│   ├── agents/         # LangGraph graph, researcher/coder/critic/reporter nodes
│   │   ├── prompts/    # system prompts (researcher.md, coder.md, reporter.md)
│   │   └── tools/      # arXiv search, PDF extract, ChromaDB vector store, sandbox exec
│   ├── backtest/       # vectorized Polars engine, signal registry, metrics
│   ├── data/           # DuckDB store, price/fundamental ingestion
│   ├── portfolio/      # CVXPy optimizer, Ledoit-Wolf risk model, signal combiner
│   ├── api/            # FastAPI app + SSE streaming router
│   └── utils/          # pydantic-settings config, structured logging
├── frontend/           # React 18 + Vite + TypeScript + Recharts dashboard
├── scripts/            # seed_data.py, run_agent.py, run_factor_library.py
└── tests/              # pytest suite (~360 tests, fully mocked LLM/HTTP)
```

### Data layer

Single DuckDB file. Three core tables:

| Table | Contents |
|---|---|
| `sp500_universe` | Survivorship-bias-free S&P 500 membership history |
| `prices` | Daily OHLCV + `log_return_1d`, `forward_return_21d` |
| `fundamentals` | Quarterly ratios with `available_date = period_end + 45d` (point-in-time anchor) |
| `research_log` | Full lineage for every agent run (papers, hypothesis, code, metrics) |

### Backtest engine

Fully vectorized over a Polars `(date × ticker)` panel. Monthly rebalance (21-day hold). Per rebalance date: queries point-in-time universe and fundamentals, builds cross-section features (lagged closes, rolling vols), computes signal scores, cross-sectional rank → z-score, forms Q5 (long) vs Q1 (short) equal-weighted quintile portfolios, measures 21-day forward returns.

### Signal registry

All signals — hand-coded and agent-discovered — live in an in-process `_REGISTRY` dict. Hand-coded signals register via `@register(SignalSpec(...))` at import time. Agent signals are compiled and inserted by `register_dynamic()` the moment they're accepted, and reloaded from `research_log` on server restart via `load_dynamic_signals()`.

### SSE streaming

The agent pipeline runs in a thread executor. Events (`node_start`, `llm_token`, `node_complete`, `run_complete`) are pushed into an `asyncio.Queue` and drained by the SSE handler. The React frontend connects via `EventSource` and renders each node's state as it arrives.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Agent orchestration | LangGraph + LangChain |
| LLM | OpenRouter — `meta-llama/llama-3.3-70b-instruct:free` |
| DataFrame engine | Polars |
| Storage | DuckDB |
| Vector store | ChromaDB (paper deduplication) |
| Price + fundamental data | yfinance |
| Portfolio optimization | CVXPy (CLARABEL/OSQP) |
| API server | FastAPI + Uvicorn |
| Frontend | React 18, Vite, TypeScript, Recharts |
| Testing | pytest (~360 tests) |

---

## Key Design Decisions

**Point-in-time correctness** — every join in the backtest engine uses `available_date ≤ rebalance_date` to anchor fundamentals, and `entry_date / exit_date` for universe membership. No look-ahead bias by construction.

**Sandboxed code execution** — LLM-generated signal code runs in a restricted namespace (allowlisted imports: `polars`, `numpy`, `math` only). The executor also validates output shape, dtype, null rate, and absence of `inf`/`nan`.

**Signal acceptance gate** — the Critic enforces IC t-stat ≥ 0.5 and net L/S Sharpe ≥ 0.3 so the registry only accumulates genuine signals.

**Hypothesis-grounded generation** — the Researcher extracts up to 3 distinct signal hypotheses per arXiv search, each with economic rationale and exact column names. ChromaDB deduplicates against previously explored papers to prevent the agent from re-proposing the same ideas.
