# pelican

**Agentic Factor Research Platform**

Pelican is a full-stack quantitative research system where LLM agents autonomously discover, implement, and backtest novel alpha signals from academic literature and alternative data. It combines a vectorized cross-sectional backtesting engine, a CVXPy portfolio optimizer, and a LangGraph agent pipeline — all wired together into a live research dashboard.

---

## What It Does

Pelican runs a closed research loop:

1. **Researcher Agent** crawls arXiv (`q-fin.PM`, `q-fin.ST`) nightly, extracts factor definitions from papers, and stores them in ChromaDB
2. **Codegen Agent** takes a parsed factor spec and generates Python signal code matching the standard signal interface
3. **Critic Agent** sandboxes and validates the generated code — checks for lookahead, NaN output, and minimum IC threshold
4. **Backtest Engine** runs accepted signals through a vectorized cross-sectional backtester, computing IC, ICIR, turnover, and quintile spreads
5. **Portfolio Constructor** combines surviving signals using long/short optimization with Ledoit-Wolf covariance shrinkage
6. **Report Agent** synthesizes results into a structured research memo, feeding rejection reasons back into the next codegen cycle
7. **Edgar Agent** pulls 10-K/10-Q filings, extracts MD&A sections, and scores YoY tone shifts as a standalone alternative data factor

---

## Architecture

```
                    LangGraph Pipeline                   
                                                         
  Researcher → Codegen → Critic → Backtest → Reporter   
       ↑                    │                    │       
       └────────────────────┘ (retry on reject)  │       
                                                 ↓      
                              Research Log (DuckDB)       

         ↕                              ↕
   FastAPI Backend              React Dashboard
   (SSE streaming)           (live node graph view)
```

**Data layer:** Polars + DuckDB, point-in-time correct, S&P 500 universe, daily OHLCV back to 2014

**Agent orchestration:** LangGraph with SSE streaming — watch Researcher → Codegen → Critic nodes light up in real time

**LLM provider:** OpenRouter (`meta-llama/llama-3.3-70b-instruct:free`)

---

## Tech Stack

| Layer | Tech |
|---|---|
| Data | Polars, DuckDB, yfinance, sec-edgar-downloader |
| Backtesting | NumPy, Polars (vectorized cross-sectional) |
| Portfolio | CVXPy, SciPy (Ledoit-Wolf shrinkage) |
| Agents | LangGraph, LangChain, ChromaDB |
| LLM | OpenRouter (Llama 3.3 70B) |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Recharts, Zustand |
| Testing | pytest |

---

## Project Structure

```
pelican/
├── pelican/
│   ├── data/              # Universe, prices, fundamentals, DuckDB store
│   ├── backtest/          # Vectorized engine, metrics, transaction costs
│   ├── factors/           # Classic factor library + registry
│   ├── portfolio/         # CVXPy optimizer, constraints, risk decomposition
│   ├── agents/
│   │   ├── graph.py       # LangGraph graph definition
│   │   ├── state.py       # Shared PipelineState
│   │   ├── researcher/    # arXiv search, PDF parsing, ChromaDB
│   │   ├── codegen/       # Signal code generation + sandbox executor
│   │   ├── critic/        # Validation: lookahead, NaN, IC threshold
│   │   ├── edgar/         # SEC filing download, MD&A extraction, scoring
│   │   └── reporter/      # Research memo synthesis
│   └── api/               # FastAPI routers, SSE endpoints, Pydantic schemas
├── frontend/              # React dashboard
├── data/
│   ├── raw/               # Downloaded data (gitignored)
│   ├── processed/         # pelican.duckdb (gitignored)
│   └── fixtures/          # Frozen test data (committed)
└── tests/                 # Mirrors package structure
```

---

## Classic Factor Library

Ground-truth signals used to validate the backtesting engine before any LLM-generated signals run through it:

| Factor | Description | Expected IC |
|---|---|---|
| Momentum | 12-1 month price return | +0.02 to +0.05 |
| Value | P/E, P/B inverse | +0.01 to +0.03 |
| Quality | ROE, low debt/equity | +0.01 to +0.04 |
| Low Volatility | Realized vol (trailing 60d) | +0.01 to +0.03 |
| Size | Market cap inverse | +0.01 to +0.02 |
| Reversal | 1-month return (negative) | +0.01 to +0.03 |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+

### Install

```bash
git clone https://github.com/jasmehar-k/pelican
cd pelican
pip install -e ".[dev]"
cp .env.example .env
```

### Seed market data

```bash
python -m pelican.data.loader --start 2014-01-01 --end 2024-01-01
```

This downloads S&P 500 daily OHLCV and fundamentals via yfinance and writes to `data/processed/pelican.duckdb`. Takes ~15 minutes on first run.

### Run tests

```bash
pytest                    # offline tests only (default)
pytest -m live            # also runs live yfinance/arXiv calls
```

### Start the backend

```bash
uvicorn pelican.api.main:app --reload
```

### Start the frontend

```bash
cd frontend
npm install
npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/agents/run` | Fire a research job, returns `run_id` |
| `GET` | `/agents/status/{run_id}` | SSE stream of node-by-node progress |
| `GET` | `/signals` | Browse all signals, sortable by IC/Sharpe/turnover |
| `GET` | `/signals/{id}` | Single signal with full backtest stats |
| `GET` | `/factors/{id}/tearsheet` | IC decay, quintile returns, drawdown |
| `POST` | `/portfolio/optimize` | Optimize over a selected signal set |

---

## Design Decisions

**Universe: S&P 500.** Liquid, well-studied, defensible. Avoids liquidity noise that would obscure signal quality in a broader universe.

**Long/short portfolio.** The quintile spread is the point of factor research. Long-only would neuter the optimizer.

**Monthly rebalance.** Standard for mixed fundamental + price signals. Each factor carries a `rebalance_freq` metadata field so momentum can run weekly without restructuring the engine.

**arXiv only for literature search.** No API keys needed, peer-reviewed quant finance papers only (`q-fin.PM`, `q-fin.ST`). Stronger research signal than general web search.

**SSE over polling.** Watching the LangGraph nodes fire in sequence in real time is the demo moment. Worth the backend complexity.

**Offline-first testing.** Tests never depend on yfinance or arXiv being up. Fixtures are frozen parquet files committed to the repo. Live tests are gated behind `pytest -m live` and excluded from CI.

---

## Known Limitations

**No incremental price updates.** If a stock split occurs after seeding, stored pre-split rows have wrong prices until a full re-seed. Always re-seed the full date range after any split.

**Zero-price delistings.** Bankrupt stocks halted at $0 produce `log(0) = -inf` log returns. These are filtered at ingest (`close > 0`) but worth monitoring.

**Wikipedia universe lag.** S&P 500 constituent changes on Wikipedia are community-maintained and sometimes logged weeks late. Fine for historical backtesting; not suitable for live use without a paid data source.

---

## Build Stages

| Stage | Description | Status |
|---|---|---|
| 1 | Data Foundation | ✅ |
| 2 | Backtesting Engine | 🔄 |
| 3 | Classic Factor Library | ⏳ |
| 4 | Portfolio Construction | ⏳ |
| 5 | Codegen Agent | ⏳ |
| 6 | Researcher Agent | ⏳ |
| 7 | Full LangGraph Pipeline | ⏳ |
| 8 | Edgar Sentiment Agent | ⏳ |
| 9 | FastAPI Backend | ⏳ |
| 10 | React Dashboard | ⏳ |

---

## License

MIT
