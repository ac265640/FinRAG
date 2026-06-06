
# FinRAG

> A production-grade, citation-enforced financial RAG system integrating state-of-the-art AI orchestration with robust enterprise software engineering.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Live Platform](https://img.shields.io/badge/Live%20Platform-Vercel-000000?style=flat&logo=vercel)](https://fin-rag-five.vercel.app)

**🌐 Live Platform:** [fin-rag-five.vercel.app](https://fin-rag-five.vercel.app)

---

## What This Does

FinRAG is an enterprise-grade financial research engine designed to query SEC filings (10-K, 10-Q, 8-K) and earnings call transcripts. It leverages advanced LLM reasoning to generate answers that are 100% grounded in source text, enforcing exact citations (company, filing period, section, and page). To eliminate hallucination, the system executes an automated refusal protocol if the evidence is insufficient.

### Key Capabilities

- **LangGraph Orchestration** — Multi-agent state machine routing requests based on query intent and complexity.
- **Hybrid Retrieval (RRF)** — Fusing BM25 sparse search and dense sentence embeddings via Reciprocal Rank Fusion.
- **Cross-Encoder Reranking** — Second-stage transformer validation for precision retrieval.
- **Multi-turn Session Memory** — Thread-safe session tracking, coreference resolution, and entity memory.
- **Automated LLM-as-a-Judge** — Real-time generation evaluation scoring citation accuracy and faithfulness.
- **Containerized Stack (Docker)** — Multi-container local orchestration (Next.js frontend, FastAPI backend, Redis, PostgreSQL).
- **Sub-Millisecond Caching (Redis)** — Ultra-fast caching for frequent prompt/response pairs.
- **Query Analytics Engine (PostgreSQL)** — Persistent SQL logging tracking token costs, latency distribution, and evaluation metrics.
- **Asynchronous Jobs (Async Tasks)** — FastAPI background task workers for parallel filing downloads, section-aware chunking, and vector indexing.
- **API Rate Limiting** — Bulletproof client rate-limiting protection.
- **Structured Logging** — Standardized, production-grade JSON logging for observability and error tracing.
- **Streaming API** — Server-Sent Events for progressive UI rendering.
- **Guardrails** — Prompt injection detection, PII filtering, and output verification.
- **CI Quality Gates** — Automated testing builds failing if faithfulness < 0.85 or citation coverage < 0.90.

---

##   Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                            │
│  POST /query  │  POST /query/stream  │  GET /metrics           │
└──────┬────────┴──────────┬───────────┴──────────┬──────────────┘
       │                   │                      │
       ▼                   ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestration                    │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Input    │──▶│ Retrieve │──▶│ Rerank   │──▶│ Route    │ │
│  │  Guard    │   │  (Hybrid)│   │ (Cross-  │   │ (Keyword │ │
│  │          │   │          │   │  Encoder) │   │  Router) │ │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘ │
│                                                      │       │
│                    ┌─────────────────┬───────────────┘       │
│                    ▼                 ▼                        │
│              ┌──────────┐     ┌──────────┐                   │
│              │ Generate │     │ Calculate │                   │
│              │ (Gemini) │     │ (Gemini)  │                   │
│              └────┬─────┘     └────┬─────┘                   │
│                   │                │                          │
│                   ▼                ▼                          │
│              ┌──────────┐   ┌──────────┐                     │
│              │ Validate │──▶│ Output   │                     │
│              │ Citations│   │ Guard    │                     │
│              └──────────┘   └──────────┘                     │
└──────────────────────────────────────────────────────────────┘
       │                                          │
       ▼                                          ▼
┌──────────────┐                        ┌──────────────────┐
│  ChromaDB    │                        │    Langfuse       │
│  + BM25      │                        │    Tracing        │
│  Vector Store│                        │    + Metrics      │
└──────────────┘                        └──────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (state machine with conditional routing) |
| Vector Store | ChromaDB (persistent, metadata-filtered) |
| Sparse Retrieval | BM25 via `rank-bm25` |
| Dense Retrieval | `sentence-transformers` (all-MiniLM-L6-v2) |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Generation | Google Gemini 2.0 Flash via `langchain-google-genai` |
| API | FastAPI + SSE (`sse-starlette`) |
| Caching | Redis (sub-millisecond prompt/response cache & rate limiting) |
| Analytics Database | PostgreSQL / Neon DB (via SQLAlchemy & asyncpg) |
| Frontend UI | Next.js (React, TypeScript, TailwindCSS) |
| Guardrails | Custom regex + policy-based input/output guards |
| Observability | Langfuse (traces, spans, token costs) |
| Evaluation | RAGAS metrics + LLM-as-Judge citation scorer |
| Config | `pydantic-settings` + versioned YAML prompts |
| CI | GitHub Actions (lint → test → eval gate) |
| Infrastructure | Docker & Docker Compose (multi-container local orchestration) |

---

## Setup

### Prerequisites

- Python 3.11+
- Google API key (for Gemini LLM)
- Docker & Docker Compose (optional, but highly recommended for complete multi-container setup)

### Option 1: Docker Compose (Quickest & Recommended)

Run the entire stack (Next.js UI, FastAPI Backend, Redis prompt cache, and PostgreSQL analytics) with a single command:

```bash
# Clone the repo
git clone https://github.com/ac265640/FinRAG.git
cd FinRAG

# Spin up all containers
docker-compose up --build
```

Make sure to edit the `.env` file generated in the project root with your credentials.

### Option 2: Local Virtual Environment Installation

```bash
# Clone the repo
git clone https://github.com/ac265640/FinRAG.git
cd FinRAG

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Environment Configuration

```bash
# Copy example env file
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required: Google Gemini API key
GOOGLE_API_KEY=your_key_here

# Optional: Langfuse observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Optional: API authentication
FINRAG_API_KEY=your_api_secret
```

---

## Quick Start

### 1. Ingest a Filing

```bash
# Download and process Apple's latest 10-K
python scripts/ingest.py --ticker AAPL --filing-type 10-K --count 1
```

This downloads the filing from SEC EDGAR, parses sections, chunks with metadata, and indexes into ChromaDB + BM25.

### 2. Start the API Server

```bash
uvicorn finrag.api.app:app --reload --port 8000
```

### 3. Query the Pipeline

```bash
# Synchronous query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple total net revenue for fiscal year 2024?"}'

# Streaming query (SSE)
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple total net revenue for fiscal year 2024?"}'
```

### 4. Check Metrics

```bash
curl http://localhost:8000/api/v1/metrics
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Synchronous JSON response |
| `/api/v1/query/stream` | POST | Server-Sent Events streaming |
| `/api/v1/sessions/{id}` | GET | Session state inspection |
| `/api/v1/sessions/{id}` | DELETE | Clear a session |
| `/api/v1/config/prompts` | GET | Active prompt versions |
| `/api/v1/metrics` | GET | Production metrics (p50/p95 latency, costs, rates) |
| `/api/v1/available-filings` | GET | List processed filing details dynamically (companies, periods, types) |
| `/api/v1/ingest` | POST | Queue asynchronous background SEC filing download and vector storage |
| `/api/v1/ingest/{id}/status` | GET | Check async background ingestion progress status |
| `/api/v1/analytics/queries` | GET | Fetch query history, cost trackers, and performance metrics |

### Query Request

```json
{
  "query": "What was Apple's free cash flow in FY2024?",
  "session_id": "optional-session-id",
  "metadata_filter": {"ticker": "AAPL"}
}
```

### Query Response

```json
{
  "answer": "Apple's free cash flow in FY2024 was...",
  "citations": [
    {
      "chunk_id": "abc123",
      "filing_reference": "AAPL 10-K FY2024, Item 7 - MD&A",
      "section": "Item 7",
      "relevance_score": 0.92
    }
  ],
  "session_id": "auto-generated-uuid",
  "confidence": 0.87,
  "route": "retrieve",
  "prompt_version": "v2",
  "metadata": {
    "request_id": "uuid",
    "trace_id": "langfuse-trace-id",
    "total_latency_ms": 1250
  }
}
```

---

## Evaluation

### Golden Dataset

50 manually verified Q/A pairs across 4 categories:

| Category | Count | Description |
|----------|-------|-------------|
| Numerical Extraction | 15 | Direct financial data queries |
| Multi-hop Comparison | 12 | Cross-document reasoning |
| Contradiction Detection | 11 | Narrative vs. data consistency |
| Out-of-scope | 12 | Should produce decline, not hallucination |

### Run Evaluations

```bash
# RAGAS metrics (faithfulness, relevancy, precision, coverage)
python -m finrag.evaluation.run_eval --mode ragas --threshold 0.85

# LLM-as-Judge citation scoring
python -m finrag.evaluation.run_eval --mode judge --threshold 0.90

# Full evaluation (both)
python -m finrag.evaluation.run_eval --mode full --output report.json

# Filter by category
python -m finrag.evaluation.run_eval --mode ragas --category numerical
```

### CI Quality Gates

Every PR triggers the [quality gate workflow](.github/workflows/quality-gate.yml):

```
lint → unit tests (60% coverage) → RAGAS eval (≥0.85) → Judge eval (≥0.90)
```

Builds fail if quality thresholds are not met.

---

## Project Structure

```
FinRAG/
├── .github/workflows/         # CI quality gate
│   └── quality-gate.yml
├── alembic/                   # PostgreSQL migration scripts & schema env
├── configs/                    # Versioned prompt configs (YAML)
├── data/                      # SEC filing database storage
│   ├── chroma/                # ChromaDB SQLite3 persistence database
│   └── raw/                   # SEC raw filing HTML files grouped by ticker
├── finrag-ui/                 # Next.js 14 frontend interactive application
│   ├── app/                   # App Router pages and analytics charts
│   ├── components/            # Chat component, citations highlight, sidebar
│   ├── lib/                   # API clients, types, and analytics helpers
│   └── Dockerfile.frontend    # Frontend Docker image configuration
├── scripts/
│   └── ingest.py              # EDGAR ingestion CLI
├── src/finrag/
│   ├── ingestion/             # EDGAR client, section chunker
│   ├── vectorstore/           # ChromaDB store
│   ├── retrieval/             # BM25, hybrid retriever
│   ├── orchestration/         # LangGraph, nodes, routing, memory
│   ├── guardrails/            # Input/output guards
│   ├── api/                   # FastAPI app, routes, middleware, MCP
│   ├── observability/         # Langfuse tracer, metrics
│   └── evaluation/            # Golden dataset, RAGAS, LLM-as-Judge
├── tests/                     # 16 test modules, 300+ tests
├── docker-compose.yml         # Local microservice container orchestration
├── Dockerfile                 # Backend FastAPI space configuration
├── DEBT_LEDGER.md             # Technical debt tracking
└── pyproject.toml             # Dependencies and tooling config
```

---

## Development

### Run Tests

```bash
# All tests
python -m pytest tests/ -v --tb=short

# Specific day/module
python -m pytest tests/test_integration.py -v

# With coverage
python -m pytest tests/ --cov=finrag --cov-report=term-missing
```

### Lint

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key (for embedding and answer generation) |
| `EDGAR_USER_AGENT` | Yes | SEC EDGAR required user agent string (e.g. `Company info@company.com`) |
| `DATABASE_URL` | Yes | PostgreSQL connection URL for logging query analytic stats |
| `REDIS_URL` | No | Redis connection URL for sub-millisecond API response caching |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse observability metrics dashboard public identifier key |
| `LANGFUSE_SECRET_KEY` | No | Langfuse observability metrics dashboard secret developer key |
| `FINRAG_API_KEY` | No | Secret bearer security token required for production API authorization |
| `FINRAG_INIT_PIPELINE` | No | Set `false` to skip backend model pipeline initialization during testing |
| `FINRAG_CORS_ORIGINS` | No | Comma-separated list or JSON array defining allowed CORS request origins |

---

## License

MIT
