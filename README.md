# FinRAG

A production-grade, citation-enforced financial research assistant over SEC filings and earnings call transcripts.

## What This Does

FinRAG answers questions about SEC filings (10-K, 10-Q, 8-K) and earnings call transcripts. Every answer is grounded in a specific paragraph from a specific filing, with company, period, section, and page attached. When evidence doesn't support a claim, the system refuses to answer.

## Stack

- **Orchestration:** LangGraph
- **Vector Store:** ChromaDB
- **Retrieval:** BM25 + sentence-transformers + cross-encoder reranking
- **Evaluation:** RAGAS + LLM-as-Judge
- **Observability:** Langfuse
- **API:** FastAPI with SSE streaming
- **Guardrails:** guardrails-ai + NeMo Guardrails

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd finrag

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your details
```

## Quick Start

```bash
# Download a filing
python scripts/ingest.py --ticker AAPL --filing-type 10-K --count 1
```

## Project  Status

🏗️ Under active development. See DEBT_LEDGER.md for known shortcuts.
