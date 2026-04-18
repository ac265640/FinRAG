# FinRAG — 15-Day Build Roadmap

> A production-grade, citation-enforced financial research assistant over SEC filings and earnings call transcripts.

FinRAG answers questions about SEC filings (10-K, 10-Q, 8-K) and earnings call transcripts. Every answer is grounded in a specific paragraph from a specific filing, with company, period, section, and page attached. When evidence doesn't support a claim, the system refuses to answer. It's continuously evaluated for faithfulness, monitored with distributed tracing, and gated by a CI pipeline that fails the build if quality drops.

---

## Roadmap Overview

| Day | Phase | Deliverable | Key Concept Unlocked | Status |
|-----|-------|-------------|---------------------|--------|
| 1 | Foundation | Project scaffold, env setup, EDGAR ingestion pipeline | SEC EDGAR API, structured data ingestion, environment isolation | ✅ Done |
| 2 | Foundation | Section-aware chunker with metadata attachment | Section-boundary chunking, metadata-enriched chunks | ✅ Done |
| 3 | Foundation | ChromaDB vector store + embedding pipeline | Vector store design, embedding model selection | ⬜ |
| 4 | Retrieval | BM25 keyword index + basic retrieval harness | Sparse retrieval, inverted indices, term frequency | ⬜ |
| 5 | Retrieval | Hybrid retrieval with Reciprocal Rank Fusion | Fusion strategies, complementary retrieval signals | ⬜ |
| 6 | Retrieval | Cross-encoder reranker + retrieval eval harness | Bi-encoder vs cross-encoder, precision at k | ⬜ |
| 7 | Generation & Safety | LangGraph orchestration: state machine, nodes, routing | State machines, explicit graph vs chains | ⬜ |
| 8 | Generation & Safety | Citation enforcer + structured output + LLM generation | Citation grounding, Pydantic output schemas | ⬜ |
| 9 | Generation & Safety | Guardrails: prompt injection, PII, NeMo, output validation | Input/output guardrails, policy enforcement | ⬜ |
| 10 | Generation & Safety | Versioned prompt configs + agent memory | Prompt versioning, conversation state management | ⬜ |
| 11 | API & Observability | FastAPI layer: streaming SSE, middleware, MCP interface | SSE streaming, middleware patterns | ⬜ |
| 12 | API & Observability | Langfuse instrumentation: traces, token cost, latency | Distributed tracing, production metrics | ⬜ |
| 13 | Evaluation & CI Gate | Golden dataset construction (50 Q/A pairs) + RAGAS eval | Evaluation dataset design, faithfulness metrics | ⬜ |
| 14 | Evaluation & CI Gate | LLM-as-Judge citation scorer + offline eval script | LLM-as-Judge pattern, citation accuracy | ⬜ |
| 15 | Evaluation & CI Gate | CI quality gate, final integration test, README, demo | CI enforcement, threshold gating, documentation | ⬜ |

---

## Phase Details

### Phase 1 — Foundation (Days 1-3)

**Goal:** Get data into the system. Clean, structured, metadata-rich.

#### Day 1: Project Scaffold & EDGAR Ingestion ✅
- Project structure with `pyproject.toml`, `src/` layout
- Config management via `pydantic-settings` (no hardcoded secrets)
- Async EDGAR client: ticker-to-CIK resolution, filing download, section parsing
- CLI ingestion script (`scripts/ingest.py`)
- 13 tests: config validation, section parsing, EDGAR integration

#### Day 2: Section-Aware Chunker
- Chunk within section boundaries, never across them
- Token-bounded windows (500-800 tokens) with 100-150 token overlap
- Each chunk carries metadata: ticker, filing_type, fiscal_period, section, page
- Handle edge cases: very short sections, very long paragraphs, tables

#### Day 3: ChromaDB Vector Store + Embeddings
- Initialize ChromaDB persistent collection
- Embed chunks using sentence-transformers
- Store chunks with full metadata for filtered retrieval
- Build embedding pipeline that processes saved filings into vectors

---

### Phase 2 — Retrieval (Days 4-6)

**Goal:** Find the right chunks for any financial question. Hybrid search, not just vectors.

#### Day 4: BM25 Keyword Index
- Build BM25 index over chunked documents using rank-bm25
- Basic retrieval harness: query in, ranked chunks out
- Why BM25: catches exact financial terms embeddings might miss ("diluted EPS", "goodwill impairment")

#### Day 5: Hybrid Retrieval with RRF
- Run BM25 and vector search concurrently
- Merge results via Reciprocal Rank Fusion
- Multi-query retrieval: generate query rephrasings for broader recall
- HyDE: Hypothetical Document Embeddings for earnings call queries
- Metadata filtering: narrow by ticker, filing period

#### Day 6: Cross-Encoder Reranker
- Rerank top 20-30 candidates with cross-encoder (sentence-transformers)
- Cross-encoder sees query + document together for higher precision
- Return top 5-8 chunks to the LLM
- Build retrieval evaluation harness: measure precision@k, recall@k

---

### Phase 3 — Generation & Safety (Days 7-10)

**Goal:** Generate grounded answers with citations. Block hallucinations and adversarial inputs.

#### Day 7: LangGraph Orchestration
- Model pipeline as explicit state machine using LangGraph
- Nodes: query router, retriever, reranker, citation checker, generator, validator
- Conditional routing: calculation node for arithmetic queries via function calling
- Maximum step count to prevent tool call loops

#### Day 8: Citation Enforcer + Structured Output
- Citation enforcement: decline when max relevance score is below threshold
- Structured output via Pydantic: answer text, citation list (chunk_id, filing_ref, section, page), confidence
- Output validation: reject responses with missing/malformed citations
- Retry once with stricter prompt before declining

#### Day 9: Guardrails Layer
- **Input:** Prompt injection detection (classifier + heuristics), PII detection/scrubbing via guardrails-ai
- **Output:** NeMo Guardrails for policy enforcement (no investment advice, mandatory disclaimers)
- Both integrated into FastAPI middleware layer

#### Day 10: Prompt Configs + Agent Memory
- Versioned prompt configs in YAML (committed to git)
- Agent memory: track loaded filings, discussed entities, prior answers across turns
- Enable follow-up questions ("how does that compare to last quarter?")

---

### Phase 4 — API & Observability (Days 11-12)

**Goal:** Expose the system as an API. Make every request debuggable.

#### Day 11: FastAPI Layer
- SSE streaming responses (citations appear progressively)
- Middleware: auth token validation, request ID injection, rate limiting, guardrails, logging
- MCP tool server interface for agent integration
- Versioned prompt configs trigger CI eval (same as code changes)

#### Day 12: Langfuse Instrumentation
- Full execution trace per request: chunks retrieved, reranker ordering, prompt version, raw LLM response, token counts
- Production metrics: p50/p95 latency, cost per request, citation coverage rate, decline rate
- Every failure classified and logged with failure type

---

### Phase 5 — Evaluation & CI Gate (Days 13-15)

**Goal:** Prove the system works. Gate deployments on quality.

#### Day 13: Golden Dataset + RAGAS
- Build 50+ manually verified Q/A pairs covering:
  - Direct numerical extraction ("what was Apple's FCF in Q3 2024?")
  - Multi-hop comparison ("did gross margin improve in the quarter where supply chain costs were flagged?")
  - Contradiction detection ("does the CEO's language match risk disclosures?")
  - Out-of-scope questions (should produce decline, not hallucination)
- RAGAS evaluation: faithfulness, answer relevancy, context precision

#### Day 14: LLM-as-Judge
- Separate model scores citation accuracy per cited chunk
- Catches misattributed citations that RAGAS metrics miss
- Offline eval script for running evaluations on demand

#### Day 15: CI Quality Gate + Final Polish
- GitHub Actions workflow: every PR triggers evaluation run
- **Hard gates:** faithfulness >= 0.85, citation coverage >= 0.90 — build fails otherwise
- Final integration test covering full pipeline
- README with setup, architecture diagram, demo instructions

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph |
| Vector Store | ChromaDB |
| Sparse Retrieval | BM25 (rank-bm25) |
| Dense Retrieval | sentence-transformers |
| Reranking | cross-encoder (sentence-transformers) |
| API | FastAPI + SSE |
| Guardrails | guardrails-ai + NeMo Guardrails |
| Observability | Langfuse |
| Evaluation | RAGAS + LLM-as-Judge |
| Config | pydantic-settings + versioned YAML |
| CI | GitHub Actions |

---

## Target Directory Structure

```
finrag/
├── pyproject.toml
├── README.md
├── ROADMAP.md                      ← this file
├── DEBT_LEDGER.md
├── configs/prompts/
│   ├── v1_generation.yaml          (Day 10)
│   └── v1_retrieval.yaml           (Day 10)
├── src/finrag/
│   ├── config.py
│   ├── ingestion/
│   │   ├── edgar_client.py         (Day 1)
│   │   └── chunker.py              (Day 2)
│   ├── vectorstore/
│   │   └── chroma_store.py         (Day 3)
│   ├── retrieval/
│   │   ├── bm25_index.py           (Day 4)
│   │   ├── hybrid.py               (Day 5)
│   │   ├── reranker.py             (Day 6)
│   │   └── eval_harness.py         (Day 6)
│   ├── orchestration/
│   │   ├── graph.py                (Day 7)
│   │   ├── nodes.py                (Day 7)
│   │   ├── citation.py             (Day 8)
│   │   ├── generator.py            (Day 8)
│   │   └── memory.py               (Day 10)
│   ├── guardrails/
│   │   ├── input_guards.py         (Day 9)
│   │   ├── output_guards.py        (Day 9)
│   │   └── nemo_config/            (Day 9)
│   ├── api/
│   │   ├── app.py                  (Day 11)
│   │   ├── middleware.py           (Day 11)
│   │   ├── routes.py              (Day 11)
│   │   └── mcp_server.py          (Day 11)
│   └── observability/
│       └── langfuse_tracer.py      (Day 12)
├── tests/                          (one test file per module)
├── eval/
│   ├── golden_dataset.json         (Day 13)
│   ├── ragas_eval.py               (Day 13)
│   ├── llm_judge.py                (Day 14)
│   └── run_eval.py                 (Day 14)
├── scripts/
│   ├── ingest.py                   (Day 1)
│   └── run_eval_ci.py              (Day 15)
└── .github/workflows/
    └── quality_gate.yml            (Day 15)
```

---

## Contribution Rules

- Every function has type hints and a docstring (Args, Returns, Raises)
- No magic numbers. Constants are named and explained.
- Errors are caught at the layer that can handle them.
- Structured logging (JSON) after Day 3. No bare `print()`.
- No hardcoded API keys. Environment variables from Day 1.
- Every module has a corresponding test file on the day it's created.
- See `DEBT_LEDGER.md` for known shortcuts and their resolution targets.
