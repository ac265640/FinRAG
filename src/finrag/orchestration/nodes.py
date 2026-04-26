"""Pipeline node functions for the FinRAG LangGraph state machine.

Each function is a graph node that reads from and writes to the
shared GraphState. Nodes are pure functions of state — no hidden
dependencies, no side effects beyond logging.

Node responsibilities:
- retrieve: Run hybrid retrieval (BM25 + vector + RRF)
- rerank: Apply cross-encoder reranking to retrieved chunks
- generate: LLM generation with citation enforcement (Day 8)
- calculate: Numerical computation via LLM with function calling
- validate: Check answer quality, citations, enforcement results
- decline: Produce a polite refusal for out-of-scope queries
- handle_error: Graceful error handling for pipeline failures

Design decisions:
- Each node catches its own exceptions and writes to state['error']
  rather than crashing the graph. This lets the validate node
  produce a user-friendly error message.
- generate uses RAGGenerator with CitationEnforcer for grounded
  answers. Retries once with stricter prompt on enforcement failure.
- Nodes increment step_count for the loop guard.
- Max steps default is 10 — enough for retrieve → rerank → generate
  with one retry, but catches infinite loops.
"""

import structlog

from finrag.orchestration.generator import RAGGenerator
from finrag.orchestration.state import GraphState
from finrag.retrieval.hybrid import HybridRetriever
from finrag.retrieval.reranker import CrossEncoderReranker

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default max steps before forced termination. 10 is generous for
# a pipeline that normally runs 3-4 nodes. Catches runaway loops.
DEFAULT_MAX_STEPS = 10

# Minimum number of chunks required after reranking for generation.
# Below this, we don't have enough evidence to generate an answer.
MIN_CHUNKS_FOR_GENERATION = 1

# Default reranker top-k for the pipeline.
PIPELINE_RERANK_TOP_K = 5


# --------------------------------------------------------------------------- #
# Node: Retrieve
# --------------------------------------------------------------------------- #


def retrieve(
    state: GraphState,
    *,
    hybrid_retriever: HybridRetriever,
) -> dict:
    """Run hybrid retrieval to fetch candidate chunks.

    Combines BM25 sparse search and ChromaDB dense search via
    Reciprocal Rank Fusion. Fetches 20-30 candidates to give
    the reranker enough signal to work with.

    Args:
        state: Current graph state with 'query' field.
        hybrid_retriever: Initialized HybridRetriever instance.

    Returns:
        Dict with retrieved_chunks, retrieval_metadata, and
        incremented step_count.
    """
    query = state.get("query", "")
    metadata_filter = state.get("metadata_filter")
    step_count = state.get("step_count", 0)

    try:
        chunks = hybrid_retriever.retrieve(
            query=query,
            n_results=30,  # Oversample for reranker
            where=metadata_filter,
            use_multi_query=True,
        )

        logger.info(
            "retrieval_complete",
            query_preview=query[:80],
            chunks_retrieved=len(chunks),
            has_filter=metadata_filter is not None,
        )

        return {
            "retrieved_chunks": chunks,
            "retrieval_metadata": {
                "num_candidates": len(chunks),
                "filter_applied": metadata_filter,
                "multi_query": True,
            },
            "step_count": step_count + 1,
        }

    except Exception as e:
        logger.error("retrieval_failed", error=str(e), query_preview=query[:80])
        return {
            "retrieved_chunks": [],
            "retrieval_metadata": {"error": str(e)},
            "error": f"Retrieval failed: {e}",
            "step_count": step_count + 1,
        }


# --------------------------------------------------------------------------- #
# Node: Rerank
# --------------------------------------------------------------------------- #


def rerank(
    state: GraphState,
    *,
    reranker: CrossEncoderReranker,
) -> dict:
    """Rerank retrieved chunks using cross-encoder scoring.

    Takes the top 20-30 candidates from hybrid retrieval and
    reorders them by true semantic relevance. Returns the top 5
    chunks for generation.

    If no chunks were retrieved (retrieval failure or empty result),
    passes through an empty list — the validate node will catch this.

    Args:
        state: Current graph state with 'retrieved_chunks' field.
        reranker: Initialized CrossEncoderReranker instance.

    Returns:
        Dict with reranked_chunks and incremented step_count.
    """
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    step_count = state.get("step_count", 0)

    if not chunks:
        logger.warning("rerank_skipped_no_chunks", query_preview=query[:80])
        return {
            "reranked_chunks": [],
            "step_count": step_count + 1,
        }

    try:
        reranked = reranker.rerank(
            query=query,
            candidates=chunks,
            top_k=PIPELINE_RERANK_TOP_K,
        )

        logger.info(
            "rerank_complete",
            query_preview=query[:80],
            input_chunks=len(chunks),
            output_chunks=len(reranked),
            top_score=reranked[0]["reranker_score"] if reranked else 0.0,
        )

        return {
            "reranked_chunks": reranked,
            "step_count": step_count + 1,
        }

    except Exception as e:
        logger.error("rerank_failed", error=str(e))
        # Fall back to unreranked chunks (degraded but functional)
        return {
            "reranked_chunks": chunks[:PIPELINE_RERANK_TOP_K],
            "error": f"Reranking failed (using unranked fallback): {e}",
            "step_count": step_count + 1,
        }


# --------------------------------------------------------------------------- #
# Node: Generate (LLM + Citation Enforcement)
# --------------------------------------------------------------------------- #


def generate(
    state: GraphState,
    *,
    rag_generator: RAGGenerator | None = None,
) -> dict:
    """Generate an answer from reranked context chunks using LLM.

    Uses RAGGenerator for LLM calls with citation enforcement.
    Falls back to stub behavior if no generator is provided
    (for backward compatibility with Day 7 tests).

    Pipeline:
    1. Pre-decline check (context quality too poor?)
    2. LLM call with structured output (CitedAnswer schema)
    3. Citation enforcement (hallucination detection)
    4. Retry once with stricter prompt if enforcement fails
    5. Return answer + enforcement status

    Args:
        state: Current graph state with 'reranked_chunks' field.
        rag_generator: Optional RAGGenerator instance. If None,
            falls back to stub behavior.

    Returns:
        Dict with answer, citations, generation_model,
        enforcement status, and incremented step_count.
    """
    query = state.get("query", "")
    chunks = state.get("reranked_chunks", [])
    step_count = state.get("step_count", 0)

    if not chunks:
        return {
            "answer": "",
            "citations": [],
            "generation_model": "none",
            "error": "No context chunks available for generation.",
            "step_count": step_count + 1,
        }

    # If no generator provided, fall back to stub (Day 7 compat)
    if rag_generator is None:
        return _generate_stub(query, chunks, step_count)

    try:
        cited_answer, enforcement_passed, errors = rag_generator.generate(
            query=query,
            context_chunks=chunks,
        )

        # Convert Pydantic model to dict for graph state
        citations = [
            {
                "chunk_id": c.chunk_id,
                "filing_reference": c.filing_reference,
                "section": c.section,
                "text_excerpt": c.text_excerpt,
                "relevance_score": c.relevance_score,
            }
            for c in cited_answer.citations
        ]

        result: dict = {
            "answer": cited_answer.answer_text,
            "citations": citations,
            "generation_model": rag_generator._model_name,
            "step_count": step_count + 1,
        }

        if not enforcement_passed:
            result["error"] = (
                "Citation enforcement failed: "
                + "; ".join(errors)
            )

        logger.info(
            "generate_complete",
            query_preview=query[:80],
            confidence=cited_answer.confidence,
            citations=len(citations),
            enforcement_passed=enforcement_passed,
        )

        return result

    except Exception as e:
        logger.error("generate_failed", error=str(e))
        return {
            "answer": f"Generation error: {e}",
            "citations": [],
            "generation_model": "error",
            "error": f"Generation failed: {e}",
            "step_count": step_count + 1,
        }


def _generate_stub(query: str, chunks: list[dict], step_count: int) -> dict:
    """Stub generator for backward compatibility (no LLM).

    Used when RAGGenerator is not provided (e.g., in Day 7 tests
    or when GOOGLE_API_KEY is not set).

    Args:
        query: User's question.
        chunks: Reranked context chunks.
        step_count: Current step count.

    Returns:
        Dict with stub answer and citations.
    """
    context_texts = [c.get("text", "") for c in chunks[:3]]
    stub_answer = (
        f"[STUB — No LLM configured]\n\n"
        f"Query: {query}\n\n"
        f"Based on {len(chunks)} retrieved chunks:\n\n"
        + "\n---\n".join(context_texts[:3])
    )

    stub_citations = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "text_preview": c.get("text", "")[:100],
            "reranker_score": c.get("reranker_score", 0.0),
            "metadata": c.get("metadata", {}),
        }
        for c in chunks
    ]

    logger.info(
        "generate_complete_stub",
        query_preview=query[:80],
        context_chunks=len(chunks),
    )

    return {
        "answer": stub_answer,
        "citations": stub_citations,
        "generation_model": "stub_v1",
        "step_count": step_count + 1,
    }


# --------------------------------------------------------------------------- #
# Node: Calculate (LLM with function calling)
# --------------------------------------------------------------------------- #


def calculate(
    state: GraphState,
    *,
    rag_generator: RAGGenerator | None = None,
) -> dict:
    """Handle numerical computation queries via LLM.

    For now, delegates to the generate node with a calculation
    hint in the query. The LLM extracts numbers from context
    and performs the requested comparison/computation.

    Future enhancement: function calling tools for precise
    arithmetic (extract → compute → format).

    Args:
        state: Current graph state with 'query' and 'reranked_chunks'.
        rag_generator: Optional RAGGenerator instance.

    Returns:
        Dict with answer, citations, and step_count.
    """
    query = state.get("query", "")
    chunks = state.get("reranked_chunks", [])
    step_count = state.get("step_count", 0)

    if not chunks:
        return {
            "answer": "",
            "citations": [],
            "generation_model": "none",
            "error": "No context chunks available for calculation.",
            "step_count": step_count + 1,
        }

    # Enhance query with calculation instruction
    calc_query = (
        f"{query}\n\n"
        f"INSTRUCTION: Extract the relevant numbers from the context, "
        f"perform the requested calculation or comparison, and show "
        f"your work step by step. Cite the source of each number."
    )

    if rag_generator is None:
        return _generate_stub(calc_query, chunks, step_count)

    try:
        cited_answer, enforcement_passed, errors = rag_generator.generate(
            query=calc_query,
            context_chunks=chunks,
        )

        citations = [
            {
                "chunk_id": c.chunk_id,
                "filing_reference": c.filing_reference,
                "section": c.section,
                "text_excerpt": c.text_excerpt,
                "relevance_score": c.relevance_score,
            }
            for c in cited_answer.citations
        ]

        result: dict = {
            "answer": cited_answer.answer_text,
            "citations": citations,
            "generation_model": rag_generator._model_name,
            "step_count": step_count + 1,
        }

        if not enforcement_passed:
            result["error"] = (
                "Citation enforcement failed: "
                + "; ".join(errors)
            )

        logger.info(
            "calculate_complete",
            query_preview=query[:80],
            confidence=cited_answer.confidence,
            citations=len(citations),
        )

        return result

    except Exception as e:
        logger.error("calculate_failed", error=str(e))
        return {
            "answer": f"Calculation error: {e}",
            "citations": [],
            "generation_model": "error",
            "error": f"Calculation failed: {e}",
            "step_count": step_count + 1,
        }


# --------------------------------------------------------------------------- #
# Node: Validate
# --------------------------------------------------------------------------- #


def validate(state: GraphState) -> dict:
    """Validate the generated answer for quality and completeness.

    Checks:
    1. Answer is non-empty
    2. Citations are present (unless stub/decline)
    3. No critical error state from upstream nodes
    4. Step count hasn't exceeded max (loop guard)

    Args:
        state: Current graph state with answer and citations.

    Returns:
        Dict with is_valid, validation_errors, and step_count.
    """
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    error = state.get("error", "")
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", DEFAULT_MAX_STEPS)
    gen_model = state.get("generation_model", "")

    errors: list[str] = []

    # Check step count (loop guard)
    if step_count >= max_steps:
        errors.append(f"Max steps exceeded ({step_count}/{max_steps})")

    # Check for upstream errors (skip soft enforcement errors)
    if error and "enforcement" not in error.lower():
        errors.append(f"Upstream error: {error}")

    # Check answer presence
    if not answer.strip():
        errors.append("Empty answer generated")

    # Check citations for real (non-stub, non-decline) answers
    is_stub = "[STUB" in answer or gen_model in ("stub_v1", "decline", "error_handler")
    if answer and not is_stub and not citations:
        errors.append("No citations provided")

    is_valid = len(errors) == 0

    logger.info(
        "validation_complete",
        is_valid=is_valid,
        errors=errors,
        step_count=step_count,
        generation_model=gen_model,
    )

    return {
        "is_valid": is_valid,
        "validation_errors": errors,
        "step_count": step_count + 1,
    }


# --------------------------------------------------------------------------- #
# Node: Decline
# --------------------------------------------------------------------------- #


def decline(state: GraphState) -> dict:
    """Produce a polite refusal for out-of-scope queries.

    Generates a clear, helpful message explaining why the query
    can't be answered and what the system CAN do instead.

    Args:
        state: Current graph state with 'query' field.

    Returns:
        Dict with decline answer, empty citations, and step_count.
    """
    query = state.get("query", "")
    step_count = state.get("step_count", 0)

    decline_answer = (
        "I'm a financial research assistant that analyzes SEC filings "
        "and earnings call transcripts. I can help you find specific "
        "data, compare financial metrics across periods, and extract "
        "information from regulatory filings.\n\n"
        "However, I cannot provide investment advice, stock recommendations, "
        "or market predictions. For those, please consult a licensed "
        "financial advisor.\n\n"
        "Try asking questions like:\n"
        "- \"What was Apple's revenue in fiscal year 2024?\"\n"
        "- \"What risk factors did Tesla disclose in their latest 10-K?\"\n"
        "- \"How did Microsoft's cloud revenue change year over year?\""
    )

    logger.info("query_declined", query_preview=query[:80])

    return {
        "answer": decline_answer,
        "citations": [],
        "generation_model": "decline",
        "is_valid": True,
        "validation_errors": [],
        "step_count": step_count + 1,
    }


# --------------------------------------------------------------------------- #
# Node: Handle Error
# --------------------------------------------------------------------------- #


def handle_error(state: GraphState) -> dict:
    """Produce a user-friendly error response.

    Called when the pipeline encounters an unrecoverable error.
    Logs the error for debugging and returns a helpful message.

    Args:
        state: Current graph state with 'error' field.

    Returns:
        Dict with error answer and step_count.
    """
    error = state.get("error", "Unknown error")
    query = state.get("query", "")
    step_count = state.get("step_count", 0)

    error_answer = (
        "I encountered an issue while processing your question. "
        "Please try rephrasing your query or try again later.\n\n"
        f"Error details: {error}"
    )

    logger.error(
        "pipeline_error",
        error=error,
        query_preview=query[:80],
        step_count=step_count,
    )

    return {
        "answer": error_answer,
        "citations": [],
        "generation_model": "error_handler",
        "is_valid": False,
        "validation_errors": [error],
        "step_count": step_count + 1,
    }
