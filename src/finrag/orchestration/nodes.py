"""Pipeline node functions for the FinRAG LangGraph state machine.

Each function is a graph node that reads from and writes to the
shared GraphState. Nodes are pure functions of state — no hidden
dependencies, no side effects beyond logging.

Node responsibilities:
- retrieve: Run hybrid retrieval (BM25 + vector + RRF)
- rerank: Apply cross-encoder reranking to retrieved chunks
- generate: Generate answer from reranked context (stub Day 7)
- calculate: Extract numbers and compute comparisons (stub Day 7)
- validate: Check answer quality and citation presence
- decline: Produce a polite refusal for out-of-scope queries
- handle_error: Graceful error handling for pipeline failures

Design decisions:
- Each node catches its own exceptions and writes to state['error']
  rather than crashing the graph. This lets the validate node
  produce a user-friendly error message.
- generate and calculate are stubs until Day 8 (LLM integration).
  They return placeholder text so the graph is fully testable now.
- Nodes increment step_count for the loop guard.
- Max steps default is 10 — enough for retrieve → rerank → generate
  with one retry, but catches infinite loops.

Debt: DAY-7-002 — generate node is a stub. Needs LLM integration
      (Day 8) with citation enforcement.
Debt: DAY-7-003 — calculate node is a stub. Needs function calling
      for arithmetic operations (Day 8).
"""

import structlog

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
# Node: Generate (Stub — LLM integration Day 8)
# --------------------------------------------------------------------------- #


def generate(state: GraphState) -> dict:
    """Generate an answer from reranked context chunks.

    Day 7 stub. Returns a formatted context summary as a placeholder.
    Day 8 will replace this with LLM generation + citation enforcement.

    The real implementation will:
    1. Build a system prompt with citation instructions
    2. Include reranked chunks as context
    3. Call the LLM with structured output (Pydantic schema)
    4. Extract answer text and citation list
    5. Pass to validate node for quality checks

    Args:
        state: Current graph state with 'reranked_chunks' field.

    Returns:
        Dict with answer, citations, generation_model, and
        incremented step_count.
    """
    query = state.get("query", "")
    chunks = state.get("reranked_chunks", [])
    step_count = state.get("step_count", 0)

    if not chunks:
        return {
            "answer": "",
            "citations": [],
            "generation_model": "stub_v1",
            "error": "No context chunks available for generation.",
            "step_count": step_count + 1,
        }

    # Stub: concatenate top chunk texts as "answer"
    context_texts = [c.get("text", "") for c in chunks[:3]]
    stub_answer = (
        f"[STUB — Day 8 will add LLM generation]\n\n"
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
        citations=len(stub_citations),
    )

    return {
        "answer": stub_answer,
        "citations": stub_citations,
        "generation_model": "stub_v1",
        "step_count": step_count + 1,
    }


# --------------------------------------------------------------------------- #
# Node: Calculate (Stub — Function calling Day 8)
# --------------------------------------------------------------------------- #


def calculate(state: GraphState) -> dict:
    """Handle numerical computation queries.

    Day 7 stub. The real implementation (Day 8) will:
    1. Retrieve relevant chunks containing numbers
    2. Extract numerical values using regex or LLM function calling
    3. Perform the requested calculation (compare, diff, ratio)
    4. Format the result with source citations

    For now, routes through the standard retrieve → generate path
    with a note that computation is needed.

    Args:
        state: Current graph state with 'query' field.

    Returns:
        Dict with answer indicating calculation is needed,
        and incremented step_count.
    """
    query = state.get("query", "")
    chunks = state.get("reranked_chunks", [])
    step_count = state.get("step_count", 0)

    stub_answer = (
        f"[STUB — Day 8 will add calculation support]\n\n"
        f"Query: {query}\n\n"
        f"This query requires numerical computation. "
        f"Available context: {len(chunks)} chunks."
    )

    logger.info(
        "calculate_stub",
        query_preview=query[:80],
        context_chunks=len(chunks),
    )

    return {
        "answer": stub_answer,
        "citations": [],
        "generation_model": "calculate_stub_v1",
        "step_count": step_count + 1,
    }


# --------------------------------------------------------------------------- #
# Node: Validate
# --------------------------------------------------------------------------- #


def validate(state: GraphState) -> dict:
    """Validate the generated answer for quality and completeness.

    Checks:
    1. Answer is non-empty
    2. Citations are present (no uncited claims)
    3. No error state from upstream nodes
    4. Step count hasn't exceeded max (loop guard)

    Day 8 will add:
    - Citation coverage check (every claim backed by a chunk)
    - Confidence threshold check
    - Retry logic for low-quality answers

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

    errors: list[str] = []

    # Check step count (loop guard)
    if step_count >= max_steps:
        errors.append(f"Max steps exceeded ({step_count}/{max_steps})")

    # Check for upstream errors
    if error:
        errors.append(f"Upstream error: {error}")

    # Check answer presence
    if not answer.strip():
        errors.append("Empty answer generated")

    # Check citations (skip for stub/decline)
    if answer and "[STUB" not in answer and not citations:
        errors.append("No citations provided")

    is_valid = len(errors) == 0

    logger.info(
        "validation_complete",
        is_valid=is_valid,
        errors=errors,
        step_count=step_count,
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
