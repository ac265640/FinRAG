"""LangGraph state machine for the FinRAG RAG pipeline.

Assembles individual nodes into a compiled graph with conditional
routing. The graph structure is:

    START → route_query ──→ retrieve → rerank → generate → validate → END
                        │                              ↑
                        ├→ calculate ─────────────────→┘
                        │                              (via retrieve → rerank)
                        └→ decline ──────────────────────→ END

Design decisions:
- Functional node wrappers: LangGraph nodes are plain callables.
  We wrap the nodes from nodes.py with their dependencies injected
  at graph construction time (partial application). This keeps
  node functions pure and testable independently.
- Conditional edges from router: LangGraph's add_conditional_edges
  routes based on the 'route' field set by route_query.
- Calculate shares retrieve → rerank path: calculation queries still
  need context from filings. They go through retrieval first, then
  the calculate node extracts numbers.
- Single compiled graph instance: compile() is expensive (~50ms).
  Build once, invoke many times.
- Step count guard in validate: prevents infinite loops if the graph
  has cycles (future retry logic).
"""

from functools import partial

import structlog
from langgraph.graph import END, StateGraph

from finrag.orchestration.nodes import (
    calculate,
    decline,
    generate,
    handle_error,
    rerank,
    retrieve,
    validate,
)
from finrag.orchestration.router import get_route, route_query
from finrag.orchestration.state import GraphState
from finrag.retrieval.hybrid import HybridRetriever
from finrag.retrieval.reranker import CrossEncoderReranker

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Graph Builder
# --------------------------------------------------------------------------- #


def build_rag_graph(
    hybrid_retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> StateGraph:
    """Build the RAG pipeline as a LangGraph state machine.

    Constructs the full graph with nodes, edges, and conditional
    routing. Returns an uncompiled StateGraph that can be compiled
    with .compile() for execution.

    Args:
        hybrid_retriever: Initialized HybridRetriever for the
            retrieve node.
        reranker: Initialized CrossEncoderReranker for the
            rerank node.

    Returns:
        Uncompiled StateGraph ready for .compile().
    """
    graph = StateGraph(GraphState)

    # --- Bind dependencies to nodes via partial application ---
    # This keeps node functions pure (testable without graph)
    # while providing runtime dependencies.
    retrieve_node = partial(retrieve, hybrid_retriever=hybrid_retriever)
    rerank_node = partial(rerank, reranker=reranker)

    # --- Add nodes ---
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate)
    graph.add_node("calculate", calculate)
    graph.add_node("validate", validate)
    graph.add_node("decline", decline)
    graph.add_node("handle_error", handle_error)

    # --- Set entry point ---
    graph.set_entry_point("route_query")

    # --- Conditional routing from router ---
    # After route_query, branch based on the 'route' field.
    graph.add_conditional_edges(
        "route_query",
        get_route,
        {
            "retrieve": "retrieve",
            "calculate": "retrieve",  # Calculate also needs retrieval first
            "decline": "decline",
        },
    )

    # --- Linear edges for retrieve pipeline ---
    graph.add_edge("retrieve", "rerank")

    # --- Conditional edge after rerank: generate vs calculate ---
    # If the route was "calculate", go to calculate node.
    # Otherwise, go to standard generate.
    graph.add_conditional_edges(
        "rerank",
        lambda state: state.get("route", "retrieve"),
        {
            "retrieve": "generate",
            "calculate": "calculate",
        },
    )

    # --- Post-generation: validate ---
    graph.add_edge("generate", "validate")
    graph.add_edge("calculate", "validate")

    # --- Terminal edges ---
    graph.add_edge("validate", END)
    graph.add_edge("decline", END)
    graph.add_edge("handle_error", END)

    logger.info("rag_graph_built", nodes=8, has_conditional_routing=True)

    return graph


def compile_rag_graph(
    hybrid_retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> object:
    """Build and compile the RAG graph for execution.

    Compilation produces an optimized, executable graph.
    This is the main entry point for creating a runnable pipeline.

    Args:
        hybrid_retriever: Initialized HybridRetriever.
        reranker: Initialized CrossEncoderReranker.

    Returns:
        Compiled LangGraph runnable.
    """
    graph = build_rag_graph(hybrid_retriever, reranker)
    compiled = graph.compile()

    logger.info("rag_graph_compiled")
    return compiled


def invoke_pipeline(
    compiled_graph: object,
    query: str,
    metadata_filter: dict | None = None,
    conversation_history: list[dict] | None = None,
    max_steps: int = 10,
) -> dict:
    """Invoke the compiled RAG pipeline with a query.

    Convenience function that sets up the initial state and
    invokes the compiled graph. Returns the final state.

    Args:
        compiled_graph: Compiled LangGraph from compile_rag_graph().
        query: User's natural language question.
        metadata_filter: Optional metadata filter for retrieval.
            Example: {"ticker": "AAPL"}
        conversation_history: Prior messages for multi-turn context.
        max_steps: Maximum allowed pipeline steps (loop guard).

    Returns:
        Final GraphState dict with answer, citations, and metadata.
    """
    initial_state: dict = {
        "query": query,
        "metadata_filter": metadata_filter,
        "conversation_history": conversation_history or [],
        "step_count": 0,
        "max_steps": max_steps,
        "messages": [],
    }

    logger.info(
        "pipeline_invoked",
        query_preview=query[:80],
        has_filter=metadata_filter is not None,
        max_steps=max_steps,
    )

    result = compiled_graph.invoke(initial_state)

    logger.info(
        "pipeline_complete",
        query_preview=query[:80],
        route=result.get("route", "unknown"),
        is_valid=result.get("is_valid", False),
        step_count=result.get("step_count", 0),
        has_answer=bool(result.get("answer", "")),
        num_citations=len(result.get("citations", [])),
    )

    return result
