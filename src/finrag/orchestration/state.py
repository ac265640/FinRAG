"""Graph state definition for the FinRAG orchestration pipeline.

Defines the TypedDict that flows through every node in the LangGraph
state machine. Every field is explicitly typed and documented.

Design decisions:
- TypedDict over Pydantic: LangGraph natively supports TypedDict for
  state. Pydantic adds overhead and is unnecessary here — we validate
  at boundaries (API layer), not between trusted internal nodes.
- Flat state over nested: Every node reads/writes to a flat namespace.
  Avoids deep attribute access and makes node contracts obvious.
- All fields Optional or have defaults: A node should never crash
  because a previous node didn't run (e.g., reranker skipped).
- Operator annotations for list fields: LangGraph requires explicit
  reducers for list fields to support parallel node execution.
"""

from typing import Annotated, TypedDict

from langgraph.graph import add_messages


# --------------------------------------------------------------------------- #
# Query Route Enum
# --------------------------------------------------------------------------- #

# Using string literals instead of enum for JSON serialization simplicity.
# Valid route values:
#   "retrieve"    — standard RAG: retrieve → rerank → generate
#   "calculate"   — arithmetic/comparison: retrieve → compute → generate
#   "decline"     — out-of-scope: refuse politely, no retrieval
VALID_ROUTES = {"retrieve", "calculate", "decline"}


# --------------------------------------------------------------------------- #
# Graph State
# --------------------------------------------------------------------------- #


class GraphState(TypedDict, total=False):
    """State flowing through the LangGraph RAG pipeline.

    Every node reads from and writes to this shared state dict.
    Fields are grouped by pipeline stage for clarity.

    Input fields (set by the API/caller):
        query: The user's natural language question.
        conversation_history: Prior messages for multi-turn context.
        metadata_filter: Optional metadata filter for retrieval.

    Router fields (set by route_query node):
        route: The chosen pipeline route.
        route_confidence: Router's confidence in the route decision.
        query_intent: Classified intent type.

    Retrieval fields (set by retrieve node):
        retrieved_chunks: Raw chunks from hybrid retrieval.
        retrieval_metadata: Stats about the retrieval run.

    Reranker fields (set by rerank node):
        reranked_chunks: Chunks reordered by cross-encoder.

    Generation fields (set by generate node):
        answer: The generated answer text.
        citations: List of citation dicts.
        generation_model: Which LLM generated the answer.

    Validation fields (set by validate node):
        is_valid: Whether the answer passed validation.
        validation_errors: List of validation failure reasons.

    Control fields:
        error: Error message if a node fails.
        step_count: Running count of nodes executed (loop guard).
        max_steps: Maximum allowed steps before forced termination.

    Message fields (LangGraph built-in):
        messages: Chat message history (uses add_messages reducer).
    """

    # --- Input ---
    query: str
    conversation_history: list[dict]
    metadata_filter: dict | None

    # --- Router ---
    route: str
    route_confidence: float
    query_intent: str

    # --- Retrieval ---
    retrieved_chunks: list[dict]
    retrieval_metadata: dict

    # --- Reranker ---
    reranked_chunks: list[dict]

    # --- Generation ---
    answer: str
    citations: list[dict]
    generation_model: str

    # --- Validation ---
    is_valid: bool
    validation_errors: list[str]

    # --- Control ---
    error: str
    step_count: int
    max_steps: int

    # --- Messages (LangGraph reducer) ---
    messages: Annotated[list, add_messages]
