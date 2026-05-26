"""Query router for the FinRAG orchestration pipeline.

Classifies incoming queries into one of three routes:
- retrieve: standard RAG pipeline (retrieve → rerank → generate)
- calculate: numerical/comparison queries needing computation
- decline: out-of-scope queries (investment advice, predictions)

Design decisions:
- Keyword-based routing as Day 7 baseline. Fast, deterministic,
  zero LLM cost. LLM-based intent classification comes on Day 10
  when we have prompt configs and can version the classifier.
- Conservative decline: only decline on strong signals (investment
  advice keywords). False declines are worse than false retrieves.
- Calculate route catches arithmetic patterns: "compare", "higher",
  "difference", percentage questions. These need post-retrieval
  computation, not just text generation.
- Confidence scores are heuristic (1.0 for strong matches, 0.7 for
  weak). These become meaningful when the LLM classifier drops in.

Debt: DAY-7-001 — Router is keyword-based. LLM-based intent
      classification will improve accuracy. Resolve on Day 10.
"""

import re

import structlog

from finrag.orchestration.state import GraphState

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Route Classification Patterns
# --------------------------------------------------------------------------- #

# Patterns that indicate out-of-scope queries. These are financial
# advice, predictions, and recommendations — territory we refuse
# to enter because we're a research assistant, not a financial advisor.
DECLINE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(should\s+i\s+(buy|sell|invest|hold))\b", re.IGNORECASE),
    re.compile(r"\b(recommend|suggestion|advice)\b", re.IGNORECASE),
    re.compile(r"\b(will\s+(the\s+)?(stock|price|market)\s+(price\s+)?(go|rise|fall|crash|drop))\b", re.IGNORECASE),
    re.compile(r"\b(predict|forecast|projection)\s+(stock|price|revenue|earnings)\b", re.IGNORECASE),
    re.compile(r"\b(best\s+(stock|investment|etf|fund))\b", re.IGNORECASE),
    re.compile(r"\b(investment\s+advice)\b", re.IGNORECASE),
    re.compile(r"\b(portfolio\s+(allocation|recommendation))\b", re.IGNORECASE),
]

# Patterns indicating the user wants a structured overview / summary
# of the entire filing, not a specific factual answer.
# These get broader retrieval (more chunks, all sections) and a
# summary-style generation prompt.
SUMMARIZE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(summarize|summarise|summary|overview|brief|highlights?)\b", re.IGNORECASE),
    re.compile(r"\b(key\s+(points|takeaways|findings|metrics|numbers|facts))\b", re.IGNORECASE),
    re.compile(r"\b(what\s+(is|was|are|were)\s+.{0,20}\s+(about|cover|discuss|contain))\b", re.IGNORECASE),
    re.compile(r"\b(overall\s+(performance|result|financials?|business))\b", re.IGNORECASE),
    re.compile(r"\b(tell\s+me\s+(about|everything|all))\b", re.IGNORECASE),
    re.compile(r"\b(main\s+(point|highlight|topic|finding|result))\b", re.IGNORECASE),
]

# Patterns indicating the query needs numerical computation after
# retrieval, not just text generation. These queries require extracting
# numbers from chunks and performing calculations or comparisons.
# NOTE: Keep this list tight — false positives send factual questions
# through the calculate path which has stricter citation enforcement.
CALCULATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(compare|comparison|versus|vs\.?)\b", re.IGNORECASE),
    re.compile(r"\b(higher|lower|greater|less|more|fewer)\s+than\b", re.IGNORECASE),
    re.compile(r"\b(increase|decrease|change|differ)\s+(by|in|from)\b", re.IGNORECASE),
    re.compile(r"\b(year.over.year|yoy|quarter.over.quarter|qoq)\b", re.IGNORECASE),
    re.compile(r"\b(growth\s+rate|cagr|margin\s+change)\b", re.IGNORECASE),
    re.compile(r"\b(how\s+much\s+did\s+.{0,30}\s+(change|grow|decline|increase|decrease))\b", re.IGNORECASE),
    re.compile(r"\b(percent(age)?|proportion|fraction)\b", re.IGNORECASE),
]


# --------------------------------------------------------------------------- #
# Router Node
# --------------------------------------------------------------------------- #


def route_query(state: GraphState) -> dict:
    """Classify query intent and determine the pipeline route.

    Examines the query text against keyword patterns to decide
    which pipeline branch to execute:

    - decline: Query asks for financial advice or predictions.
      We refuse these — FinRAG is a research tool, not an advisor.
    - calculate: Query involves numerical comparison or computation.
      Needs retrieve → compute → generate path.
    - retrieve: Default. Standard RAG: retrieve → rerank → generate.

    Priority order: decline > calculate > retrieve.
    Decline is checked first because safety overrides utility.

    Args:
        state: Current graph state with 'query' field.

    Returns:
        Dict with route, route_confidence, query_intent, and
        incremented step_count.
    """
    query = state.get("query", "")
    step_count = state.get("step_count", 0)

    if not query.strip():
        logger.warning("empty_query_received")
        return {
            "route": "decline",
            "route_confidence": 1.0,
            "query_intent": "empty",
            "step_count": step_count + 1,
        }

    # Check decline patterns first (safety > utility)
    for pattern in DECLINE_PATTERNS:
        if pattern.search(query):
            logger.info(
                "query_declined",
                query_preview=query[:80],
                matched_pattern=pattern.pattern,
            )
            return {
                "route": "decline",
                "route_confidence": 0.9,
                "query_intent": "financial_advice",
                "step_count": step_count + 1,
            }

    # Check summarize patterns — detected before calculate so broad
    # queries like "summarize and compare" go to summarize path
    for pattern in SUMMARIZE_PATTERNS:
        if pattern.search(query):
            logger.info("query_routed_summarize", query_preview=query[:80])
            return {
                "route": "retrieve",          # Same graph path — no new nodes needed
                "route_confidence": 0.85,
                "query_intent": "summarize",  # Signal to retrieve node to go broad
                "step_count": step_count + 1,
            }

    # Check calculate patterns
    for pattern in CALCULATE_PATTERNS:
        if pattern.search(query):
            logger.info(
                "query_routed_calculate",
                query_preview=query[:80],
                matched_pattern=pattern.pattern,
            )
            return {
                "route": "calculate",
                "route_confidence": 0.8,
                "query_intent": "numerical_comparison",
                "step_count": step_count + 1,
            }

    # Default: standard retrieval
    logger.info("query_routed_retrieve", query_preview=query[:80])
    return {
        "route": "retrieve",
        "route_confidence": 0.7,
        "query_intent": "factual_extraction",
        "step_count": step_count + 1,
    }


def get_route(state: GraphState) -> str:
    """Edge routing function for LangGraph conditional edges.

    Called by LangGraph to determine which node to execute next
    after the router node. Returns the route string that maps
    to the next node in the graph.

    Args:
        state: Current graph state with 'route' field set.

    Returns:
        Route string: "retrieve", "calculate", or "decline".
    """
    return state.get("route", "retrieve")
