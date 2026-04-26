"""Guardrails pipeline integration for the LangGraph state machine.

Provides LangGraph-compatible node functions that wrap the input
and output guard systems. These nodes fit into the existing graph
as pre-processing (before routing) and post-processing (after
generation, before delivery).

Updated graph structure with guardrails:

    START → input_guard → route_query → retrieve → rerank → generate → output_guard → validate → END
                 │                                                          │
                 └→ (blocked) → END                                         └→ (blocked) → END

Design decisions:
- Input guard as first node: catches bad queries before any compute.
  A blocked query never hits retrieval, reranking, or the LLM.
- Output guard after generation: scrubs PII, adds disclaimers,
  checks for advice language before the answer leaves the system.
- Guard results stored in state: downstream nodes (validate) can
  inspect guard warnings. Langfuse (Day 12) will trace guard
  activity per request.
- Blocked responses are polite: they explain what happened and
  suggest alternatives, just like the decline node.
"""

import structlog

from finrag.guardrails.input_guard import GuardReport, run_input_guards
from finrag.guardrails.output_guard import OutputGuardReport, run_output_guards
from finrag.orchestration.state import GraphState

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Node: Input Guard
# --------------------------------------------------------------------------- #


def guard_input(state: GraphState) -> dict:
    """LangGraph node: run input guards on the user's query.

    If any guard returns BLOCK severity, sets a blocked flag
    and a user-friendly error message. The graph's conditional
    edge checks this flag to skip directly to END.

    Args:
        state: Current graph state with 'query' field.

    Returns:
        Dict with guard results and optional block status.
    """
    query = state.get("query", "")
    step_count = state.get("step_count", 0)

    report = run_input_guards(query)

    result: dict = {
        "step_count": step_count + 1,
    }

    if not report.allowed:
        # Query blocked — set answer directly
        result["answer"] = (
            "Your query was blocked by our safety system.\n\n"
            f"Reason: {report.results[-1].message}\n\n"
            "Please rephrase your question. I can help with:\n"
            "- Factual data from SEC filings (revenue, expenses, risks)\n"
            "- Financial metric comparisons across periods\n"
            "- Information extraction from 10-K and 10-Q filings"
        )
        result["citations"] = []
        result["generation_model"] = "input_guard"
        result["is_valid"] = True
        result["validation_errors"] = []
        result["input_guard_blocked"] = True
        result["input_guard_reason"] = report.results[-1].message

        logger.warning(
            "query_blocked_by_guard",
            guard=report.blocked_by,
            reason=report.results[-1].message,
        )
    else:
        result["input_guard_blocked"] = False
        if report.warnings:
            result["input_guard_warnings"] = report.warnings
            logger.info(
                "input_guard_warnings",
                warnings=report.warnings,
            )

    return result


# --------------------------------------------------------------------------- #
# Node: Output Guard
# --------------------------------------------------------------------------- #


def guard_output(state: GraphState) -> dict:
    """LangGraph node: run output guards on the generated answer.

    Scrubs PII, checks for advice language, and adds financial
    disclaimers. Modifies the answer in-place (via state update).

    Args:
        state: Current graph state with 'answer' field.

    Returns:
        Dict with scrubbed answer and guard metadata.
    """
    answer = state.get("answer", "")
    gen_model = state.get("generation_model", "")
    step_count = state.get("step_count", 0)

    # Skip for empty/decline/error answers
    if not answer or gen_model in ("decline", "error_handler", "input_guard"):
        return {"step_count": step_count + 1}

    report = run_output_guards(answer, gen_model)

    result: dict = {
        "step_count": step_count + 1,
    }

    if not report.allowed:
        # Output blocked (misleading financial promises)
        result["answer"] = report.scrubbed_answer
        result["citations"] = []
        result["generation_model"] = "output_guard_blocked"
        result["output_guard_blocked"] = True

        logger.warning(
            "output_blocked_by_guard",
            answer_preview=answer[:80],
        )
    else:
        # Apply scrubbed answer (may have PII removed, disclaimer added)
        result["answer"] = report.scrubbed_answer
        result["output_guard_blocked"] = False

        if report.redactions_made > 0:
            result["output_guard_redactions"] = report.redactions_made
        if report.disclaimer_added:
            result["output_guard_disclaimer"] = True
        if report.warnings:
            result["output_guard_warnings"] = report.warnings

    return result


# --------------------------------------------------------------------------- #
# Edge routing helpers
# --------------------------------------------------------------------------- #


def is_input_blocked(state: GraphState) -> str:
    """Conditional edge: check if input guard blocked the query.

    Used by add_conditional_edges after the input_guard node
    to decide whether to proceed to routing or go to END.

    Args:
        state: Current graph state.

    Returns:
        "blocked" if query was blocked, "allowed" otherwise.
    """
    if state.get("input_guard_blocked", False):
        return "blocked"
    return "allowed"
