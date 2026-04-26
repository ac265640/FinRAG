"""Citation enforcer for the FinRAG pipeline.

Validates that every citation in a generated answer maps to a real
retrieved chunk. Catches hallucinated citations, missing citations,
and low-confidence answers.

The enforcer is a post-generation validation layer, not a generation
constraint. It runs AFTER the LLM produces an answer and BEFORE the
answer reaches the user. This separation is intentional:
- Generation: LLM does its best to cite correctly
- Enforcement: system verifies and rejects bad citations

Design decisions:
- Whitelist validation: only chunk_ids present in the retrieved
  context are valid. Everything else is a hallucinated citation.
- Minimum citation count: answers must have at least 1 citation.
  Uncited answers are rejected because they can't be verified.
- Confidence threshold: below 0.3, the system declines rather
  than risk a low-quality answer. Tunable per-deployment.
- Relevance floor: top reranker score must exceed a minimum
  (default 0.2). If the best chunk is barely relevant, the
  answer is likely to be unreliable.

Debt: DAY-8-001 — Citation coverage is coarse (presence-only).
      A fine-grained check mapping individual sentences to citations
      would catch partial hallucinations. Resolve on Day 14.
"""

import structlog

from finrag.orchestration.schemas import CitedAnswer

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Minimum confidence score to accept an answer.
# Below this, the system declines to answer.
DEFAULT_CONFIDENCE_THRESHOLD = 0.3

# Minimum number of citations required.
MIN_CITATIONS = 1

# Minimum reranker score for the top chunk.
# If the best chunk scores below this, context quality is too poor.
DEFAULT_RELEVANCE_FLOOR = 0.2


# --------------------------------------------------------------------------- #
# Enforcement Result
# --------------------------------------------------------------------------- #


class EnforcementResult:
    """Result of citation enforcement checks.

    Attributes:
        is_valid: Whether all checks passed.
        errors: List of enforcement failure descriptions.
        hallucinated_ids: Chunk IDs cited but not in context.
        valid_citation_count: Number of valid citations.
    """

    def __init__(self) -> None:
        """Initialize with no errors."""
        self.is_valid: bool = True
        self.errors: list[str] = []
        self.hallucinated_ids: list[str] = []
        self.valid_citation_count: int = 0

    def add_error(self, error: str) -> None:
        """Record an enforcement failure.

        Args:
            error: Description of the failure.
        """
        self.is_valid = False
        self.errors.append(error)


# --------------------------------------------------------------------------- #
# Citation Enforcer
# --------------------------------------------------------------------------- #


class CitationEnforcer:
    """Validates citations in generated answers.

    Ensures every cited chunk_id exists in the retrieved context,
    the answer meets minimum citation and confidence requirements,
    and the top relevance score exceeds the quality floor.

    Args:
        confidence_threshold: Minimum confidence to accept (0-1).
        min_citations: Minimum required citations.
        relevance_floor: Minimum top reranker score.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        min_citations: int = MIN_CITATIONS,
        relevance_floor: float = DEFAULT_RELEVANCE_FLOOR,
    ) -> None:
        """Initialize the citation enforcer.

        Args:
            confidence_threshold: Min confidence score (default 0.3).
            min_citations: Min citation count (default 1).
            relevance_floor: Min top reranker score (default 0.2).
        """
        self._confidence_threshold = confidence_threshold
        self._min_citations = min_citations
        self._relevance_floor = relevance_floor

        logger.info(
            "citation_enforcer_initialized",
            confidence_threshold=confidence_threshold,
            min_citations=min_citations,
            relevance_floor=relevance_floor,
        )

    def enforce(
        self,
        answer: CitedAnswer,
        context_chunks: list[dict],
    ) -> EnforcementResult:
        """Run all citation enforcement checks.

        Checks performed (in order):
        1. Confidence threshold: is the model confident enough?
        2. Relevance floor: is the best chunk relevant enough?
        3. Citation count: are there enough citations?
        4. Citation validity: do all chunk_ids exist in context?

        Args:
            answer: The structured answer from the LLM.
            context_chunks: The reranked chunks that were provided
                as context to the LLM. Each must have 'chunk_id'.

        Returns:
            EnforcementResult with validity status and errors.
        """
        result = EnforcementResult()

        # Build whitelist of valid chunk_ids from context
        valid_ids = {c["chunk_id"] for c in context_chunks}

        # Check 1: Confidence threshold
        if answer.confidence < self._confidence_threshold:
            result.add_error(
                f"Confidence {answer.confidence:.2f} below threshold "
                f"{self._confidence_threshold:.2f}"
            )

        # Check 2: Relevance floor (top reranker score)
        if context_chunks:
            top_score = max(
                c.get("reranker_score", 0.0) for c in context_chunks
            )
            if top_score < self._relevance_floor:
                result.add_error(
                    f"Top relevance score {top_score:.2f} below floor "
                    f"{self._relevance_floor:.2f}"
                )

        # Check 3: Minimum citation count
        if len(answer.citations) < self._min_citations:
            result.add_error(
                f"Only {len(answer.citations)} citations, "
                f"minimum required: {self._min_citations}"
            )

        # Check 4: Citation validity (hallucination detection)
        valid_count = 0
        for citation in answer.citations:
            if citation.chunk_id in valid_ids:
                valid_count += 1
            else:
                result.hallucinated_ids.append(citation.chunk_id)
                result.add_error(
                    f"Hallucinated citation: chunk_id '{citation.chunk_id}' "
                    f"not found in retrieved context"
                )

        result.valid_citation_count = valid_count

        logger.info(
            "citation_enforcement_complete",
            is_valid=result.is_valid,
            total_citations=len(answer.citations),
            valid_citations=valid_count,
            hallucinated=len(result.hallucinated_ids),
            confidence=answer.confidence,
            errors=result.errors,
        )

        return result

    def should_decline(
        self,
        context_chunks: list[dict],
    ) -> tuple[bool, str]:
        """Pre-generation check: should we decline before even calling the LLM?

        If the reranked context is too poor, there's no point generating
        an answer. This saves an LLM call and avoids hallucination.

        Args:
            context_chunks: Reranked chunks with reranker_score.

        Returns:
            Tuple of (should_decline, reason).
        """
        if not context_chunks:
            return True, "No context chunks available after retrieval."

        top_score = max(
            c.get("reranker_score", 0.0) for c in context_chunks
        )

        if top_score < self._relevance_floor:
            return True, (
                f"Best relevance score ({top_score:.2f}) is below "
                f"the quality threshold ({self._relevance_floor:.2f}). "
                f"Insufficient evidence to answer reliably."
            )

        return False, ""
