"""LLM-powered answer generation with citation enforcement.

Replaces the Day 7 generate stub with actual LLM calls.
Uses Google Gemini via langchain-google-genai for generation,
with Pydantic structured output to guarantee the response
format matches the CitedAnswer schema.

Pipeline position:
    retrieve → rerank → [pre-decline check] → generate → enforce → validate

Design decisions:
- Google Gemini (gemini-2.0-flash) for fast, cost-effective generation.
  Flash model is sufficient for grounded extraction — creativity doesn't
  help when the answer must come from the provided context.
- System prompt is explicit about citation rules: every claim must
  cite a chunk_id from the provided context. No external knowledge.
- Context formatting: each chunk is wrapped with its chunk_id and
  metadata, so the LLM can reference them by ID.
- Retry once with stricter prompt if citation enforcement fails.
  Second attempt uses a "correction" prompt that includes the
  specific enforcement errors from the first attempt.
- Pre-generation decline: if context quality is too poor (below
  relevance floor), we skip the LLM call entirely to save cost.

Debt: DAY-8-002 — Using gemini-2.0-flash for all queries. Harder
      queries (multi-hop, contradiction) may need gemini-pro.
      Evaluate on Day 13 with golden dataset.
"""

import os

import structlog
from langchain_google_genai import ChatGoogleGenerativeAI

from finrag.orchestration.citation import CitationEnforcer
from finrag.orchestration.schemas import CitedAnswer, Citation, build_filing_reference

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.1  # Low temp for factual extraction
DEFAULT_MAX_RETRIES = 1  # One retry with stricter prompt

# System prompt for citation-grounded generation
SYSTEM_PROMPT = """You are a financial research assistant that answers questions
using ONLY the provided context from SEC filings and earnings call transcripts.

CRITICAL RULES:
1. ONLY use information from the provided context chunks. Never use external knowledge.
2. Every factual claim MUST cite the specific chunk_id it comes from.
3. If the context does not contain enough information to answer, say so explicitly.
4. Never provide investment advice, stock recommendations, or market predictions.
5. Be precise with numbers — quote them exactly as they appear in the source.

Your response must be a JSON object with these fields:
- answer_text: Your answer in clear, professional language.
- citations: List of objects, each with:
  - chunk_id: The exact chunk_id from the context that supports this claim.
  - filing_reference: Human-readable source (e.g., "AAPL 10-K FY2024, Item 7").
  - section: The section name from the filing.
  - text_excerpt: A short quote (max 200 chars) from the source.
  - relevance_score: How relevant this source is (0.0 to 1.0).
- confidence: Your confidence that the answer is correct and fully supported (0.0 to 1.0).
- reasoning: Brief explanation of how you derived the answer from the sources."""

# Stricter prompt for retry after citation failure
RETRY_PROMPT_SUFFIX = """

IMPORTANT CORRECTION: Your previous answer had citation errors:
{errors}

Please fix these issues:
- Only cite chunk_ids that exist in the provided context.
- Ensure every factual claim has a supporting citation.
- If you cannot find supporting evidence, lower your confidence score.
- Do NOT fabricate or hallucinate chunk_ids."""


# --------------------------------------------------------------------------- #
# Context Formatter
# --------------------------------------------------------------------------- #


def format_context_for_llm(chunks: list[dict]) -> str:
    """Format reranked chunks as numbered context for the LLM.

    Each chunk is wrapped with its chunk_id and metadata so the
    LLM can reference it by ID in its citations.

    Args:
        chunks: Reranked chunks with chunk_id, text, and metadata.

    Returns:
        Formatted context string ready for the LLM prompt.
    """
    parts: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        score = chunk.get("reranker_score", 0.0)
        filing_ref = build_filing_reference(metadata)

        header = f"[CHUNK {i}] chunk_id: {chunk_id}"
        if filing_ref:
            header += f" | source: {filing_ref}"
        header += f" | relevance: {score:.2f}"

        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class RAGGenerator:
    """LLM-powered answer generator with citation enforcement.

    Generates answers using Google Gemini, validates citations
    against the retrieved context, and retries once with a
    stricter prompt if enforcement fails.

    Args:
        model_name: Gemini model name (default gemini-2.0-flash).
        temperature: Generation temperature (default 0.1).
        max_retries: Max retry attempts on enforcement failure.
        citation_enforcer: Optional custom CitationEnforcer instance.
        api_key: Optional Google API key (falls back to env var).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        citation_enforcer: CitationEnforcer | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the RAG generator.

        Args:
            model_name: Gemini model identifier.
            temperature: Sampling temperature (0-1).
            max_retries: Retry count on citation failure.
            citation_enforcer: Custom enforcer (default creates one).
            api_key: Google API key (default: GOOGLE_API_KEY env var).
        """
        self._model_name = model_name
        self._temperature = temperature
        self._max_retries = max_retries
        self._enforcer = citation_enforcer or CitationEnforcer()

        # Resolve API key
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

        # Lazy-init the LLM (defer to first call to allow testing without key)
        self._api_key = resolved_key
        self._llm: ChatGoogleGenerativeAI | None = None

        logger.info(
            "generator_initialized",
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
            has_api_key=bool(resolved_key),
        )

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Lazy-load the LLM client.

        Returns:
            Initialized ChatGoogleGenerativeAI instance.

        Raises:
            ValueError: If no API key is configured.
        """
        if self._llm is None:
            if not self._api_key:
                msg = (
                    "No Google API key configured. Set GOOGLE_API_KEY "
                    "environment variable or pass api_key to RAGGenerator."
                )
                raise ValueError(msg)

            self._llm = ChatGoogleGenerativeAI(
                model=self._model_name,
                temperature=self._temperature,
                google_api_key=self._api_key,
            )
            logger.info("llm_client_initialized", model=self._model_name)
        return self._llm

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
    ) -> tuple[CitedAnswer, bool, list[str]]:
        """Generate a cited answer from context chunks.

        Workflow:
        1. Pre-decline check: skip LLM if context quality too poor
        2. Format context and call LLM with structured output
        3. Parse response into CitedAnswer
        4. Run citation enforcement
        5. If enforcement fails, retry once with stricter prompt
        6. Return final answer + enforcement status

        Args:
            query: User's question.
            context_chunks: Reranked chunks with chunk_id, text, metadata.

        Returns:
            Tuple of:
            - CitedAnswer: The structured answer
            - bool: Whether citation enforcement passed
            - list[str]: Any enforcement errors
        """
        # Pre-decline check
        should_decline, decline_reason = self._enforcer.should_decline(context_chunks)
        if should_decline:
            logger.info("pre_generation_decline", reason=decline_reason)
            return (
                CitedAnswer(
                    answer_text=decline_reason,
                    citations=[],
                    confidence=0.0,
                    reasoning="Declined due to insufficient context quality.",
                ),
                False,
                [decline_reason],
            )

        # Format context
        context_str = format_context_for_llm(context_chunks)

        # First attempt
        answer = self._call_llm(query, context_str)
        enforcement = self._enforcer.enforce(answer, context_chunks)

        if enforcement.is_valid:
            logger.info(
                "generation_accepted_first_attempt",
                confidence=answer.confidence,
                citations=len(answer.citations),
            )
            return answer, True, []

        # Retry with stricter prompt
        if self._max_retries > 0:
            logger.warning(
                "generation_retry",
                errors=enforcement.errors,
                attempt=2,
            )
            error_str = "\n".join(f"- {e}" for e in enforcement.errors)
            answer = self._call_llm(
                query, context_str, retry_errors=error_str
            )
            enforcement = self._enforcer.enforce(answer, context_chunks)

            if enforcement.is_valid:
                logger.info(
                    "generation_accepted_retry",
                    confidence=answer.confidence,
                    citations=len(answer.citations),
                )
                return answer, True, []

        # Both attempts failed
        logger.warning(
            "generation_enforcement_failed",
            errors=enforcement.errors,
            hallucinated_ids=enforcement.hallucinated_ids,
        )
        return answer, False, enforcement.errors

    def _call_llm(
        self,
        query: str,
        context: str,
        retry_errors: str | None = None,
    ) -> CitedAnswer:
        """Call the LLM and parse the response into CitedAnswer.

        Args:
            query: User's question.
            context: Formatted context string.
            retry_errors: If set, appends retry correction prompt.

        Returns:
            Parsed CitedAnswer from LLM response.
        """
        llm = self._get_llm()

        # Build prompt
        system = SYSTEM_PROMPT
        if retry_errors:
            system += RETRY_PROMPT_SUFFIX.format(errors=retry_errors)

        user_message = (
            f"QUESTION: {query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"Generate a cited answer using ONLY the context above."
        )

        try:
            # Use structured output via with_structured_output
            structured_llm = llm.with_structured_output(CitedAnswer)
            result = structured_llm.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ]
            )

            logger.info(
                "llm_call_complete",
                model=self._model_name,
                confidence=result.confidence,
                citations=len(result.citations),
                is_retry=retry_errors is not None,
            )

            return result

        except Exception as e:
            logger.error("llm_call_failed", error=str(e))
            # Return a low-confidence answer on failure
            return CitedAnswer(
                answer_text=f"Generation failed: {e}",
                citations=[],
                confidence=0.0,
                reasoning=f"LLM call error: {e}",
            )
