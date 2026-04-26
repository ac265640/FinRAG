"""Pydantic output schemas for the FinRAG generation pipeline.

Defines the structured output format that the LLM must produce.
Every field is typed, validated, and documented. Downstream consumers
(API, frontend, Langfuse) rely on these exact field names.

Design decisions:
- Pydantic over plain dicts: type safety, validation, serialization.
  The API layer (Day 11) will return these models directly as JSON.
- Citation as separate model: each citation maps one claim to one
  source chunk. This makes citation coverage measurable.
- Confidence score: 0-1 float, reflects model self-assessment of
  answer quality given the available context. Used by the validate
  node to decide whether to accept or retry.
- filing_reference: human-readable source string like
  "AAPL 10-K FY2024, Item 7 - MD&A". Constructed from chunk metadata.
"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single citation linking a claim to a source chunk.

    Each citation represents one piece of evidence supporting
    the answer. The chunk_id must map to a real retrieved chunk —
    hallucinated chunk_ids are caught by the citation enforcer.

    Attributes:
        chunk_id: The unique identifier of the source chunk.
            Must match a chunk_id from the retrieved context.
        filing_reference: Human-readable source reference.
            Format: "TICKER FILING_TYPE PERIOD, SECTION".
            Example: "AAPL 10-K FY2024, Item 7 - MD&A"
        section: Filing section name (e.g., "Item 1A - Risk Factors").
        text_excerpt: Short excerpt from the source chunk that
            supports the claim. Max 200 chars.
        relevance_score: Reranker score for this chunk (0-1).
    """

    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier from retrieved context.",
    )
    filing_reference: str = Field(
        default="",
        description="Human-readable filing reference (e.g., 'AAPL 10-K FY2024, Item 7').",
    )
    section: str = Field(
        default="",
        description="Filing section name.",
    )
    text_excerpt: str = Field(
        default="",
        max_length=300,
        description="Supporting text excerpt from the source chunk.",
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reranker relevance score (0-1).",
    )


class CitedAnswer(BaseModel):
    """Structured answer with mandatory citations.

    The LLM must produce output conforming to this schema.
    Every factual claim in the answer must be backed by at least
    one citation. The citation enforcer validates this post-generation.

    Attributes:
        answer_text: The generated answer in natural language.
            Must be grounded in the provided context chunks.
        citations: List of citations supporting the answer.
            Each maps a claim to a specific source chunk.
        confidence: Model's self-assessed confidence (0-1).
            Below the threshold (default 0.3), the system
            declines to answer rather than risk hallucination.
        reasoning: Brief explanation of how the answer was
            derived from the sources. Helps with debugging
            and Langfuse trace inspection.
    """

    answer_text: str = Field(
        ...,
        min_length=1,
        description="Generated answer grounded in retrieved context.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations linking claims to source chunks.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Model confidence in the answer (0-1).",
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning chain for debugging.",
    )


def build_filing_reference(metadata: dict) -> str:
    """Build a human-readable filing reference from chunk metadata.

    Constructs a string like "AAPL 10-K FY2024, Item 7 - MD&A"
    from the structured metadata attached to each chunk.

    Args:
        metadata: Chunk metadata dict with optional keys:
            ticker, form_type, fiscal_period, section_name.

    Returns:
        Formatted filing reference string.
    """
    parts: list[str] = []

    ticker = metadata.get("ticker", "")
    if ticker:
        parts.append(ticker)

    form_type = metadata.get("form_type", "")
    if form_type:
        parts.append(form_type)

    period = metadata.get("fiscal_period", "")
    if period:
        parts.append(period)

    base = " ".join(parts)

    section = metadata.get("section_name", "")
    if section:
        return f"{base}, {section}" if base else section

    return base
