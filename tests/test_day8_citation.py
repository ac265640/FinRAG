"""Tests for Day 8: Pydantic schemas, citation enforcement, and LLM generation.

Tests cover:
- Schemas: CitedAnswer and Citation validation, filing reference builder
- CitationEnforcer: hallucinated ID detection, confidence threshold,
  relevance floor, pre-generation decline
- RAGGenerator: mock LLM calls, retry logic, pre-decline
- Updated nodes: generate/calculate with RAGGenerator, stub fallback
"""

from unittest.mock import MagicMock, patch

import pytest

from finrag.orchestration.citation import (
    CitationEnforcer,
    EnforcementResult,
)
from finrag.orchestration.schemas import (
    CitedAnswer,
    Citation,
    build_filing_reference,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def valid_chunks() -> list[dict]:
    """Chunks that simulate a successful retrieval + rerank."""
    return [
        {
            "chunk_id": "aapl_rev_001",
            "text": "Apple reported total revenue of $383.3 billion for FY2024.",
            "metadata": {
                "ticker": "AAPL",
                "form_type": "10-K",
                "fiscal_period": "FY2024",
                "section_name": "Item 7 - MD&A",
            },
            "reranker_score": 0.95,
            "reranker_rank": 1,
        },
        {
            "chunk_id": "aapl_svc_002",
            "text": "Services revenue increased 13 percent YoY to $96.2B.",
            "metadata": {
                "ticker": "AAPL",
                "form_type": "10-K",
                "fiscal_period": "FY2024",
                "section_name": "Item 7 - MD&A",
            },
            "reranker_score": 0.82,
            "reranker_rank": 2,
        },
        {
            "chunk_id": "aapl_risk_003",
            "text": "Supply chain risks remain elevated in Asia-Pacific.",
            "metadata": {
                "ticker": "AAPL",
                "form_type": "10-K",
                "fiscal_period": "FY2024",
                "section_name": "Item 1A - Risk Factors",
            },
            "reranker_score": 0.65,
            "reranker_rank": 3,
        },
    ]


@pytest.fixture
def valid_answer(valid_chunks: list[dict]) -> CitedAnswer:
    """A correctly cited answer."""
    return CitedAnswer(
        answer_text="Apple reported $383.3B in total revenue for FY2024.",
        citations=[
            Citation(
                chunk_id="aapl_rev_001",
                filing_reference="AAPL 10-K FY2024, Item 7 - MD&A",
                section="Item 7 - MD&A",
                text_excerpt="total revenue of $383.3 billion",
                relevance_score=0.95,
            ),
        ],
        confidence=0.92,
        reasoning="Direct extraction from MD&A section.",
    )


@pytest.fixture
def enforcer() -> CitationEnforcer:
    """Default citation enforcer."""
    return CitationEnforcer()


# --------------------------------------------------------------------------- #
# Schema Tests
# --------------------------------------------------------------------------- #


class TestCitationSchema:
    """Tests for the Citation Pydantic model."""

    def test_basic_creation(self) -> None:
        """Create a citation with required fields."""
        c = Citation(chunk_id="chunk_001")
        assert c.chunk_id == "chunk_001"
        assert c.relevance_score == 0.0

    def test_full_citation(self) -> None:
        """Citation with all fields populated."""
        c = Citation(
            chunk_id="aapl_001",
            filing_reference="AAPL 10-K FY2024, Item 7",
            section="Item 7 - MD&A",
            text_excerpt="Revenue was $383B",
            relevance_score=0.95,
        )
        assert c.chunk_id == "aapl_001"
        assert c.relevance_score == 0.95

    def test_score_bounds(self) -> None:
        """Relevance score must be 0-1."""
        with pytest.raises(ValueError):
            Citation(chunk_id="x", relevance_score=1.5)
        with pytest.raises(ValueError):
            Citation(chunk_id="x", relevance_score=-0.1)


class TestCitedAnswerSchema:
    """Tests for the CitedAnswer Pydantic model."""

    def test_basic_creation(self) -> None:
        """Create answer with required fields only."""
        a = CitedAnswer(answer_text="Apple had $383B revenue.")
        assert a.answer_text == "Apple had $383B revenue."
        assert a.citations == []
        assert a.confidence == 0.5

    def test_with_citations(self) -> None:
        """Answer with citation list."""
        a = CitedAnswer(
            answer_text="Revenue was $383B.",
            citations=[Citation(chunk_id="c1"), Citation(chunk_id="c2")],
            confidence=0.9,
        )
        assert len(a.citations) == 2

    def test_empty_answer_rejected(self) -> None:
        """Empty answer_text fails validation."""
        with pytest.raises(ValueError):
            CitedAnswer(answer_text="")

    def test_confidence_bounds(self) -> None:
        """Confidence must be 0-1."""
        with pytest.raises(ValueError):
            CitedAnswer(answer_text="x", confidence=1.5)


class TestBuildFilingReference:
    """Tests for the filing reference builder."""

    def test_full_metadata(self) -> None:
        """All fields present."""
        ref = build_filing_reference({
            "ticker": "AAPL",
            "form_type": "10-K",
            "fiscal_period": "FY2024",
            "section_name": "Item 7 - MD&A",
        })
        assert ref == "AAPL 10-K FY2024, Item 7 - MD&A"

    def test_partial_metadata(self) -> None:
        """Only ticker and form_type."""
        ref = build_filing_reference({
            "ticker": "MSFT",
            "form_type": "10-Q",
        })
        assert ref == "MSFT 10-Q"

    def test_empty_metadata(self) -> None:
        """Empty metadata returns empty string."""
        ref = build_filing_reference({})
        assert ref == ""

    def test_section_only(self) -> None:
        """Only section name present."""
        ref = build_filing_reference({"section_name": "Item 1A"})
        assert ref == "Item 1A"


# --------------------------------------------------------------------------- #
# Citation Enforcer Tests
# --------------------------------------------------------------------------- #


class TestCitationEnforcer:
    """Tests for citation enforcement logic."""

    def test_valid_answer_passes(
        self,
        enforcer: CitationEnforcer,
        valid_answer: CitedAnswer,
        valid_chunks: list[dict],
    ) -> None:
        """Correctly cited answer passes enforcement."""
        result = enforcer.enforce(valid_answer, valid_chunks)
        assert result.is_valid is True
        assert result.errors == []
        assert result.valid_citation_count == 1
        assert result.hallucinated_ids == []

    def test_hallucinated_citation_detected(
        self,
        enforcer: CitationEnforcer,
        valid_chunks: list[dict],
    ) -> None:
        """Citation with non-existent chunk_id is caught."""
        answer = CitedAnswer(
            answer_text="Revenue was $383B.",
            citations=[
                Citation(chunk_id="FAKE_ID_999", relevance_score=0.8),
            ],
            confidence=0.9,
        )
        result = enforcer.enforce(answer, valid_chunks)
        assert result.is_valid is False
        assert "FAKE_ID_999" in result.hallucinated_ids
        assert any("Hallucinated" in e for e in result.errors)

    def test_mixed_valid_and_hallucinated(
        self,
        enforcer: CitationEnforcer,
        valid_chunks: list[dict],
    ) -> None:
        """Mix of real and hallucinated citations."""
        answer = CitedAnswer(
            answer_text="Revenue was $383B.",
            citations=[
                Citation(chunk_id="aapl_rev_001"),  # Valid
                Citation(chunk_id="HALLUCINATED_X"),  # Fake
            ],
            confidence=0.9,
        )
        result = enforcer.enforce(answer, valid_chunks)
        assert result.is_valid is False
        assert result.valid_citation_count == 1
        assert result.hallucinated_ids == ["HALLUCINATED_X"]

    def test_low_confidence_rejected(
        self,
        enforcer: CitationEnforcer,
        valid_chunks: list[dict],
    ) -> None:
        """Answer below confidence threshold is rejected."""
        answer = CitedAnswer(
            answer_text="Maybe revenue was around $380B?",
            citations=[Citation(chunk_id="aapl_rev_001")],
            confidence=0.1,  # Below 0.3 threshold
        )
        result = enforcer.enforce(answer, valid_chunks)
        assert result.is_valid is False
        assert any("Confidence" in e for e in result.errors)

    def test_no_citations_rejected(
        self,
        enforcer: CitationEnforcer,
        valid_chunks: list[dict],
    ) -> None:
        """Answer with no citations is rejected."""
        answer = CitedAnswer(
            answer_text="Revenue was high.",
            citations=[],
            confidence=0.9,
        )
        result = enforcer.enforce(answer, valid_chunks)
        assert result.is_valid is False
        assert any("citations" in e.lower() for e in result.errors)

    def test_low_relevance_floor_rejected(self) -> None:
        """Top reranker score below floor triggers rejection."""
        enforcer = CitationEnforcer(relevance_floor=0.5)
        low_quality_chunks = [
            {"chunk_id": "c1", "reranker_score": 0.1},
            {"chunk_id": "c2", "reranker_score": 0.15},
        ]
        answer = CitedAnswer(
            answer_text="Some answer.",
            citations=[Citation(chunk_id="c1")],
            confidence=0.9,
        )
        result = enforcer.enforce(answer, low_quality_chunks)
        assert result.is_valid is False
        assert any("relevance" in e.lower() for e in result.errors)

    def test_custom_thresholds(self) -> None:
        """Custom enforcer thresholds work."""
        strict = CitationEnforcer(
            confidence_threshold=0.8,
            min_citations=3,
            relevance_floor=0.5,
        )
        answer = CitedAnswer(
            answer_text="Revenue data.",
            citations=[Citation(chunk_id="c1")],
            confidence=0.75,
        )
        chunks = [{"chunk_id": "c1", "reranker_score": 0.6}]
        result = strict.enforce(answer, chunks)
        assert result.is_valid is False
        # Should fail on both confidence and citation count
        assert len(result.errors) >= 2


class TestPreGenerationDecline:
    """Tests for the pre-generation decline check."""

    def test_empty_chunks_declined(self) -> None:
        """No chunks → decline."""
        enforcer = CitationEnforcer()
        should_decline, reason = enforcer.should_decline([])
        assert should_decline is True
        assert "No context chunks" in reason

    def test_low_quality_declined(self) -> None:
        """All chunks below relevance floor → decline."""
        enforcer = CitationEnforcer(relevance_floor=0.5)
        chunks = [
            {"chunk_id": "c1", "reranker_score": 0.1},
            {"chunk_id": "c2", "reranker_score": 0.2},
        ]
        should_decline, reason = enforcer.should_decline(chunks)
        assert should_decline is True
        assert "quality threshold" in reason

    def test_good_quality_accepted(self, valid_chunks: list[dict]) -> None:
        """High quality chunks → proceed."""
        enforcer = CitationEnforcer()
        should_decline, _ = enforcer.should_decline(valid_chunks)
        assert should_decline is False


# --------------------------------------------------------------------------- #
# EnforcementResult Tests
# --------------------------------------------------------------------------- #


class TestEnforcementResult:
    """Tests for the EnforcementResult dataclass."""

    def test_default_valid(self) -> None:
        """New result starts valid."""
        r = EnforcementResult()
        assert r.is_valid is True
        assert r.errors == []

    def test_add_error_invalidates(self) -> None:
        """Adding error flips is_valid."""
        r = EnforcementResult()
        r.add_error("test error")
        assert r.is_valid is False
        assert "test error" in r.errors


# --------------------------------------------------------------------------- #
# Generator Tests (mocked LLM)
# --------------------------------------------------------------------------- #


class TestRAGGenerator:
    """Tests for RAGGenerator with mocked LLM."""

    def test_pre_decline_on_empty_chunks(self) -> None:
        """Generator declines when no chunks provided."""
        from finrag.orchestration.generator import RAGGenerator

        gen = RAGGenerator(api_key="fake-key")
        answer, passed, errors = gen.generate("What was revenue?", [])

        assert passed is False
        assert answer.confidence == 0.0
        assert len(errors) > 0

    def test_pre_decline_on_low_quality(self) -> None:
        """Generator declines when top score below floor."""
        from finrag.orchestration.generator import RAGGenerator

        gen = RAGGenerator(api_key="fake-key")
        low_chunks = [
            {"chunk_id": "c1", "text": "irrelevant", "metadata": {}, "reranker_score": 0.05},
        ]
        answer, passed, errors = gen.generate("What was revenue?", low_chunks)

        assert passed is False
        assert answer.confidence == 0.0

    @patch("finrag.orchestration.generator.ChatGoogleGenerativeAI")
    def test_successful_generation(self, mock_llm_cls: MagicMock, valid_chunks: list[dict]) -> None:
        """Mocked LLM returns valid CitedAnswer."""
        from finrag.orchestration.generator import RAGGenerator

        # Mock the structured output chain
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = CitedAnswer(
            answer_text="Apple reported $383.3B in revenue for FY2024.",
            citations=[
                Citation(
                    chunk_id="aapl_rev_001",
                    filing_reference="AAPL 10-K FY2024",
                    section="Item 7",
                    text_excerpt="$383.3 billion",
                    relevance_score=0.95,
                ),
            ],
            confidence=0.92,
            reasoning="Direct extraction.",
        )

        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_structured
        mock_llm_cls.return_value = mock_llm_instance

        gen = RAGGenerator(api_key="fake-key")
        answer, passed, errors = gen.generate("What was Apple revenue?", valid_chunks)

        assert passed is True
        assert answer.confidence == 0.92
        assert len(answer.citations) == 1
        assert answer.citations[0].chunk_id == "aapl_rev_001"

    @patch("finrag.orchestration.generator.ChatGoogleGenerativeAI")
    def test_hallucinated_citation_triggers_retry(
        self, mock_llm_cls: MagicMock, valid_chunks: list[dict]
    ) -> None:
        """Hallucinated citation triggers retry with stricter prompt."""
        from finrag.orchestration.generator import RAGGenerator

        # First call: hallucinated citation
        bad_answer = CitedAnswer(
            answer_text="Revenue was $383B.",
            citations=[Citation(chunk_id="FAKE_HALLUCINATED")],
            confidence=0.8,
        )
        # Second call (retry): valid citation
        good_answer = CitedAnswer(
            answer_text="Apple reported $383.3B revenue.",
            citations=[Citation(chunk_id="aapl_rev_001")],
            confidence=0.85,
        )

        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [bad_answer, good_answer]

        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_structured
        mock_llm_cls.return_value = mock_llm_instance

        gen = RAGGenerator(api_key="fake-key")
        answer, passed, errors = gen.generate("What was revenue?", valid_chunks)

        assert passed is True
        assert answer.citations[0].chunk_id == "aapl_rev_001"
        assert mock_structured.invoke.call_count == 2

    @patch("finrag.orchestration.generator.ChatGoogleGenerativeAI")
    def test_both_attempts_fail(
        self, mock_llm_cls: MagicMock, valid_chunks: list[dict]
    ) -> None:
        """Both generation attempts failing returns errors."""
        from finrag.orchestration.generator import RAGGenerator

        bad_answer = CitedAnswer(
            answer_text="Revenue was high.",
            citations=[Citation(chunk_id="FAKE")],
            confidence=0.1,
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = bad_answer

        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_structured
        mock_llm_cls.return_value = mock_llm_instance

        gen = RAGGenerator(api_key="fake-key")
        answer, passed, errors = gen.generate("What was revenue?", valid_chunks)

        assert passed is False
        assert len(errors) > 0

    def test_no_api_key_raises(self) -> None:
        """Missing API key raises ValueError on LLM call."""
        from finrag.orchestration.generator import RAGGenerator

        gen = RAGGenerator(api_key="")
        with pytest.raises(ValueError, match="No Google API key"):
            gen._get_llm()


# --------------------------------------------------------------------------- #
# Context Formatter Tests
# --------------------------------------------------------------------------- #


class TestFormatContext:
    """Tests for the context formatting helper."""

    def test_formats_chunks(self, valid_chunks: list[dict]) -> None:
        """Chunks are formatted with IDs and metadata."""
        from finrag.orchestration.generator import format_context_for_llm

        result = format_context_for_llm(valid_chunks)

        assert "aapl_rev_001" in result
        assert "AAPL 10-K FY2024, Item 7 - MD&A" in result
        assert "0.95" in result
        assert "---" in result  # separator

    def test_empty_chunks(self) -> None:
        """Empty chunk list returns empty string."""
        from finrag.orchestration.generator import format_context_for_llm

        assert format_context_for_llm([]) == ""


# --------------------------------------------------------------------------- #
# Updated Node Tests
# --------------------------------------------------------------------------- #


class TestUpdatedGenerateNode:
    """Tests for the updated generate node with RAGGenerator."""

    def test_stub_fallback_without_generator(self) -> None:
        """No generator → stub behavior (backward compat)."""
        from finrag.orchestration.nodes import generate

        state = {
            "query": "What was revenue?",
            "reranked_chunks": [
                {"chunk_id": "c1", "text": "Revenue was $383B.", "metadata": {}, "reranker_score": 0.9},
            ],
            "step_count": 2,
        }
        result = generate(state)

        assert "[STUB" in result["answer"]
        assert result["generation_model"] == "stub_v1"

    def test_empty_chunks_returns_error(self) -> None:
        """No chunks → error state."""
        from finrag.orchestration.nodes import generate

        state = {"query": "test", "reranked_chunks": [], "step_count": 2}
        result = generate(state)

        assert result["answer"] == ""
        assert "No context" in result["error"]

    @patch("finrag.orchestration.generator.ChatGoogleGenerativeAI")
    def test_with_generator(self, mock_llm_cls: MagicMock) -> None:
        """Generate with real RAGGenerator (mocked LLM)."""
        from finrag.orchestration.generator import RAGGenerator
        from finrag.orchestration.nodes import generate

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = CitedAnswer(
            answer_text="Apple revenue was $383B.",
            citations=[Citation(chunk_id="c1", relevance_score=0.9)],
            confidence=0.9,
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_cls.return_value = mock_llm

        gen = RAGGenerator(api_key="fake")
        state = {
            "query": "What was revenue?",
            "reranked_chunks": [
                {"chunk_id": "c1", "text": "Revenue $383B.", "metadata": {}, "reranker_score": 0.9},
            ],
            "step_count": 2,
        }
        result = generate(state, rag_generator=gen)

        assert result["answer"] == "Apple revenue was $383B."
        assert len(result["citations"]) == 1


class TestUpdatedValidateNode:
    """Tests for updated validate node handling enforcement errors."""

    def test_enforcement_error_not_fatal(self) -> None:
        """Enforcement errors don't block validation."""
        from finrag.orchestration.nodes import validate

        state = {
            "answer": "Apple revenue was $383B.",
            "citations": [{"chunk_id": "c1"}],
            "error": "Citation enforcement failed: hallucinated ID",
            "step_count": 3,
            "max_steps": 10,
            "generation_model": "gemini-2.0-flash",
        }
        result = validate(state)

        # Enforcement errors are soft — answer still valid
        assert result["is_valid"] is True

    def test_real_error_is_fatal(self) -> None:
        """Non-enforcement errors block validation."""
        from finrag.orchestration.nodes import validate

        state = {
            "answer": "Some answer.",
            "citations": [{"chunk_id": "c1"}],
            "error": "Retrieval failed: connection timeout",
            "step_count": 3,
            "max_steps": 10,
        }
        result = validate(state)

        assert result["is_valid"] is False
