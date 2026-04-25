"""Tests for LangGraph orchestration pipeline.

Tests cover:
- Query router: route classification, decline patterns, calculate patterns
- Pipeline nodes: retrieve, rerank, generate, validate, decline
- Graph construction: graph builds and compiles
- End-to-end: full pipeline invocation with mock retrievers
- Edge cases: empty queries, error states, step count guards
"""

from unittest.mock import MagicMock, patch

import pytest

from finrag.orchestration.graph import (
    build_rag_graph,
    compile_rag_graph,
    invoke_pipeline,
)
from finrag.orchestration.nodes import (
    calculate,
    decline,
    generate,
    handle_error,
    rerank,
    validate,
    retrieve,
)
from finrag.orchestration.router import get_route, route_query
from finrag.orchestration.state import VALID_ROUTES


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Sample retrieved chunks for testing."""
    return [
        {
            "chunk_id": "aapl_rev_001",
            "text": (
                "Apple Inc. reported total revenue of $383.3 billion "
                "for fiscal year 2024."
            ),
            "metadata": {"ticker": "AAPL", "section_name": "Item 7 - MD&A"},
            "rrf_score": 0.032,
        },
        {
            "chunk_id": "aapl_rev_002",
            "text": (
                "Services revenue increased 13 percent year over year "
                "to $96.2 billion."
            ),
            "metadata": {"ticker": "AAPL", "section_name": "Item 7 - MD&A"},
            "rrf_score": 0.028,
        },
        {
            "chunk_id": "aapl_risk_001",
            "text": (
                "Apple faces risks from supply chain disruptions "
                "in the Asia-Pacific region."
            ),
            "metadata": {"ticker": "AAPL", "section_name": "Item 1A - Risk Factors"},
            "rrf_score": 0.022,
        },
    ]


@pytest.fixture
def reranked_chunks() -> list[dict]:
    """Sample reranked chunks for testing."""
    return [
        {
            "chunk_id": "aapl_rev_001",
            "text": "Apple Inc. reported total revenue of $383.3 billion.",
            "metadata": {"ticker": "AAPL"},
            "reranker_score": 0.95,
            "reranker_rank": 1,
        },
        {
            "chunk_id": "aapl_rev_002",
            "text": "Services revenue increased 13 percent.",
            "metadata": {"ticker": "AAPL"},
            "reranker_score": 0.82,
            "reranker_rank": 2,
        },
    ]


# --------------------------------------------------------------------------- #
# Router Tests
# --------------------------------------------------------------------------- #


class TestRouteQuery:
    """Tests for query routing logic."""

    def test_factual_query_routes_to_retrieve(self) -> None:
        """Standard factual query → retrieve route."""
        state = {"query": "What was Apple's revenue in 2024?", "step_count": 0}
        result = route_query(state)
        assert result["route"] == "retrieve"
        assert result["query_intent"] == "factual_extraction"

    def test_comparison_query_routes_to_calculate(self) -> None:
        """Numerical comparison → calculate route."""
        state = {"query": "Compare Apple's revenue to Microsoft's", "step_count": 0}
        result = route_query(state)
        assert result["route"] == "calculate"
        assert result["query_intent"] == "numerical_comparison"

    def test_yoy_query_routes_to_calculate(self) -> None:
        """Year-over-year query → calculate route."""
        state = {"query": "What was the YoY growth in revenue?", "step_count": 0}
        result = route_query(state)
        assert result["route"] == "calculate"

    def test_percentage_query_routes_to_calculate(self) -> None:
        """Percentage query → calculate route."""
        state = {"query": "What percentage of revenue came from services?", "step_count": 0}
        result = route_query(state)
        assert result["route"] == "calculate"

    def test_investment_advice_declined(self) -> None:
        """Investment advice → decline route."""
        state = {"query": "Should I buy Apple stock?", "step_count": 0}
        result = route_query(state)
        assert result["route"] == "decline"
        assert result["query_intent"] == "financial_advice"

    def test_stock_prediction_declined(self) -> None:
        """Stock prediction → decline route."""
        state = {"query": "Can you predict stock price movements?", "step_count": 0}
        result = route_query(state)
        assert result["route"] == "decline"

    def test_recommendation_declined(self) -> None:
        """Recommendation request → decline route."""
        state = {"query": "What's the best stock to invest in?", "step_count": 0}
        result = route_query(state)
        assert result["route"] == "decline"

    def test_empty_query_declined(self) -> None:
        """Empty query → decline route."""
        state = {"query": "", "step_count": 0}
        result = route_query(state)
        assert result["route"] == "decline"
        assert result["query_intent"] == "empty"

    def test_step_count_incremented(self) -> None:
        """Router increments step count."""
        state = {"query": "What was revenue?", "step_count": 3}
        result = route_query(state)
        assert result["step_count"] == 4

    def test_route_confidence_present(self) -> None:
        """Route includes confidence score."""
        state = {"query": "What was revenue?", "step_count": 0}
        result = route_query(state)
        assert 0.0 <= result["route_confidence"] <= 1.0

    def test_all_routes_are_valid(self) -> None:
        """All routes returned are in VALID_ROUTES."""
        queries = [
            "What was revenue?",
            "Compare margins",
            "Should I buy stock?",
        ]
        for q in queries:
            result = route_query({"query": q, "step_count": 0})
            assert result["route"] in VALID_ROUTES


class TestGetRoute:
    """Tests for the edge routing function."""

    def test_returns_route_from_state(self) -> None:
        """Returns the route field."""
        assert get_route({"route": "calculate"}) == "calculate"

    def test_defaults_to_retrieve(self) -> None:
        """Missing route defaults to retrieve."""
        assert get_route({}) == "retrieve"


# --------------------------------------------------------------------------- #
# Node Tests
# --------------------------------------------------------------------------- #


class TestRetrieveNode:
    """Tests for the retrieve node."""

    def test_successful_retrieval(self, sample_chunks: list[dict]) -> None:
        """Retrieval returns chunks and metadata."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks

        state = {"query": "Apple revenue", "step_count": 0}
        result = retrieve(state, hybrid_retriever=mock_retriever)

        assert len(result["retrieved_chunks"]) == 3
        assert result["retrieval_metadata"]["num_candidates"] == 3
        assert result["step_count"] == 1

    def test_retrieval_with_filter(self, sample_chunks: list[dict]) -> None:
        """Metadata filter is passed to retriever."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks

        state = {
            "query": "Apple revenue",
            "metadata_filter": {"ticker": "AAPL"},
            "step_count": 0,
        }
        result = retrieve(state, hybrid_retriever=mock_retriever)

        mock_retriever.retrieve.assert_called_once_with(
            query="Apple revenue",
            n_results=30,
            where={"ticker": "AAPL"},
            use_multi_query=True,
        )
        assert result["retrieval_metadata"]["filter_applied"] == {"ticker": "AAPL"}

    def test_retrieval_failure_returns_error(self) -> None:
        """Retrieval exception produces error state."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RuntimeError("connection failed")

        state = {"query": "test query", "step_count": 0}
        result = retrieve(state, hybrid_retriever=mock_retriever)

        assert result["retrieved_chunks"] == []
        assert "Retrieval failed" in result["error"]
        assert result["step_count"] == 1


class TestRerankNode:
    """Tests for the rerank node."""

    def test_successful_rerank(self, sample_chunks: list[dict]) -> None:
        """Reranking returns reranked chunks."""
        mock_reranker = MagicMock()
        reranked = [
            {**c, "reranker_score": 0.9 - i * 0.1, "reranker_rank": i + 1}
            for i, c in enumerate(sample_chunks[:2])
        ]
        mock_reranker.rerank.return_value = reranked

        state = {
            "query": "Apple revenue",
            "retrieved_chunks": sample_chunks,
            "step_count": 1,
        }
        result = rerank(state, reranker=mock_reranker)

        assert len(result["reranked_chunks"]) == 2
        assert result["step_count"] == 2

    def test_rerank_empty_chunks(self) -> None:
        """Empty chunks produces empty reranked list."""
        mock_reranker = MagicMock()
        state = {"query": "test", "retrieved_chunks": [], "step_count": 1}
        result = rerank(state, reranker=mock_reranker)

        assert result["reranked_chunks"] == []
        mock_reranker.rerank.assert_not_called()

    def test_rerank_failure_falls_back(self, sample_chunks: list[dict]) -> None:
        """Rerank failure falls back to unranked top-k."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("model error")

        state = {
            "query": "test",
            "retrieved_chunks": sample_chunks,
            "step_count": 1,
        }
        result = rerank(state, reranker=mock_reranker)

        assert len(result["reranked_chunks"]) > 0
        assert "Reranking failed" in result["error"]


class TestGenerateNode:
    """Tests for the generate stub node."""

    def test_generate_with_chunks(self, reranked_chunks: list[dict]) -> None:
        """Generate produces stub answer with citations."""
        state = {
            "query": "What was revenue?",
            "reranked_chunks": reranked_chunks,
            "step_count": 2,
        }
        result = generate(state)

        assert "[STUB" in result["answer"]
        assert len(result["citations"]) == 2
        assert result["generation_model"] == "stub_v1"
        assert result["step_count"] == 3

    def test_generate_no_chunks(self) -> None:
        """Generate with no chunks returns error."""
        state = {"query": "test", "reranked_chunks": [], "step_count": 2}
        result = generate(state)

        assert result["answer"] == ""
        assert "No context chunks" in result["error"]


class TestCalculateNode:
    """Tests for the calculate stub node."""

    def test_calculate_returns_stub(self) -> None:
        """Calculate produces stub response."""
        state = {
            "query": "Compare revenue to expenses",
            "reranked_chunks": [],
            "step_count": 2,
        }
        result = calculate(state)

        assert "[STUB" in result["answer"]
        assert "numerical computation" in result["answer"]


class TestValidateNode:
    """Tests for the validate node."""

    def test_valid_answer(self, reranked_chunks: list[dict]) -> None:
        """Valid answer passes validation."""
        state = {
            "answer": "Apple reported $383.3 billion in revenue.",
            "citations": [{"chunk_id": "aapl_rev_001"}],
            "step_count": 3,
            "max_steps": 10,
        }
        result = validate(state)

        assert result["is_valid"] is True
        assert result["validation_errors"] == []

    def test_empty_answer_fails(self) -> None:
        """Empty answer fails validation."""
        state = {"answer": "", "citations": [], "step_count": 3, "max_steps": 10}
        result = validate(state)

        assert result["is_valid"] is False
        assert any("Empty answer" in e for e in result["validation_errors"])

    def test_max_steps_exceeded(self) -> None:
        """Exceeding max steps fails validation."""
        state = {
            "answer": "Some answer",
            "citations": [{"chunk_id": "x"}],
            "step_count": 11,
            "max_steps": 10,
        }
        result = validate(state)

        assert result["is_valid"] is False
        assert any("Max steps" in e for e in result["validation_errors"])

    def test_upstream_error_fails(self) -> None:
        """Upstream error propagates to validation."""
        state = {
            "answer": "Some answer",
            "citations": [{"chunk_id": "x"}],
            "error": "retrieval timed out",
            "step_count": 3,
            "max_steps": 10,
        }
        result = validate(state)

        assert result["is_valid"] is False

    def test_stub_answer_passes_without_citations(self) -> None:
        """Stub answers (with [STUB marker) pass without citations."""
        state = {
            "answer": "[STUB — Day 8 will add LLM generation]",
            "citations": [],
            "step_count": 3,
            "max_steps": 10,
        }
        result = validate(state)

        assert result["is_valid"] is True


class TestDeclineNode:
    """Tests for the decline node."""

    def test_decline_response(self) -> None:
        """Decline produces helpful message."""
        state = {"query": "Should I buy Apple stock?", "step_count": 1}
        result = decline(state)

        assert "cannot provide investment advice" in result["answer"]
        assert result["citations"] == []
        assert result["is_valid"] is True
        assert result["generation_model"] == "decline"


class TestHandleErrorNode:
    """Tests for the error handler node."""

    def test_error_response(self) -> None:
        """Error handler returns user-friendly message."""
        state = {"error": "Database unreachable", "query": "test", "step_count": 3}
        result = handle_error(state)

        assert "encountered an issue" in result["answer"]
        assert "Database unreachable" in result["answer"]
        assert result["is_valid"] is False


# --------------------------------------------------------------------------- #
# Graph Construction Tests
# --------------------------------------------------------------------------- #


class TestGraphConstruction:
    """Tests for graph building and compilation."""

    def test_graph_builds(self) -> None:
        """Graph builds without error."""
        mock_retriever = MagicMock()
        mock_reranker = MagicMock()
        graph = build_rag_graph(mock_retriever, mock_reranker)
        assert graph is not None

    def test_graph_compiles(self) -> None:
        """Graph compiles without error."""
        mock_retriever = MagicMock()
        mock_reranker = MagicMock()
        compiled = compile_rag_graph(mock_retriever, mock_reranker)
        assert compiled is not None


# --------------------------------------------------------------------------- #
# End-to-End Pipeline Tests
# --------------------------------------------------------------------------- #


class TestPipelineE2E:
    """End-to-end tests with mock components."""

    def test_retrieve_pipeline(self) -> None:
        """Full retrieve → rerank → generate → validate pipeline."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {
                "chunk_id": "aapl_001",
                "text": "Apple reported $383B in revenue.",
                "metadata": {"ticker": "AAPL"},
                "rrf_score": 0.03,
            },
        ]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {
                "chunk_id": "aapl_001",
                "text": "Apple reported $383B in revenue.",
                "metadata": {"ticker": "AAPL"},
                "reranker_score": 0.95,
                "reranker_rank": 1,
            },
        ]

        compiled = compile_rag_graph(mock_retriever, mock_reranker)
        result = invoke_pipeline(compiled, "What was Apple's revenue?")

        assert result["route"] == "retrieve"
        assert result["answer"]  # Non-empty
        assert result["is_valid"] is True
        mock_retriever.retrieve.assert_called()
        mock_reranker.rerank.assert_called()

    def test_decline_pipeline(self) -> None:
        """Decline route skips retrieval entirely."""
        mock_retriever = MagicMock()
        mock_reranker = MagicMock()

        compiled = compile_rag_graph(mock_retriever, mock_reranker)
        result = invoke_pipeline(compiled, "Should I buy AAPL stock?")

        assert result["route"] == "decline"
        assert "cannot provide investment advice" in result["answer"]
        assert result["is_valid"] is True
        mock_retriever.retrieve.assert_not_called()

    def test_calculate_pipeline(self) -> None:
        """Calculate route goes through retrieve → rerank → calculate."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {
                "chunk_id": "data_001",
                "text": "Revenue was $100B in 2023 and $120B in 2024.",
                "metadata": {},
                "rrf_score": 0.03,
            },
        ]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {
                "chunk_id": "data_001",
                "text": "Revenue was $100B in 2023 and $120B in 2024.",
                "metadata": {},
                "reranker_score": 0.9,
                "reranker_rank": 1,
            },
        ]

        compiled = compile_rag_graph(mock_retriever, mock_reranker)
        result = invoke_pipeline(compiled, "Compare revenue between 2023 and 2024")

        assert result["route"] == "calculate"
        assert "[STUB" in result["answer"]
        mock_retriever.retrieve.assert_called()

    def test_pipeline_with_metadata_filter(self) -> None:
        """Metadata filter is passed through the pipeline."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {
                "chunk_id": "aapl_001",
                "text": "Apple revenue data.",
                "metadata": {"ticker": "AAPL"},
                "rrf_score": 0.03,
            },
        ]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {
                "chunk_id": "aapl_001",
                "text": "Apple revenue data.",
                "metadata": {"ticker": "AAPL"},
                "reranker_score": 0.9,
                "reranker_rank": 1,
            },
        ]

        compiled = compile_rag_graph(mock_retriever, mock_reranker)
        result = invoke_pipeline(
            compiled,
            "What was revenue?",
            metadata_filter={"ticker": "AAPL"},
        )

        # Verify filter was passed to retriever
        call_kwargs = mock_retriever.retrieve.call_args
        assert call_kwargs.kwargs["where"] == {"ticker": "AAPL"}
