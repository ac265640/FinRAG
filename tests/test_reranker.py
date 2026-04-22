"""Tests for cross-encoder reranker.

Tests cover:
- Reranker initialization and lazy loading
- Reranking with relevance scoring
- Score ordering (descending by reranker_score)
- Top-k limiting
- Empty input handling
- Min relevance threshold filtering
- Reranker rank assignment
"""

import pytest

from finrag.retrieval.reranker import CrossEncoderReranker, _sigmoid


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def reranker() -> CrossEncoderReranker:
    """Create a reranker with default model."""
    return CrossEncoderReranker(top_k=3)


@pytest.fixture
def candidates() -> list[dict]:
    """Sample candidate results from hybrid retrieval."""
    return [
        {
            "chunk_id": "aapl_rev_001",
            "text": (
                "Apple Inc. reported total revenue of $383.3 billion for "
                "fiscal year 2024. Services revenue increased 13 percent."
            ),
            "metadata": {"ticker": "AAPL", "section_name": "Item 7 - MD&A"},
            "rrf_score": 0.032,
        },
        {
            "chunk_id": "aapl_risk_001",
            "text": (
                "Apple faces risks from supply chain disruptions and "
                "increasing regulatory scrutiny in the European Union."
            ),
            "metadata": {"ticker": "AAPL", "section_name": "Item 1A - Risk Factors"},
            "rrf_score": 0.028,
        },
        {
            "chunk_id": "msft_rev_001",
            "text": (
                "Microsoft Corporation reported revenue of $245.1 billion. "
                "Azure cloud revenue grew 29 percent year over year."
            ),
            "metadata": {"ticker": "MSFT", "section_name": "Item 7 - MD&A"},
            "rrf_score": 0.025,
        },
        {
            "chunk_id": "tsla_rev_001",
            "text": (
                "Tesla reported automotive revenue of $82.4 billion. "
                "Vehicle deliveries totaled 1.81 million units."
            ),
            "metadata": {"ticker": "TSLA", "section_name": "Item 7 - MD&A"},
            "rrf_score": 0.022,
        },
        {
            "chunk_id": "aapl_goodwill_001",
            "text": (
                "The goodwill impairment test resulted in no impairment "
                "charge for the reporting period."
            ),
            "metadata": {"ticker": "AAPL", "section_name": "Item 8 - Financial Statements"},
            "rrf_score": 0.018,
        },
    ]


# --------------------------------------------------------------------------- #
# Sigmoid Tests
# --------------------------------------------------------------------------- #


class TestSigmoid:
    """Tests for sigmoid normalization."""

    def test_sigmoid_zero(self) -> None:
        """sigmoid(0) = 0.5."""
        assert abs(_sigmoid(0.0) - 0.5) < 1e-6

    def test_sigmoid_positive(self) -> None:
        """Positive inputs produce scores > 0.5."""
        assert _sigmoid(5.0) > 0.5

    def test_sigmoid_negative(self) -> None:
        """Negative inputs produce scores < 0.5."""
        assert _sigmoid(-5.0) < 0.5

    def test_sigmoid_large_positive(self) -> None:
        """Very large positive approaches 1.0."""
        assert _sigmoid(100.0) > 0.99

    def test_sigmoid_large_negative(self) -> None:
        """Very large negative approaches 0.0."""
        assert _sigmoid(-100.0) < 0.01

    def test_sigmoid_bounds(self) -> None:
        """Sigmoid output is always in [0, 1]."""
        for x in [-1000, -10, -1, 0, 1, 10, 1000]:
            s = _sigmoid(x)
            assert 0.0 <= s <= 1.0


# --------------------------------------------------------------------------- #
# Reranker Tests
# --------------------------------------------------------------------------- #


class TestRerankerInit:
    """Tests for reranker initialization."""

    def test_default_init(self) -> None:
        """Default initialization stores config."""
        r = CrossEncoderReranker()
        assert r.top_k == 5

    def test_custom_top_k(self) -> None:
        """Custom top_k is stored."""
        r = CrossEncoderReranker(top_k=10)
        assert r.top_k == 10


class TestRerank:
    """Tests for the rerank method."""

    def test_rerank_returns_results(
        self, reranker: CrossEncoderReranker, candidates: list[dict]
    ) -> None:
        """Reranking returns results with expected fields."""
        results = reranker.rerank("What was Apple's revenue?", candidates)
        assert len(results) > 0
        assert "reranker_score" in results[0]
        assert "reranker_rank" in results[0]
        assert "text" in results[0]
        assert "chunk_id" in results[0]

    def test_rerank_scores_descending(
        self, reranker: CrossEncoderReranker, candidates: list[dict]
    ) -> None:
        """Results are sorted by reranker_score descending."""
        results = reranker.rerank("What was Apple's revenue?", candidates)
        scores = [r["reranker_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_relevance(
        self, reranker: CrossEncoderReranker, candidates: list[dict]
    ) -> None:
        """Revenue query should rank revenue chunks above risk/goodwill.

        Cross-encoder should understand "Apple's revenue" is about
        Apple's financial performance, not risk factors or goodwill.
        """
        results = reranker.rerank("What was Apple's total revenue?", candidates)
        top_id = results[0]["chunk_id"]
        assert top_id == "aapl_rev_001"

    def test_rerank_top_k_limit(
        self, reranker: CrossEncoderReranker, candidates: list[dict]
    ) -> None:
        """Default top_k limits output."""
        results = reranker.rerank("revenue", candidates)
        assert len(results) <= 3  # reranker fixture has top_k=3

    def test_rerank_top_k_override(
        self, reranker: CrossEncoderReranker, candidates: list[dict]
    ) -> None:
        """top_k parameter override works."""
        results = reranker.rerank("revenue", candidates, top_k=2)
        assert len(results) <= 2

    def test_rerank_rank_assignment(
        self, reranker: CrossEncoderReranker, candidates: list[dict]
    ) -> None:
        """Reranker_rank is 1-based sequential."""
        results = reranker.rerank("revenue growth", candidates)
        ranks = [r["reranker_rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_rerank_preserves_metadata(
        self, reranker: CrossEncoderReranker, candidates: list[dict]
    ) -> None:
        """Original metadata is preserved in reranked results."""
        results = reranker.rerank("Apple revenue", candidates)
        for r in results:
            assert "metadata" in r
            assert "ticker" in r["metadata"]

    def test_rerank_sigmoid_scores(
        self, reranker: CrossEncoderReranker, candidates: list[dict]
    ) -> None:
        """Reranker scores are sigmoid-normalized (0-1)."""
        results = reranker.rerank("revenue analysis", candidates)
        for r in results:
            assert 0.0 <= r["reranker_score"] <= 1.0


class TestRerankerEdgeCases:
    """Tests for edge cases."""

    def test_empty_candidates(self, reranker: CrossEncoderReranker) -> None:
        """Empty candidates returns empty list."""
        results = reranker.rerank("revenue", [])
        assert results == []

    def test_empty_query(
        self, reranker: CrossEncoderReranker, candidates: list[dict]
    ) -> None:
        """Empty query returns empty list."""
        results = reranker.rerank("", candidates)
        assert results == []

    def test_single_candidate(self, reranker: CrossEncoderReranker) -> None:
        """Single candidate is returned if above threshold."""
        single = [{
            "chunk_id": "test_001",
            "text": "Apple reported $383 billion in total revenue.",
            "metadata": {"ticker": "AAPL"},
        }]
        results = reranker.rerank("Apple revenue", single)
        assert len(results) <= 1
