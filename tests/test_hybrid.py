"""Tests for hybrid retrieval with Reciprocal Rank Fusion.

Tests cover:
- RRF fusion algorithm (score calculation, deduplication, ordering)
- Multi-query expansion (financial term synonyms)
- HyDE passthrough stub
- HybridRetriever end-to-end (with mocked stores)
- Metadata filtering through hybrid layer
- Edge cases (empty results, single source, all duplicates)

Both ChromaStore and BM25Index are mocked to keep tests fast
and focused on fusion logic, not embedding/tokenization.
"""

from unittest.mock import MagicMock, patch

import pytest

from finrag.retrieval.hybrid import (
    DEFAULT_CANDIDATES_PER_RETRIEVER,
    FINANCIAL_SYNONYMS,
    RRF_K,
    HybridRetriever,
    expand_financial_query,
    hyde_passthrough,
    reciprocal_rank_fusion,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_result(chunk_id: str, score: float = 0.0, distance: float = 0.0) -> dict:
    """Create a minimal result dict for testing."""
    result = {
        "chunk_id": chunk_id,
        "text": f"Text for {chunk_id}",
        "metadata": {
            "ticker": chunk_id.split("_")[0].upper(),
            "company_name": "Test Corp",
            "form_type": "10-K",
            "filing_date": "2024-10-31",
            "section_name": "Item 7 - MD&A",
            "chunk_index": 0,
            "total_chunks_in_section": 5,
            "token_count": 40,
        },
    }
    if score:
        result["score"] = score
    if distance:
        result["distance"] = distance
    return result


def _make_dense_results(*chunk_ids: str) -> list[dict]:
    """Create a ranked list of dense (vector) results."""
    return [_make_result(cid, distance=0.1 * (i + 1)) for i, cid in enumerate(chunk_ids)]


def _make_sparse_results(*chunk_ids: str) -> list[dict]:
    """Create a ranked list of sparse (BM25) results."""
    return [_make_result(cid, score=10.0 - i) for i, cid in enumerate(chunk_ids)]


# --------------------------------------------------------------------------- #
# RRF Tests
# --------------------------------------------------------------------------- #


class TestReciprocalRankFusion:
    """Tests for the RRF fusion algorithm."""

    def test_single_list(self) -> None:
        """RRF with one list just assigns scores based on rank."""
        results = [_make_result("a"), _make_result("b"), _make_result("c")]
        fused = reciprocal_rank_fusion([results], k=60)
        assert len(fused) == 3
        # First item should have highest RRF score
        assert fused[0]["chunk_id"] == "a"
        assert fused[0]["rrf_score"] == pytest.approx(1.0 / 61)

    def test_two_lists_overlap_boosts(self) -> None:
        """Documents in both lists get higher scores than single-list docs."""
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("b"), _make_result("c")]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)

        # "b" appears in both lists, should have highest RRF score
        scores = {r["chunk_id"]: r["rrf_score"] for r in fused}
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_two_lists_no_overlap(self) -> None:
        """Non-overlapping lists produce separate entries."""
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("c"), _make_result("d")]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        assert len(fused) == 4
        chunk_ids = {r["chunk_id"] for r in fused}
        assert chunk_ids == {"a", "b", "c", "d"}

    def test_score_ordering(self) -> None:
        """Fused results are sorted by RRF score descending."""
        dense = [_make_result("a"), _make_result("b"), _make_result("c")]
        sparse = [_make_result("c"), _make_result("b"), _make_result("a")]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_sources_tracked(self) -> None:
        """Each result tracks which sources found it."""
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("b"), _make_result("c")]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        sources = {r["chunk_id"]: r["retrieval_sources"] for r in fused}

        assert sources["a"] == ["dense"]
        assert sorted(sources["b"]) == ["dense", "sparse"]
        assert sources["c"] == ["sparse"]

    def test_removes_source_specific_scores(self) -> None:
        """RRF removes 'score' and 'distance' fields from results."""
        dense = [_make_result("a", distance=0.1)]
        sparse = [_make_result("b", score=5.0)]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        for result in fused:
            assert "score" not in result
            assert "distance" not in result
            assert "rrf_score" in result

    def test_empty_lists(self) -> None:
        """RRF with empty lists returns empty result."""
        fused = reciprocal_rank_fusion([[], []], k=60)
        assert fused == []

    def test_one_empty_list(self) -> None:
        """RRF with one empty list still works."""
        dense = [_make_result("a")]
        fused = reciprocal_rank_fusion([dense, []], k=60)
        assert len(fused) == 1
        assert fused[0]["chunk_id"] == "a"

    def test_custom_k_value(self) -> None:
        """Lower k gives more weight to top-ranked documents."""
        dense = [_make_result("a"), _make_result("b")]

        fused_low_k = reciprocal_rank_fusion([dense], k=1)
        fused_high_k = reciprocal_rank_fusion([dense], k=100)

        # With low k, the score difference between rank 1 and 2 is larger
        diff_low = fused_low_k[0]["rrf_score"] - fused_low_k[1]["rrf_score"]
        diff_high = fused_high_k[0]["rrf_score"] - fused_high_k[1]["rrf_score"]
        assert diff_low > diff_high

    def test_rrf_score_formula(self) -> None:
        """Verify RRF score matches the expected formula."""
        dense = [_make_result("a")]
        sparse = [_make_result("a")]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)

        # "a" is rank 1 in both lists: 1/(60+1) + 1/(60+1)
        expected = 2.0 / 61
        assert fused[0]["rrf_score"] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Multi-Query Expansion Tests
# --------------------------------------------------------------------------- #


class TestExpandFinancialQuery:
    """Tests for rule-based multi-query expansion."""

    def test_always_includes_original(self) -> None:
        """Original query is always the first variation."""
        result = expand_financial_query("what is the company doing")
        assert result[0] == "what is the company doing"

    def test_no_expansion_for_unknown_terms(self) -> None:
        """Queries without financial terms return only the original."""
        result = expand_financial_query("how is the weather today")
        assert len(result) == 1

    def test_revenue_expansion(self) -> None:
        """'revenue' triggers synonym expansion."""
        result = expand_financial_query("What was Apple's revenue?")
        assert len(result) > 1
        # Check at least one synonym made it
        lower_results = [r.lower() for r in result]
        has_synonym = any(
            "net sales" in r or "total revenue" in r
            for r in lower_results
        )
        assert has_synonym

    def test_eps_expansion(self) -> None:
        """'eps' triggers earnings per share expansion."""
        result = expand_financial_query("What is the current eps?")
        assert len(result) > 1
        lower_results = [r.lower() for r in result]
        has_synonym = any("earnings per share" in r for r in lower_results)
        assert has_synonym

    def test_no_duplicates(self) -> None:
        """Expanded queries are deduplicated."""
        result = expand_financial_query("revenue growth from revenue")
        # Check uniqueness
        lower_results = [r.lower().strip() for r in result]
        assert len(lower_results) == len(set(lower_results))

    def test_multiple_terms_expand(self) -> None:
        """Multiple financial terms in one query all expand."""
        result = expand_financial_query("revenue and margin analysis")
        # Both 'revenue' and 'margin' should trigger synonyms
        assert len(result) >= 3


# --------------------------------------------------------------------------- #
# HyDE Tests
# --------------------------------------------------------------------------- #


class TestHydePassthrough:
    """Tests for the HyDE no-op stub."""

    def test_returns_query_unchanged(self) -> None:
        """Passthrough returns the exact same string."""
        query = "What was Apple's free cash flow in Q3 2024?"
        assert hyde_passthrough(query) == query

    def test_empty_string(self) -> None:
        """Passthrough handles empty strings."""
        assert hyde_passthrough("") == ""


# --------------------------------------------------------------------------- #
# HybridRetriever Tests (mocked stores)
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_chroma() -> MagicMock:
    """Create a mock ChromaStore."""
    store = MagicMock()
    store.query.return_value = _make_dense_results("aapl_rev", "aapl_risk", "msft_rev")
    store.get_stats.return_value = {"total_chunks": 10, "collection_name": "test"}
    return store


@pytest.fixture
def mock_bm25() -> MagicMock:
    """Create a mock BM25Index."""
    index = MagicMock()
    index.query.return_value = _make_sparse_results("aapl_rev", "tsla_rev", "aapl_risk")
    index.get_stats.return_value = {"total_documents": 10, "is_built": True}
    return index


@pytest.fixture
def retriever(mock_chroma: MagicMock, mock_bm25: MagicMock) -> HybridRetriever:
    """Create a HybridRetriever with mocked stores."""
    return HybridRetriever(
        chroma_store=mock_chroma,
        bm25_index=mock_bm25,
        rrf_k=60,
        candidates_per_retriever=20,
    )


class TestHybridRetrieverInit:
    """Tests for retriever initialization."""

    def test_default_config(self, mock_chroma: MagicMock, mock_bm25: MagicMock) -> None:
        """Default config uses standard RRF k and candidates."""
        r = HybridRetriever(chroma_store=mock_chroma, bm25_index=mock_bm25)
        stats = r.get_stats()
        assert stats["rrf_k"] == RRF_K
        assert stats["candidates_per_retriever"] == DEFAULT_CANDIDATES_PER_RETRIEVER

    def test_custom_config(self, mock_chroma: MagicMock, mock_bm25: MagicMock) -> None:
        """Custom k and candidates are stored."""
        r = HybridRetriever(
            chroma_store=mock_chroma,
            bm25_index=mock_bm25,
            rrf_k=30,
            candidates_per_retriever=50,
        )
        stats = r.get_stats()
        assert stats["rrf_k"] == 30
        assert stats["candidates_per_retriever"] == 50

    def test_custom_multi_query_fn(
        self, mock_chroma: MagicMock, mock_bm25: MagicMock
    ) -> None:
        """Custom multi-query function is used when provided."""
        custom_fn = MagicMock(return_value=["q1", "q2"])
        r = HybridRetriever(
            chroma_store=mock_chroma,
            bm25_index=mock_bm25,
            multi_query_fn=custom_fn,
        )
        r.retrieve("test query", use_multi_query=True)
        custom_fn.assert_called_once_with("test query")

    def test_custom_hyde_fn(
        self, mock_chroma: MagicMock, mock_bm25: MagicMock
    ) -> None:
        """Custom HyDE function is used when provided."""
        custom_hyde = MagicMock(return_value="hypothetical document about revenue")
        r = HybridRetriever(
            chroma_store=mock_chroma,
            bm25_index=mock_bm25,
            hyde_fn=custom_hyde,
        )
        r.retrieve("what is revenue", use_hyde=True)
        custom_hyde.assert_called_once_with("what is revenue")


class TestHybridRetrieve:
    """Tests for the main retrieve method."""

    def test_basic_retrieve(self, retriever: HybridRetriever) -> None:
        """Basic retrieve returns fused results with expected fields."""
        results = retriever.retrieve("Apple revenue", n_results=5)
        assert len(results) > 0
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "metadata" in r
            assert "rrf_score" in r
            assert "retrieval_sources" in r

    def test_overlap_boosted(
        self, retriever: HybridRetriever
    ) -> None:
        """Chunks found by both stores rank higher.

        mock_chroma returns: aapl_rev, aapl_risk, msft_rev
        mock_bm25 returns: aapl_rev, tsla_rev, aapl_risk

        aapl_rev is rank 1 in both -> highest RRF score.
        aapl_risk is rank 2 in dense, rank 3 in sparse -> second highest.
        """
        results = retriever.retrieve("Apple revenue", n_results=10)
        assert results[0]["chunk_id"] == "aapl_rev"

    def test_both_stores_called(
        self,
        retriever: HybridRetriever,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Both BM25 and vector search are called."""
        retriever.retrieve("test query")
        mock_chroma.query.assert_called_once()
        mock_bm25.query.assert_called_once()

    def test_n_results_limit(self, retriever: HybridRetriever) -> None:
        """n_results caps the output length."""
        results = retriever.retrieve("test", n_results=2)
        assert len(results) <= 2

    def test_empty_query(self, retriever: HybridRetriever) -> None:
        """Empty query returns empty results."""
        results = retriever.retrieve("")
        assert results == []

    def test_whitespace_query(self, retriever: HybridRetriever) -> None:
        """Whitespace-only query returns empty results."""
        results = retriever.retrieve("   ")
        assert results == []

    def test_metadata_filter_passed_through(
        self,
        retriever: HybridRetriever,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Metadata filter is forwarded to both stores."""
        where = {"ticker": "AAPL"}
        retriever.retrieve("revenue", where=where)

        # Both stores should receive the filter
        mock_chroma.query.assert_called_once()
        mock_bm25.query.assert_called_once()
        assert mock_chroma.query.call_args.kwargs["where"] == where
        assert mock_bm25.query.call_args.kwargs["where"] == where

    def test_compound_filter_passed_through(
        self,
        retriever: HybridRetriever,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Compound $and filter is forwarded to both stores."""
        where = {"$and": [{"ticker": "AAPL"}, {"form_type": "10-K"}]}
        retriever.retrieve("revenue", where=where)

        assert mock_chroma.query.call_args.kwargs["where"] == where
        assert mock_bm25.query.call_args.kwargs["where"] == where

    def test_scores_are_descending(self, retriever: HybridRetriever) -> None:
        """Results are sorted by RRF score descending."""
        results = retriever.retrieve("test", n_results=10)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestMultiQueryRetrieve:
    """Tests for multi-query retrieval through HybridRetriever."""

    def test_multi_query_calls_stores_per_variant(
        self,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Multi-query calls stores once per query variant."""
        # Use a custom fn that returns 3 variants
        custom_fn = MagicMock(return_value=["q1", "q2", "q3"])
        r = HybridRetriever(
            chroma_store=mock_chroma,
            bm25_index=mock_bm25,
            multi_query_fn=custom_fn,
        )
        r.retrieve("original query", use_multi_query=True)

        # Both stores should be called 3 times (once per variant)
        assert mock_chroma.query.call_count == 3
        assert mock_bm25.query.call_count == 3

    def test_multi_query_deduplicates(
        self,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Multi-query deduplicates results within each source."""
        # Both variants return the same results
        mock_chroma.query.return_value = _make_dense_results("aapl_rev")
        mock_bm25.query.return_value = _make_sparse_results("aapl_rev")

        custom_fn = MagicMock(return_value=["q1", "q2"])
        r = HybridRetriever(
            chroma_store=mock_chroma,
            bm25_index=mock_bm25,
            multi_query_fn=custom_fn,
        )
        results = r.retrieve("test", use_multi_query=True)

        # Should only have one result despite 2 variants returning same chunk
        assert len(results) == 1
        assert results[0]["chunk_id"] == "aapl_rev"

    def test_multi_query_disabled_by_default(
        self,
        retriever: HybridRetriever,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Multi-query is off by default (single query variant)."""
        retriever.retrieve("test query")
        # Only one call per store
        assert mock_chroma.query.call_count == 1
        assert mock_bm25.query.call_count == 1


class TestHyDERetrieve:
    """Tests for HyDE through HybridRetriever."""

    def test_hyde_transforms_dense_query(
        self,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """HyDE transforms the query for vector search only."""
        custom_hyde = MagicMock(return_value="hypothetical answer about revenue")
        r = HybridRetriever(
            chroma_store=mock_chroma,
            bm25_index=mock_bm25,
            hyde_fn=custom_hyde,
        )
        r.retrieve("what is revenue", use_hyde=True)

        # Dense search should use HyDE query
        dense_query = mock_chroma.query.call_args.kwargs.get(
            "query_text", mock_chroma.query.call_args.args[0] if mock_chroma.query.call_args.args else None
        )
        assert dense_query == "hypothetical answer about revenue"

        # Sparse search should use original query
        sparse_query = mock_bm25.query.call_args.kwargs.get(
            "query_text", mock_bm25.query.call_args.args[0] if mock_bm25.query.call_args.args else None
        )
        assert sparse_query == "what is revenue"

    def test_hyde_disabled_by_default(
        self,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """HyDE is off by default (same query for both stores)."""
        r = HybridRetriever(chroma_store=mock_chroma, bm25_index=mock_bm25)
        r.retrieve("test query")

        dense_query = mock_chroma.query.call_args.kwargs.get("query_text")
        sparse_query = mock_bm25.query.call_args.kwargs.get("query_text")
        assert dense_query == sparse_query == "test query"


class TestSingleSourceRetrieve:
    """Tests for single-source retrieval bypasses."""

    def test_dense_only(
        self,
        retriever: HybridRetriever,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Dense-only retrieval skips BM25."""
        results = retriever.retrieve_dense_only("test query", n_results=3)
        mock_chroma.query.assert_called_once()
        mock_bm25.query.assert_not_called()

    def test_sparse_only(
        self,
        retriever: HybridRetriever,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Sparse-only retrieval skips vector search."""
        results = retriever.retrieve_sparse_only("test query", n_results=3)
        mock_bm25.query.assert_called_once()
        mock_chroma.query.assert_not_called()

    def test_dense_only_with_filter(
        self,
        retriever: HybridRetriever,
        mock_chroma: MagicMock,
    ) -> None:
        """Dense-only passes metadata filter through."""
        where = {"ticker": "AAPL"}
        retriever.retrieve_dense_only("test", where=where)
        assert mock_chroma.query.call_args.kwargs["where"] == where

    def test_sparse_only_with_filter(
        self,
        retriever: HybridRetriever,
        mock_bm25: MagicMock,
    ) -> None:
        """Sparse-only passes metadata filter through."""
        where = {"ticker": "MSFT"}
        retriever.retrieve_sparse_only("test", where=where)
        assert mock_bm25.query.call_args.kwargs["where"] == where


class TestEdgeCases:
    """Tests for edge cases and failure scenarios."""

    def test_both_stores_return_empty(
        self,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Both stores returning nothing produces empty results."""
        mock_chroma.query.return_value = []
        mock_bm25.query.return_value = []

        r = HybridRetriever(chroma_store=mock_chroma, bm25_index=mock_bm25)
        results = r.retrieve("nonexistent topic")
        assert results == []

    def test_only_dense_has_results(
        self,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Only vector search returning results still works."""
        mock_chroma.query.return_value = _make_dense_results("aapl_rev")
        mock_bm25.query.return_value = []

        r = HybridRetriever(chroma_store=mock_chroma, bm25_index=mock_bm25)
        results = r.retrieve("Apple revenue")
        assert len(results) == 1
        assert results[0]["retrieval_sources"] == ["dense"]

    def test_only_sparse_has_results(
        self,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Only BM25 returning results still works."""
        mock_chroma.query.return_value = []
        mock_bm25.query.return_value = _make_sparse_results("aapl_eps")

        r = HybridRetriever(chroma_store=mock_chroma, bm25_index=mock_bm25)
        results = r.retrieve("diluted earnings per share")
        assert len(results) == 1
        assert results[0]["retrieval_sources"] == ["sparse"]

    def test_all_same_chunk(
        self,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Both stores returning the same single chunk deduplicates correctly."""
        mock_chroma.query.return_value = _make_dense_results("aapl_rev")
        mock_bm25.query.return_value = _make_sparse_results("aapl_rev")

        r = HybridRetriever(chroma_store=mock_chroma, bm25_index=mock_bm25)
        results = r.retrieve("Apple revenue")
        assert len(results) == 1
        assert sorted(results[0]["retrieval_sources"]) == ["dense", "sparse"]


class TestStats:
    """Tests for stats reporting."""

    def test_stats_includes_components(self, retriever: HybridRetriever) -> None:
        """Stats include config and sub-component stats."""
        stats = retriever.get_stats()
        assert "rrf_k" in stats
        assert "candidates_per_retriever" in stats
        assert "bm25_stats" in stats
        assert "chroma_stats" in stats
