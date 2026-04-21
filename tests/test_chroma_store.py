"""Tests for ChromaDB vector store.

Tests cover:
- Store initialization and collection creation
- Chunk embedding and upsertion
- Semantic search (unfiltered and filtered)
- Idempotent upserts (same chunks don't duplicate)
- Deletion by ticker
- Collection reset
- Stats reporting
- Edge cases (empty queries, no results)
"""

from pathlib import Path

import pytest

from finrag.ingestion.chunker import Chunk, ChunkMetadata
from finrag.vectorstore.chroma_store import ChromaStore


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    """Create a fresh ChromaStore with temp directory."""
    return ChromaStore(
        persist_dir=tmp_path / "chroma_test",
        collection_name="test_filings",
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks with realistic financial content."""
    return [
        Chunk(
            text=(
                "Apple Inc. reported total revenue of $383.3 billion for fiscal year 2024. "
                "This represents a 2% increase compared to fiscal 2023. Services revenue "
                "increased 13% driven by the App Store and advertising."
            ),
            metadata=ChunkMetadata(
                chunk_id="aapl_revenue_001",
                ticker="AAPL",
                company_name="Apple Inc.",
                form_type="10-K",
                filing_date="2024-10-31",
                section_name="Item 7 - MD&A",
                chunk_index=0,
                total_chunks_in_section=5,
                token_count=52,
            ),
        ),
        Chunk(
            text=(
                "The Company faces risks related to global economic conditions, "
                "supply chain disruptions, and regulatory changes in key markets. "
                "Competition in the technology industry remains intense."
            ),
            metadata=ChunkMetadata(
                chunk_id="aapl_risk_001",
                ticker="AAPL",
                company_name="Apple Inc.",
                form_type="10-K",
                filing_date="2024-10-31",
                section_name="Item 1A - Risk Factors",
                chunk_index=0,
                total_chunks_in_section=10,
                token_count=45,
            ),
        ),
        Chunk(
            text=(
                "Microsoft Corporation reported revenue of $245.1 billion for fiscal 2024. "
                "Cloud computing revenue from Azure grew 29% year-over-year. "
                "Operating income increased to $109.4 billion."
            ),
            metadata=ChunkMetadata(
                chunk_id="msft_revenue_001",
                ticker="MSFT",
                company_name="Microsoft Corporation",
                form_type="10-K",
                filing_date="2024-07-31",
                section_name="Item 7 - MD&A",
                chunk_index=0,
                total_chunks_in_section=8,
                token_count=48,
            ),
        ),
        Chunk(
            text=(
                "Apple's gross margin was 46.2% for fiscal 2024, compared to 44.1% "
                "for fiscal 2023. The increase was driven by a favorable shift in "
                "product mix toward higher-margin Services."
            ),
            metadata=ChunkMetadata(
                chunk_id="aapl_margin_001",
                ticker="AAPL",
                company_name="Apple Inc.",
                form_type="10-K",
                filing_date="2024-10-31",
                section_name="Item 7 - MD&A",
                chunk_index=1,
                total_chunks_in_section=5,
                token_count=47,
            ),
        ),
        Chunk(
            text=(
                "Tesla reported automotive revenue of $82.4 billion for fiscal 2024. "
                "Vehicle deliveries totaled 1.81 million units. Energy generation and "
                "storage revenue grew 67% year over year."
            ),
            metadata=ChunkMetadata(
                chunk_id="tsla_revenue_001",
                ticker="TSLA",
                company_name="Tesla, Inc.",
                form_type="10-K",
                filing_date="2025-01-29",
                section_name="Item 7 - MD&A",
                chunk_index=0,
                total_chunks_in_section=6,
                token_count=50,
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestStoreInit:
    """Tests for store initialization."""

    def test_init_creates_collection(self, store: ChromaStore) -> None:
        """Store creates a collection on init."""
        assert store.count == 0
        assert store.collection_name == "test_filings"

    def test_init_creates_persist_dir(self, tmp_path: Path) -> None:
        """Store creates the persist directory if it doesn't exist."""
        persist_dir = tmp_path / "new_dir" / "chroma"
        ChromaStore(persist_dir=persist_dir, collection_name="test")
        assert persist_dir.exists()

    def test_init_idempotent(self, tmp_path: Path) -> None:
        """Creating store twice with same params doesn't error."""
        persist_dir = tmp_path / "chroma"
        s1 = ChromaStore(persist_dir=persist_dir, collection_name="test")
        s2 = ChromaStore(persist_dir=persist_dir, collection_name="test")
        assert s1.count == s2.count == 0


class TestAddChunks:
    """Tests for embedding and upserting chunks."""

    def test_add_chunks(self, store: ChromaStore, sample_chunks: list[Chunk]) -> None:
        """Adding chunks increases collection count."""
        added = store.add_chunks(sample_chunks)
        assert added == 5
        assert store.count == 5

    def test_add_empty_list(self, store: ChromaStore) -> None:
        """Adding empty list returns 0 and doesn't error."""
        added = store.add_chunks([])
        assert added == 0
        assert store.count == 0

    def test_upsert_idempotent(self, store: ChromaStore, sample_chunks: list[Chunk]) -> None:
        """Adding the same chunks twice doesn't duplicate them."""
        store.add_chunks(sample_chunks)
        store.add_chunks(sample_chunks)  # Second upsert
        assert store.count == 5  # Still 5, not 10


class TestQuery:
    """Tests for semantic search."""

    def test_basic_query(self, store: ChromaStore, sample_chunks: list[Chunk]) -> None:
        """Basic query returns relevant results."""
        store.add_chunks(sample_chunks)
        results = store.query("What was Apple's revenue?", n_results=3)
        assert len(results) == 3
        # Results should have expected fields
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]
        assert "chunk_id" in results[0]

    def test_query_relevance(self, store: ChromaStore, sample_chunks: list[Chunk]) -> None:
        """Revenue query should rank revenue chunks higher than risk chunks."""
        store.add_chunks(sample_chunks)
        results = store.query("total revenue and financial performance", n_results=5)
        # Top result should be about revenue, not risk factors
        top_text = results[0]["text"].lower()
        assert "revenue" in top_text

    def test_query_with_ticker_filter(
        self, store: ChromaStore, sample_chunks: list[Chunk]
    ) -> None:
        """Filtered query only returns chunks matching the filter."""
        store.add_chunks(sample_chunks)
        results = store.query(
            "revenue growth",
            n_results=5,
            where={"ticker": "AAPL"},
        )
        for r in results:
            assert r["metadata"]["ticker"] == "AAPL"

    def test_query_with_section_filter(
        self, store: ChromaStore, sample_chunks: list[Chunk]
    ) -> None:
        """Section filter narrows results to specific section."""
        store.add_chunks(sample_chunks)
        results = store.query(
            "risks and challenges",
            n_results=5,
            where={"section_name": "Item 1A - Risk Factors"},
        )
        for r in results:
            assert r["metadata"]["section_name"] == "Item 1A - Risk Factors"

    def test_query_with_compound_filter(
        self, store: ChromaStore, sample_chunks: list[Chunk]
    ) -> None:
        """Compound filter with $and operator works."""
        store.add_chunks(sample_chunks)
        results = store.query(
            "financial results",
            n_results=5,
            where={
                "$and": [
                    {"ticker": "AAPL"},
                    {"section_name": "Item 7 - MD&A"},
                ]
            },
        )
        for r in results:
            assert r["metadata"]["ticker"] == "AAPL"
            assert r["metadata"]["section_name"] == "Item 7 - MD&A"

    def test_query_no_results_with_impossible_filter(
        self, store: ChromaStore, sample_chunks: list[Chunk]
    ) -> None:
        """Filter that matches nothing returns empty list."""
        store.add_chunks(sample_chunks)
        results = store.query(
            "revenue",
            n_results=5,
            where={"ticker": "NONEXISTENT"},
        )
        assert len(results) == 0

    def test_query_n_results_limit(
        self, store: ChromaStore, sample_chunks: list[Chunk]
    ) -> None:
        """n_results limits the number of returned results."""
        store.add_chunks(sample_chunks)
        results = store.query("financial data", n_results=2)
        assert len(results) == 2


class TestDeleteAndReset:
    """Tests for deletion operations."""

    def test_delete_by_ticker(
        self, store: ChromaStore, sample_chunks: list[Chunk]
    ) -> None:
        """Deleting by ticker removes only that ticker's chunks."""
        store.add_chunks(sample_chunks)
        assert store.count == 5
        store.delete_by_ticker("AAPL")
        # AAPL had 3 chunks (revenue, risk, margin), MSFT had 1, TSLA had 1
        assert store.count == 2

    def test_reset(self, store: ChromaStore, sample_chunks: list[Chunk]) -> None:
        """Reset clears the entire collection."""
        store.add_chunks(sample_chunks)
        assert store.count == 5
        store.reset()
        assert store.count == 0


class TestStats:
    """Tests for stats reporting."""

    def test_stats_empty(self, store: ChromaStore) -> None:
        """Stats on empty collection."""
        stats = store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["collection_name"] == "test_filings"

    def test_stats_with_data(
        self, store: ChromaStore, sample_chunks: list[Chunk]
    ) -> None:
        """Stats with data shows tickers and form types."""
        store.add_chunks(sample_chunks)
        stats = store.get_stats()
        assert stats["total_chunks"] == 5
        assert "AAPL" in stats["unique_tickers"]
        assert "MSFT" in stats["unique_tickers"]
        assert "TSLA" in stats["unique_tickers"]
        assert "10-K" in stats["unique_form_types"]
