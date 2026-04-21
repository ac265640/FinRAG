"""Tests for BM25 sparse retrieval index.

Tests cover:
- Tokenizer behavior (case, decimals, punctuation)
- Index building and document counting
- Keyword search relevance (exact terms rank higher)
- Metadata-filtered search (ticker, section, compound)
- Index serialization (save/load round-trip)
- Edge cases (empty index, empty query, no matches)
- Stats reporting
"""

from pathlib import Path

import pytest

from finrag.ingestion.chunker import Chunk, ChunkMetadata
from finrag.retrieval.bm25_index import BM25Index, tokenize


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def index() -> BM25Index:
    """Create a fresh BM25 index."""
    return BM25Index()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Financial chunks covering different tickers and topics."""
    return [
        Chunk(
            text=(
                "Apple Inc. reported total revenue of $383.3 billion for fiscal year 2024. "
                "Services revenue increased 13 percent year over year driven by the App Store."
            ),
            metadata=ChunkMetadata(
                chunk_id="aapl_rev_001",
                ticker="AAPL",
                company_name="Apple Inc.",
                form_type="10-K",
                filing_date="2024-10-31",
                section_name="Item 7 - MD&A",
                chunk_index=0,
                total_chunks_in_section=5,
                token_count=40,
            ),
        ),
        Chunk(
            text=(
                "The Company's diluted earnings per share was $6.08 for fiscal 2024, "
                "compared to $6.13 for fiscal 2023. Net income was $94.8 billion."
            ),
            metadata=ChunkMetadata(
                chunk_id="aapl_eps_001",
                ticker="AAPL",
                company_name="Apple Inc.",
                form_type="10-K",
                filing_date="2024-10-31",
                section_name="Item 7 - MD&A",
                chunk_index=1,
                total_chunks_in_section=5,
                token_count=38,
            ),
        ),
        Chunk(
            text=(
                "Apple faces risks from supply chain disruptions, geopolitical tensions, "
                "and increasing regulatory scrutiny in the European Union and China."
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
                token_count=35,
            ),
        ),
        Chunk(
            text=(
                "Microsoft Corporation reported revenue of $245.1 billion. "
                "Azure cloud revenue grew 29 percent year over year. "
                "Operating income increased to $109.4 billion."
            ),
            metadata=ChunkMetadata(
                chunk_id="msft_rev_001",
                ticker="MSFT",
                company_name="Microsoft Corporation",
                form_type="10-K",
                filing_date="2024-07-31",
                section_name="Item 7 - MD&A",
                chunk_index=0,
                total_chunks_in_section=8,
                token_count=42,
            ),
        ),
        Chunk(
            text=(
                "Tesla reported automotive revenue of $82.4 billion. "
                "Vehicle deliveries totaled 1.81 million units. "
                "Gross margin for the automotive segment was 18.2 percent."
            ),
            metadata=ChunkMetadata(
                chunk_id="tsla_rev_001",
                ticker="TSLA",
                company_name="Tesla, Inc.",
                form_type="10-K",
                filing_date="2025-01-29",
                section_name="Item 7 - MD&A",
                chunk_index=0,
                total_chunks_in_section=6,
                token_count=44,
            ),
        ),
        Chunk(
            text=(
                "The goodwill impairment test resulted in no impairment charge "
                "for the reporting period. Intangible assets were $5.2 billion."
            ),
            metadata=ChunkMetadata(
                chunk_id="aapl_goodwill_001",
                ticker="AAPL",
                company_name="Apple Inc.",
                form_type="10-K",
                filing_date="2024-10-31",
                section_name="Item 8 - Financial Statements",
                chunk_index=0,
                total_chunks_in_section=20,
                token_count=32,
            ),
        ),
    ]


@pytest.fixture
def built_index(index: BM25Index, sample_chunks: list[Chunk]) -> BM25Index:
    """Return an index pre-loaded with sample chunks."""
    index.add_chunks(sample_chunks)
    return index


# --------------------------------------------------------------------------- #
# Tokenizer Tests
# --------------------------------------------------------------------------- #


class TestTokenizer:
    """Tests for the financial-aware tokenizer."""

    def test_lowercase(self) -> None:
        """Tokens are lowercased."""
        tokens = tokenize("Apple AAPL Revenue")
        assert tokens == ["apple", "aapl", "revenue"]

    def test_preserves_decimals(self) -> None:
        """Decimal numbers stay as one token."""
        tokens = tokenize("Margin was 46.2 percent")
        assert "46.2" in tokens

    def test_strips_punctuation(self) -> None:
        """Punctuation is removed."""
        tokens = tokenize("revenue, of $383.3 billion.")
        assert "revenue" in tokens
        assert "383.3" in tokens
        assert "billion" in tokens
        assert "," not in tokens
        assert "$" not in tokens

    def test_empty_string(self) -> None:
        """Empty string produces empty list."""
        assert tokenize("") == []

    def test_numbers_preserved(self) -> None:
        """Plain numbers are kept."""
        tokens = tokenize("Q3 2024 fiscal year")
        assert "q3" in tokens
        assert "2024" in tokens


# --------------------------------------------------------------------------- #
# Index Tests
# --------------------------------------------------------------------------- #


class TestBM25Init:
    """Tests for index initialization."""

    def test_empty_index(self, index: BM25Index) -> None:
        """New index has zero documents."""
        assert index.count == 0
        assert not index.is_built

    def test_custom_params(self) -> None:
        """Custom k1 and b are stored."""
        idx = BM25Index(k1=2.0, b=0.5)
        stats = idx.get_stats()
        assert stats["k1"] == 2.0
        assert stats["b"] == 0.5


class TestAddChunks:
    """Tests for adding documents to the index."""

    def test_add_chunks(self, index: BM25Index, sample_chunks: list[Chunk]) -> None:
        """Adding chunks builds the index."""
        count = index.add_chunks(sample_chunks)
        assert count == 6
        assert index.is_built

    def test_add_empty_list(self, index: BM25Index) -> None:
        """Adding empty list is a no-op."""
        count = index.add_chunks([])
        assert count == 0
        assert not index.is_built

    def test_incremental_add(self, index: BM25Index, sample_chunks: list[Chunk]) -> None:
        """Adding chunks in batches accumulates."""
        index.add_chunks(sample_chunks[:3])
        assert index.count == 3
        index.add_chunks(sample_chunks[3:])
        assert index.count == 6


class TestQuery:
    """Tests for BM25 search."""

    def test_basic_query(self, built_index: BM25Index) -> None:
        """Basic query returns results with expected fields."""
        results = built_index.query("diluted earnings", n_results=3)
        assert len(results) > 0
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]
        assert "chunk_id" in results[0]

    def test_exact_term_relevance(self, built_index: BM25Index) -> None:
        """Exact financial terms should find the right chunk.

        'diluted earnings per share' should rank the EPS chunk highest.
        This is exactly where BM25 beats vector search.
        """
        results = built_index.query("diluted earnings per share", n_results=3)
        assert results[0]["chunk_id"] == "aapl_eps_001"

    def test_goodwill_impairment(self, built_index: BM25Index) -> None:
        """Specific accounting term 'goodwill impairment' finds correct chunk."""
        results = built_index.query("goodwill impairment", n_results=3)
        assert results[0]["chunk_id"] == "aapl_goodwill_001"

    def test_ticker_specific_query(self, built_index: BM25Index) -> None:
        """Query containing ticker symbol boosts that company."""
        results = built_index.query("TSLA automotive deliveries", n_results=3)
        assert results[0]["metadata"]["ticker"] == "TSLA"

    def test_scores_are_descending(self, built_index: BM25Index) -> None:
        """Results are sorted by score descending."""
        results = built_index.query("revenue billion", n_results=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_n_results_limit(self, built_index: BM25Index) -> None:
        """n_results caps the number of returned results."""
        results = built_index.query("revenue", n_results=2)
        assert len(results) <= 2

    def test_query_empty_index(self, index: BM25Index) -> None:
        """Querying empty index returns empty list."""
        results = index.query("revenue")
        assert results == []

    def test_query_empty_string(self, built_index: BM25Index) -> None:
        """Empty query string returns empty list."""
        results = built_index.query("")
        assert results == []


class TestFilteredQuery:
    """Tests for metadata-filtered BM25 search."""

    def test_filter_by_ticker(self, built_index: BM25Index) -> None:
        """Ticker filter only returns matching chunks."""
        results = built_index.query(
            "revenue",
            n_results=10,
            where={"ticker": "AAPL"},
        )
        for r in results:
            assert r["metadata"]["ticker"] == "AAPL"

    def test_filter_by_section(self, built_index: BM25Index) -> None:
        """Section filter narrows to specific section."""
        results = built_index.query(
            "risks regulatory",
            n_results=10,
            where={"section_name": "Item 1A - Risk Factors"},
        )
        for r in results:
            assert r["metadata"]["section_name"] == "Item 1A - Risk Factors"

    def test_compound_filter(self, built_index: BM25Index) -> None:
        """Compound $and filter works."""
        results = built_index.query(
            "revenue",
            n_results=10,
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

    def test_filter_no_matches(self, built_index: BM25Index) -> None:
        """Filter that matches nothing returns empty list."""
        results = built_index.query(
            "revenue",
            where={"ticker": "NONEXISTENT"},
        )
        assert len(results) == 0


class TestSerialization:
    """Tests for save/load round-trip."""

    def test_save_and_load(
        self, built_index: BM25Index, tmp_path: Path
    ) -> None:
        """Index survives save/load cycle with same results."""
        save_path = tmp_path / "bm25.pkl"
        built_index.save(save_path)
        assert save_path.exists()

        loaded = BM25Index.load(save_path)
        assert loaded.count == built_index.count
        assert loaded.is_built

        # Same query should produce same results
        orig_results = built_index.query("diluted earnings per share", n_results=3)
        loaded_results = loaded.query("diluted earnings per share", n_results=3)

        assert len(orig_results) == len(loaded_results)
        assert orig_results[0]["chunk_id"] == loaded_results[0]["chunk_id"]

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            BM25Index.load(tmp_path / "nope.pkl")


class TestStats:
    """Tests for stats reporting."""

    def test_stats_empty(self, index: BM25Index) -> None:
        """Stats on empty index."""
        stats = index.get_stats()
        assert stats["total_documents"] == 0
        assert not stats["is_built"]

    def test_stats_with_data(self, built_index: BM25Index) -> None:
        """Stats with data shows tickers and averages."""
        stats = built_index.get_stats()
        assert stats["total_documents"] == 6
        assert stats["is_built"]
        assert "AAPL" in stats["unique_tickers"]
        assert "MSFT" in stats["unique_tickers"]
        assert "TSLA" in stats["unique_tickers"]
        assert stats["avg_tokens_per_doc"] > 0
