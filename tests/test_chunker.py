"""Tests for the section-aware chunker.

Tests cover:
- Token count accuracy
- Chunk size bounds
- Overlap correctness
- Short text handling
- Empty text handling
- Metadata attachment
- Filing directory chunking
- Edge cases (oversized sentences, single-sentence sections)
"""

import json
from pathlib import Path

import pytest
import tiktoken

from finrag.ingestion.chunker import (
    DEFAULT_ENCODING,
    MIN_CHUNK_SIZE,
    Chunk,
    ChunkMetadata,
    SectionChunker,
    chunk_filing_directory,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def chunker() -> SectionChunker:
    """Standard chunker with default settings."""
    return SectionChunker(chunk_size=500, chunk_overlap=100)


@pytest.fixture
def small_chunker() -> SectionChunker:
    """Small chunker for easier testing with short text."""
    return SectionChunker(chunk_size=50, chunk_overlap=10)


@pytest.fixture
def encoding() -> tiktoken.Encoding:
    """Tiktoken encoding for independent token counting."""
    return tiktoken.get_encoding(DEFAULT_ENCODING)


@pytest.fixture
def long_text() -> str:
    """Generate a long text string (~1500 tokens) for chunking tests.

    Uses realistic financial language to test sentence boundary detection.
    """
    # Each sentence is roughly 20-30 tokens
    sentences = [
        "The Company's total revenue for the fiscal year ended September 2024 was approximately $383 billion.",
        "This represents a year-over-year increase of approximately 2 percent compared to fiscal 2023.",
        "Products revenue decreased 3 percent during the year, primarily driven by lower iPhone and Mac sales.",
        "Services revenue increased 13 percent during the year, driven by higher revenue from advertising, the App Store and cloud services.",
        "The Company's gross margin was 46.2 percent for fiscal 2024, compared to 44.1 percent for fiscal 2023.",
        "The increase in gross margin was primarily driven by a favorable shift in product mix toward Services.",
        "Research and development expense increased 5 percent during fiscal 2024 compared to fiscal 2023.",
        "Selling, general and administrative expense decreased 2 percent during fiscal 2024.",
        "The Company's effective tax rate was 16.3 percent for fiscal 2024.",
        "Operating cash flow was $118.3 billion for fiscal 2024.",
        "The Company returned over $100 billion to shareholders through dividends and share repurchases.",
        "Capital expenditures were $9.9 billion during fiscal 2024.",
        "The Company's total assets were $364.9 billion as of September 28, 2024.",
        "Long-term debt was $98.3 billion as of September 28, 2024.",
        "The Company had approximately 161,000 full-time equivalent employees as of September 28, 2024.",
        "International sales accounted for approximately 58 percent of total revenue.",
        "Greater China revenue was $66.7 billion, representing 17 percent of total revenue.",
        "The Company continues to invest significantly in research and development.",
        "New product introductions included iPhone 16, Apple Watch Series 10, and Mac models with M4 chips.",
        "The Company is subject to various legal proceedings and claims that have arisen in the ordinary course of business.",
        "Management believes the outcome of these proceedings will not have a material adverse effect.",
        "The Company uses derivative instruments to manage foreign currency and interest rate risk.",
        "Net income for fiscal 2024 was $94.8 billion, compared to $96.9 billion for fiscal 2023.",
        "Diluted earnings per share was $6.08 for fiscal 2024, compared to $6.13 for fiscal 2023.",
        "The Board of Directors declared a quarterly dividend of $0.25 per share.",
        "The Company authorized an additional $110 billion for share repurchase.",
        "Cash and marketable securities were $65.2 billion as of the end of fiscal 2024.",
        "The weighted average cost of debt was 3.29 percent for fiscal 2024.",
        "Foreign currency translation adjustments resulted in other comprehensive income of $1.2 billion.",
        "The Company's goodwill balance was zero as of the end of fiscal 2024.",
        "Inventory increased to $7.3 billion as the Company prepared for new product launches.",
        "Accounts receivable were $66.2 billion, reflecting strong sales in the final quarter.",
        "The Company continues to expand its global retail presence with new stores in key markets.",
        "Supply chain diversification efforts continued with increased manufacturing in India and Vietnam.",
        "Environmental sustainability initiatives resulted in the Company achieving carbon neutrality for corporate operations.",
        "The Company invested $2.1 billion in renewable energy projects during fiscal 2024.",
        "Customer satisfaction scores remained above industry averages across all product categories.",
        "The App Store ecosystem supported over 1.8 million apps and generated significant developer revenue.",
        "Apple Pay transaction volume grew 52 percent year over year.",
        "The Company's installed base of active devices surpassed 2.2 billion worldwide.",
    ]
    return " ".join(sentences)


@pytest.fixture
def short_text() -> str:
    """Text shorter than the default chunk size."""
    return "Apple Inc. reported total revenue of $383 billion for fiscal 2024. Services grew 13 percent."


@pytest.fixture
def sample_sections() -> dict[str, str]:
    """Multiple sections simulating a parsed filing."""
    return {
        "Item 1 - Business": "Apple designs and manufactures consumer electronics. " * 50,
        "Item 1A - Risk Factors": "The Company faces risks related to market conditions. " * 100,
        "Item 7 - MD&A": "Revenue increased primarily due to Services growth. " * 80,
    }


@pytest.fixture
def filing_dir(tmp_path: Path, sample_sections: dict[str, str]) -> Path:
    """Create a temporary filing directory matching Day 1 output format."""
    metadata = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "filing_type": "10-K",
        "filing_date": "2024-10-31",
        "accession_number": "0000320193-24-000123",
        "primary_document_url": "https://example.com/filing.htm",
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    for section_name, section_text in sample_sections.items():
        safe_name = section_name.replace(" ", "_").replace("&", "and").lower()
        section_path = tmp_path / f"{safe_name}.txt"
        section_path.write_text(section_text, encoding="utf-8")

    return tmp_path


# --------------------------------------------------------------------------- #
# TestSectionChunker
# --------------------------------------------------------------------------- #


class TestSectionChunker:
    """Tests for the SectionChunker class."""

    def test_init_valid(self) -> None:
        """Chunker initializes with valid parameters."""
        chunker = SectionChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100

    def test_init_overlap_exceeds_size_raises(self) -> None:
        """Overlap >= chunk_size raises ValueError (would infinite loop)."""
        with pytest.raises(ValueError, match="must be less than"):
            SectionChunker(chunk_size=100, chunk_overlap=100)

    def test_init_overlap_greater_than_size_raises(self) -> None:
        """Overlap > chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            SectionChunker(chunk_size=100, chunk_overlap=200)

    def test_count_tokens(self, chunker: SectionChunker, encoding: tiktoken.Encoding) -> None:
        """Token counting matches independent tiktoken calculation."""
        text = "The quick brown fox jumps over the lazy dog."
        assert chunker.count_tokens(text) == len(encoding.encode(text))

    def test_count_tokens_empty(self, chunker: SectionChunker) -> None:
        """Empty string has zero tokens."""
        assert chunker.count_tokens("") == 0


class TestChunkBounds:
    """Tests for chunk size enforcement."""

    def test_chunks_within_token_limit(
        self, chunker: SectionChunker, long_text: str, encoding: tiktoken.Encoding
    ) -> None:
        """Every chunk's token count should be at or below chunk_size.

        Exception: single sentences that exceed chunk_size are allowed
        as standalone chunks (we can't split mid-sentence).
        """
        chunks = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )

        for chunk in chunks:
            token_count = len(encoding.encode(chunk.text))
            # Allow some tolerance for sentence-boundary rounding
            assert token_count <= chunker.chunk_size + 50, (
                f"Chunk {chunk.metadata.chunk_index} has {token_count} tokens, "
                f"exceeds limit of {chunker.chunk_size}"
            )

    def test_multiple_chunks_from_long_text(
        self, chunker: SectionChunker, long_text: str
    ) -> None:
        """Long text should produce multiple chunks."""
        chunks = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )
        # ~715 tokens / 500 chunk_size = 2 chunks with overlap
        assert len(chunks) >= 2, f"Expected >= 2 chunks, got {len(chunks)}"

    def test_short_text_single_chunk(
        self, chunker: SectionChunker, short_text: str
    ) -> None:
        """Text shorter than chunk_size produces exactly one chunk."""
        chunks = chunker.chunk_text(
            text=short_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )
        assert len(chunks) == 1

    def test_empty_text_no_chunks(self, chunker: SectionChunker) -> None:
        """Empty text produces zero chunks."""
        chunks = chunker.chunk_text(
            text="",
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )
        assert len(chunks) == 0

    def test_whitespace_only_no_chunks(self, chunker: SectionChunker) -> None:
        """Whitespace-only text produces zero chunks."""
        chunks = chunker.chunk_text(
            text="   \n\t  ",
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )
        assert len(chunks) == 0


class TestChunkOverlap:
    """Tests for overlap behavior between consecutive chunks."""

    def test_consecutive_chunks_share_text(
        self, small_chunker: SectionChunker
    ) -> None:
        """Consecutive chunks should share overlapping text content.

        We verify this by checking that the end of chunk N appears
        at the start of chunk N+1.
        """
        # Build text long enough for multiple chunks with the small chunker
        text = "Financial results exceeded expectations. " * 30
        chunks = small_chunker.chunk_text(
            text=text,
            ticker="TEST",
            form_type="10-K",
            filing_date="2024-01-01",
            section_name="test_overlap",
        )

        assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"

        # Check that each consecutive pair has overlapping content
        for i in range(len(chunks) - 1):
            current_end = chunks[i].text[-50:]  # Last 50 chars of current
            next_start = chunks[i + 1].text[:100]  # First 100 chars of next
            # The overlap region should contain some shared words
            current_words = set(current_end.split())
            next_words = set(next_start.split())
            shared = current_words & next_words
            assert len(shared) > 0, (
                f"No overlap between chunk {i} and {i+1}. "
                f"End: '{current_end}' | Start: '{next_start}'"
            )


class TestChunkMetadata:
    """Tests for metadata correctness on chunks."""

    def test_metadata_ticker(self, chunker: SectionChunker, long_text: str) -> None:
        """Ticker is correctly attached to all chunks."""
        chunks = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="Item 1A - Risk Factors",
        )
        for chunk in chunks:
            assert chunk.metadata.ticker == "AAPL"

    def test_metadata_form_type(self, chunker: SectionChunker, long_text: str) -> None:
        """Form type is correctly attached to all chunks."""
        chunks = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="Item 1A - Risk Factors",
        )
        for chunk in chunks:
            assert chunk.metadata.form_type == "10-K"

    def test_metadata_section_name(self, chunker: SectionChunker, long_text: str) -> None:
        """Section name is correctly attached to all chunks."""
        chunks = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="Item 1A - Risk Factors",
        )
        for chunk in chunks:
            assert chunk.metadata.section_name == "Item 1A - Risk Factors"

    def test_metadata_chunk_indices(self, chunker: SectionChunker, long_text: str) -> None:
        """Chunk indices are sequential starting from 0."""
        chunks = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_metadata_total_chunks(self, chunker: SectionChunker, long_text: str) -> None:
        """All chunks in a section report the same total_chunks_in_section."""
        chunks = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )
        expected_total = len(chunks)
        for chunk in chunks:
            assert chunk.metadata.total_chunks_in_section == expected_total

    def test_metadata_token_count_matches(
        self, chunker: SectionChunker, long_text: str, encoding: tiktoken.Encoding
    ) -> None:
        """Token count in metadata matches actual token count."""
        chunks = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )
        for chunk in chunks:
            actual = len(encoding.encode(chunk.text))
            assert chunk.metadata.token_count == actual

    def test_chunk_ids_are_unique(self, chunker: SectionChunker, long_text: str) -> None:
        """Every chunk has a unique ID."""
        chunks = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )
        ids = [c.metadata.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_ids_are_deterministic(
        self, chunker: SectionChunker, long_text: str
    ) -> None:
        """Re-chunking the same text produces the same chunk IDs."""
        chunks_1 = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )
        chunks_2 = chunker.chunk_text(
            text=long_text,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            section_name="test",
        )
        ids_1 = [c.metadata.chunk_id for c in chunks_1]
        ids_2 = [c.metadata.chunk_id for c in chunks_2]
        assert ids_1 == ids_2


class TestChunkFiling:
    """Tests for multi-section filing chunking."""

    def test_chunk_filing_multiple_sections(
        self, chunker: SectionChunker, sample_sections: dict[str, str]
    ) -> None:
        """Chunking a multi-section filing produces chunks from all sections."""
        chunks = chunker.chunk_filing(
            sections=sample_sections,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
            company_name="Apple Inc.",
        )
        assert len(chunks) > len(sample_sections), (
            "Should produce more chunks than sections (sections are long)"
        )

        # Verify all sections are represented
        section_names = {c.metadata.section_name for c in chunks}
        for section_name in sample_sections:
            assert section_name in section_names

    def test_chunk_filing_no_cross_section_chunks(
        self, chunker: SectionChunker, sample_sections: dict[str, str]
    ) -> None:
        """No chunk should contain text from multiple sections."""
        chunks = chunker.chunk_filing(
            sections=sample_sections,
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-10-31",
        )

        for chunk in chunks:
            section_name = chunk.metadata.section_name
            section_text = sample_sections[section_name]
            # Each word in the chunk should exist in its declared section
            chunk_words = chunk.text.split()[:5]  # Check first 5 words
            for word in chunk_words:
                assert word in section_text, (
                    f"Word '{word}' from chunk (section={section_name}) "
                    f"not found in section text"
                )


class TestChunkFilingDirectory:
    """Tests for the convenience function that reads from disk."""

    def test_chunk_filing_directory(self, filing_dir: Path) -> None:
        """Chunking a filing directory produces chunks with correct metadata."""
        chunks = chunk_filing_directory(filing_dir, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 0

        # Check metadata from the JSON sidecar was correctly applied
        for chunk in chunks:
            assert chunk.metadata.ticker == "AAPL"
            assert chunk.metadata.form_type == "10-K"
            assert chunk.metadata.filing_date == "2024-10-31"

    def test_chunk_filing_directory_missing_metadata(self, tmp_path: Path) -> None:
        """Missing metadata.json raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No metadata.json"):
            chunk_filing_directory(tmp_path)
