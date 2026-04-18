"""Section-aware chunker for SEC filings.

Splits section-isolated text into token-bounded chunks with overlap.
Each chunk carries metadata (ticker, filing type, section, date) so
downstream retrieval can filter and cite precisely.

Design decisions:
- tiktoken for token counting (aligns with LLM context windows)
- Sliding window with overlap to preserve context continuity
- Sentence-boundary-aware splitting to avoid mid-sentence breaks
- Chunks never cross section boundaries (enforced by caller)

Debt: DAY-2-001 — Sentence boundary detection uses simple heuristics.
      A spaCy sentencizer would be more robust but adds a heavy dependency.
"""

import hashlib
import json
import re
from pathlib import Path

import structlog
import tiktoken
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default encoding used by GPT-4 and text-embedding-ada-002.
# Using the same tokenizer as the embedding model ensures chunk sizes
# are accurate relative to the model's actual token consumption.
DEFAULT_ENCODING = "cl100k_base"

# Chunk size bounds. 500 tokens is a sweet spot:
# - Large enough for semantic coherence
# - Small enough for precise retrieval
DEFAULT_CHUNK_SIZE = 500

# Overlap ensures context continuity across chunk boundaries.
# 100 tokens ≈ 20% of chunk size — standard RAG best practice.
DEFAULT_CHUNK_OVERLAP = 100

# Minimum chunk size to avoid creating tiny, low-value chunks
# from short sections (e.g., "Item 4 - Mine Safety Disclosures: None.")
MIN_CHUNK_SIZE = 50

# Sentence-ending punctuation pattern for boundary-aware splitting
SENTENCE_END_PATTERN = re.compile(r"[.!?]\s+")


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk for retrieval and citation.

    This metadata enables:
    - Filtered retrieval (e.g., only AAPL 10-K filings)
    - Precise citations (section name, filing date)
    - Deduplication (chunk_id is a content hash)

    Attributes:
        chunk_id: Deterministic hash of chunk content + metadata.
        ticker: Stock ticker symbol (e.g., "AAPL").
        company_name: Full company name from EDGAR.
        form_type: Filing form type (e.g., "10-K").
        filing_date: Date the filing was submitted (YYYY-MM-DD).
        section_name: Section the chunk belongs to (e.g., "Item 1A - Risk Factors").
        chunk_index: Position of this chunk within its section (0-indexed).
        total_chunks_in_section: Total number of chunks in this section.
        token_count: Actual token count of the chunk text.
    """

    chunk_id: str = Field(description="Deterministic hash of chunk content + metadata")
    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(default="", description="Full company name")
    form_type: str = Field(description="Filing form type (10-K, 10-Q, etc.)")
    filing_date: str = Field(description="Filing date in YYYY-MM-DD format")
    section_name: str = Field(description="Section this chunk belongs to")
    chunk_index: int = Field(description="Position within section (0-indexed)")
    total_chunks_in_section: int = Field(description="Total chunks in this section")
    token_count: int = Field(description="Token count of the chunk text")


class Chunk(BaseModel):
    """A single chunk of text with attached metadata.

    Attributes:
        text: The chunk text content.
        metadata: Metadata for retrieval filtering and citation.
    """

    text: str = Field(description="Chunk text content")
    metadata: ChunkMetadata = Field(description="Chunk metadata for retrieval and citation")


# --------------------------------------------------------------------------- #
# Chunker
# --------------------------------------------------------------------------- #


class SectionChunker:
    """Token-bounded, section-aware chunker with overlap.

    Splits text into chunks of approximately `chunk_size` tokens with
    `chunk_overlap` tokens of overlap between consecutive chunks.
    Attempts to break at sentence boundaries when possible.

    Args:
        chunk_size: Target token count per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.
        encoding_name: tiktoken encoding to use for tokenization.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        encoding_name: str = DEFAULT_ENCODING,
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Target token count per chunk.
            chunk_overlap: Overlap between consecutive chunks in tokens.
            encoding_name: tiktoken encoding name (e.g., "cl100k_base").

        Raises:
            ValueError: If chunk_overlap >= chunk_size (would cause infinite loop).
        """
        if chunk_overlap >= chunk_size:
            msg = (
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size}). Otherwise chunks never advance."
            )
            raise ValueError(msg)

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._encoding = tiktoken.get_encoding(encoding_name)

        logger.debug(
            "chunker_initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding=encoding_name,
        )

    @property
    def chunk_size(self) -> int:
        """Target token count per chunk."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Overlap between consecutive chunks in tokens."""
        return self._chunk_overlap

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        return len(self._encoding.encode(text))

    def _generate_chunk_id(self, text: str, metadata_seed: str) -> str:
        """Generate a deterministic chunk ID from content and metadata.

        Uses SHA-256 hash truncated to 16 hex chars. Deterministic means
        re-chunking the same filing produces the same IDs, enabling
        idempotent ingestion.

        Args:
            text: Chunk text content.
            metadata_seed: String combining metadata fields for uniqueness.

        Returns:
            16-character hex string as chunk ID.
        """
        content = f"{metadata_seed}|{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def chunk_text(
        self,
        text: str,
        ticker: str,
        form_type: str,
        filing_date: str,
        section_name: str,
        company_name: str = "",
    ) -> list[Chunk]:
        """Split a single section's text into token-bounded chunks.

        Tokenizes the entire text once, then slices the token array
        with overlap. Tries to find sentence boundaries (periods, newlines)
        near each slice point for cleaner breaks.

        Args:
            text: Section text to chunk.
            ticker: Stock ticker symbol.
            form_type: Filing form type.
            filing_date: Filing date string.
            section_name: Section name for metadata.
            company_name: Company name for metadata.

        Returns:
            List of Chunk objects with metadata.
        """
        text = text.strip()
        if not text:
            return []

        # Tokenize entire text ONCE — this is the only encode() call
        all_tokens = self._encoding.encode(text)
        total_tokens = len(all_tokens)

        # If the entire section fits in one chunk, return it directly
        if total_tokens <= self._chunk_size:
            if total_tokens < MIN_CHUNK_SIZE:
                logger.debug(
                    "section_too_short",
                    section=section_name,
                    tokens=total_tokens,
                    min_required=MIN_CHUNK_SIZE,
                )

            metadata_seed = f"{ticker}|{form_type}|{filing_date}|{section_name}|0"
            chunk_id = self._generate_chunk_id(text, metadata_seed)
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                ticker=ticker,
                company_name=company_name,
                form_type=form_type,
                filing_date=filing_date,
                section_name=section_name,
                chunk_index=0,
                total_chunks_in_section=1,
                token_count=total_tokens,
            )
            return [Chunk(text=text, metadata=metadata)]

        # Slide through token array, creating chunks with overlap
        step = self._chunk_size - self._chunk_overlap
        raw_chunks: list[str] = []
        pos = 0

        while pos < total_tokens:
            end = min(pos + self._chunk_size, total_tokens)
            chunk_text = self._encoding.decode(all_tokens[pos:end])

            # Try to find a clean break near the end (sentence boundary)
            # Only look if we're not at the end of the text
            if end < total_tokens and len(chunk_text) > 100:
                # Search last 20% of chunk for a sentence boundary
                search_region = chunk_text[int(len(chunk_text) * 0.8):]
                # Look for period+space, newline, or other natural breaks
                boundary = -1
                for pattern in ['. ', '.\n', '\n\n', '\n', '; ']:
                    idx = search_region.rfind(pattern)
                    if idx != -1:
                        boundary = idx + len(pattern)
                        break

                if boundary > 0:
                    # Trim chunk at the boundary
                    trim_point = int(len(chunk_text) * 0.8) + boundary
                    chunk_text = chunk_text[:trim_point]
                    # Recalculate actual token count for proper step
                    actual_tokens = len(self._encoding.encode(chunk_text))
                    # Next chunk starts after this one minus overlap
                    pos += max(actual_tokens - self._chunk_overlap, step // 2)
                else:
                    pos += step
            else:
                pos += step

            raw_chunks.append(chunk_text.strip())

        # Build final chunks with proper metadata
        total_chunks = len(raw_chunks)
        result: list[Chunk] = []
        for idx, chunk_str in enumerate(raw_chunks):
            if not chunk_str:
                continue
            metadata_seed = f"{ticker}|{form_type}|{filing_date}|{section_name}|{idx}"
            chunk_id = self._generate_chunk_id(chunk_str, metadata_seed)
            token_count = len(self._encoding.encode(chunk_str))
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                ticker=ticker,
                company_name=company_name,
                form_type=form_type,
                filing_date=filing_date,
                section_name=section_name,
                chunk_index=idx,
                total_chunks_in_section=total_chunks,
                token_count=token_count,
            )
            result.append(Chunk(text=chunk_str, metadata=metadata))

        logger.info(
            "section_chunked",
            section=section_name,
            total_tokens=total_tokens,
            chunks_created=len(result),
        )
        return result

    def chunk_filing(
        self,
        sections: dict[str, str],
        ticker: str,
        form_type: str,
        filing_date: str,
        company_name: str = "",
    ) -> list[Chunk]:
        """Chunk all sections of a filing.

        Iterates over each section from the parsed filing and produces
        token-bounded chunks. No chunk crosses a section boundary.

        Args:
            sections: Dict mapping section name to section text.
            ticker: Stock ticker symbol.
            form_type: Filing form type.
            filing_date: Filing date string.
            company_name: Company name for metadata.

        Returns:
            List of all Chunk objects across all sections.
        """
        all_chunks: list[Chunk] = []

        for section_name, section_text in sections.items():
            section_chunks = self.chunk_text(
                text=section_text,
                ticker=ticker,
                form_type=form_type,
                filing_date=filing_date,
                section_name=section_name,
                company_name=company_name,
            )
            all_chunks.extend(section_chunks)

        logger.info(
            "filing_chunked",
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            sections_processed=len(sections),
            total_chunks=len(all_chunks),
        )
        return all_chunks


def chunk_filing_directory(
    filing_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a saved filing directory from Day 1 ingestion.

    Reads the metadata.json sidecar and all section .txt files,
    then produces chunks with full metadata.

    Args:
        filing_dir: Path to a filing directory (e.g., data/raw/AAPL_10-K_20251031).
        chunk_size: Target token count per chunk.
        chunk_overlap: Overlap between consecutive chunks in tokens.

    Returns:
        List of Chunk objects.

    Raises:
        FileNotFoundError: If metadata.json is missing.
        ValueError: If the filing directory structure is invalid.
    """
    metadata_path = filing_dir / "metadata.json"
    if not metadata_path.exists():
        msg = f"No metadata.json found in {filing_dir}"
        raise FileNotFoundError(msg)

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    ticker = metadata["ticker"]
    form_type = metadata["filing_type"]
    filing_date = metadata["filing_date"]
    company_name = metadata.get("company_name", "")

    # Read all section files
    sections: dict[str, str] = {}
    for section_file in sorted(filing_dir.glob("*.txt")):
        # Reconstruct section name from filename
        # e.g., "item_1a_-_risk_factors.txt" -> "item_1a_-_risk_factors"
        section_key = section_file.stem
        sections[section_key] = section_file.read_text(encoding="utf-8")

    chunker = SectionChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_filing(
        sections=sections,
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        company_name=company_name,
    )

    logger.info(
        "filing_directory_chunked",
        directory=str(filing_dir),
        total_chunks=len(chunks),
    )
    return chunks
