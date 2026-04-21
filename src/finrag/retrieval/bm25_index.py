"""BM25 sparse retrieval index for SEC filings.

Provides keyword-based retrieval using the Okapi BM25 algorithm.
Complements dense vector search by catching exact financial terms
that embedding models often miss (ticker symbols, accounting terms,
specific numerical references like "diluted EPS").

Design decisions:
- rank-bm25 for lightweight, dependency-free BM25 implementation
- Custom financial-aware tokenizer (preserves ticker symbols, numbers)
- In-memory index with serialization for persistence
- Metadata-filtered search to narrow by ticker/section/form_type
- Returns scored results compatible with Day 5 hybrid fusion

Debt: DAY-4-001 — Tokenizer uses simple regex. A finance-specific
      tokenizer (handling $, %, B/M/K suffixes) would improve recall.
"""

import json
import pickle
import re
from pathlib import Path

import structlog
from rank_bm25 import BM25Okapi

from finrag.ingestion.chunker import Chunk

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# BM25 parameters (Okapi defaults are well-studied)
# k1: term frequency saturation. Higher = more weight on repeated terms.
# b: document length normalization. 0 = no normalization, 1 = full.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

# Pattern for tokenization: split on non-alphanumeric, keep numbers/decimals
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:\.[0-9]+)*")


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing.

    Financial-aware tokenizer that:
    - Lowercases for case-insensitive matching
    - Preserves decimal numbers (e.g., "46.2" stays as one token)
    - Preserves ticker-like tokens (e.g., "AAPL")
    - Strips punctuation but keeps alphanumeric content

    Args:
        text: Raw text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


# --------------------------------------------------------------------------- #
# BM25Index
# --------------------------------------------------------------------------- #


class BM25Index:
    """BM25 sparse retrieval index with metadata filtering.

    Indexes chunked documents for keyword-based retrieval.
    Supports filtered search by metadata fields and serialization
    to disk for persistence.

    Args:
        k1: BM25 term frequency saturation parameter.
        b: BM25 document length normalization parameter.
    """

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        """Initialize an empty BM25 index.

        Args:
            k1: Term frequency saturation (default 1.5).
            b: Length normalization (default 0.75).
        """
        self._k1 = k1
        self._b = b

        # Storage for indexed documents
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

        logger.debug("bm25_index_initialized", k1=k1, b=b)

    @property
    def count(self) -> int:
        """Number of documents in the index."""
        return len(self._chunks)

    @property
    def is_built(self) -> bool:
        """Whether the index has been built."""
        return self._bm25 is not None

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the index and rebuild.

        Tokenizes each chunk's text and rebuilds the BM25 index.
        Can be called multiple times to add more documents.

        Args:
            chunks: List of Chunk objects to index.

        Returns:
            Total number of documents in the index after adding.
        """
        if not chunks:
            return self.count

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self._chunks.append(chunk)
            self._tokenized_corpus.append(tokens)

        # Rebuild BM25 index with all documents
        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self._k1,
            b=self._b,
        )

        logger.info(
            "bm25_index_built",
            total_documents=len(self._chunks),
            new_documents=len(chunks),
            avg_tokens=sum(len(t) for t in self._tokenized_corpus) // max(len(self._tokenized_corpus), 1),
        )
        return self.count

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Search the BM25 index with optional metadata filtering.

        Scores all documents against the query, optionally filters
        by metadata, and returns the top-n results sorted by score.

        Args:
            query_text: Natural language query string.
            n_results: Maximum number of results to return.
            where: Optional metadata filter dict.
                Supports flat key-value: {"ticker": "AAPL"}
                and compound: {"$and": [{"ticker": "AAPL"}, {"form_type": "10-K"}]}

        Returns:
            List of result dicts sorted by BM25 score (descending),
            each containing:
            - text: The chunk text
            - metadata: Dict of chunk metadata
            - score: BM25 relevance score
            - chunk_id: The deterministic chunk ID
        """
        if not self.is_built:
            logger.warning("bm25_query_on_empty_index")
            return []

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        # Score all documents
        scores = self._bm25.get_scores(query_tokens)

        # Build (score, index) pairs and apply metadata filter
        scored_indices: list[tuple[float, int]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue

            chunk = self._chunks[idx]

            # Apply metadata filter
            if where and not self._matches_filter(chunk, where):
                continue

            scored_indices.append((score, idx))

        # Sort by score descending, take top n
        scored_indices.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_indices[:n_results]

        # Build output
        output: list[dict] = []
        for score, idx in top_results:
            chunk = self._chunks[idx]
            output.append({
                "chunk_id": chunk.metadata.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "ticker": chunk.metadata.ticker,
                    "company_name": chunk.metadata.company_name,
                    "form_type": chunk.metadata.form_type,
                    "filing_date": chunk.metadata.filing_date,
                    "section_name": chunk.metadata.section_name,
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks_in_section": chunk.metadata.total_chunks_in_section,
                    "token_count": chunk.metadata.token_count,
                },
                "score": float(score),
            })

        logger.info(
            "bm25_query_executed",
            query_preview=query_text[:80],
            query_tokens=len(query_tokens),
            candidates=len(scored_indices),
            returned=len(output),
            filter=where,
        )
        return output

    def _matches_filter(self, chunk: Chunk, where: dict) -> bool:
        """Check if a chunk matches a metadata filter.

        Supports:
        - Simple: {"ticker": "AAPL"}
        - Compound: {"$and": [{"ticker": "AAPL"}, {"form_type": "10-K"}]}

        Args:
            chunk: Chunk to check.
            where: Filter dict.

        Returns:
            True if the chunk matches the filter.
        """
        if "$and" in where:
            return all(
                self._matches_filter(chunk, sub_filter)
                for sub_filter in where["$and"]
            )

        meta = chunk.metadata
        for key, value in where.items():
            actual = getattr(meta, key, None)
            if actual != value:
                return False
        return True

    def save(self, path: Path) -> None:
        """Serialize the index to disk.

        Saves the chunks and tokenized corpus. The BM25 index
        is rebuilt on load (it's fast).

        Args:
            path: File path to save to (.pkl).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [c.model_dump() for c in self._chunks],
            "tokenized_corpus": self._tokenized_corpus,
            "k1": self._k1,
            "b": self._b,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(
            "bm25_index_saved",
            path=str(path),
            documents=len(self._chunks),
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load a serialized index from disk.

        Args:
            path: File path to load from (.pkl).

        Returns:
            Reconstructed BM25Index.

        Raises:
            FileNotFoundError: If the index file doesn't exist.
        """
        if not path.exists():
            msg = f"BM25 index file not found: {path}"
            raise FileNotFoundError(msg)

        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        index = cls(k1=data["k1"], b=data["b"])
        index._chunks = [Chunk(**c) for c in data["chunks"]]
        index._tokenized_corpus = data["tokenized_corpus"]

        if index._tokenized_corpus:
            index._bm25 = BM25Okapi(
                index._tokenized_corpus,
                k1=index._k1,
                b=index._b,
            )

        logger.info(
            "bm25_index_loaded",
            path=str(path),
            documents=len(index._chunks),
        )
        return index

    def get_stats(self) -> dict:
        """Return index statistics.

        Returns:
            Dict with index metadata and counts.
        """
        stats: dict = {
            "total_documents": self.count,
            "is_built": self.is_built,
            "k1": self._k1,
            "b": self._b,
        }

        if self._chunks:
            tickers = {c.metadata.ticker for c in self._chunks}
            form_types = {c.metadata.form_type for c in self._chunks}
            stats["unique_tickers"] = sorted(tickers)
            stats["unique_form_types"] = sorted(form_types)
            avg_tokens = sum(len(t) for t in self._tokenized_corpus) // len(self._tokenized_corpus)
            stats["avg_tokens_per_doc"] = avg_tokens

        return stats
