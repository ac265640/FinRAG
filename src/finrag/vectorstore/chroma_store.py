"""ChromaDB vector store with sentence-transformer embeddings.

Provides persistent storage and retrieval for chunked SEC filings.
Embeddings are generated locally via sentence-transformers, stored
in ChromaDB with full metadata for filtered semantic search.

Design decisions:
- ChromaDB persistent client for local-first, zero-cost operation
- sentence-transformers for offline, reproducible embeddings
- Metadata stored alongside vectors for filtered retrieval
- Batch upsert with deterministic IDs for idempotent ingestion
- Collection-per-use-case pattern (one collection for filings)

Debt: DAY-3-001 — Using all-MiniLM-L6-v2 (384-dim). Upgrade to
      a finance-tuned model (e.g., FinBERT) for better domain recall.
"""

from pathlib import Path

import chromadb
import structlog
from sentence_transformers import SentenceTransformer

from finrag.ingestion.chunker import Chunk

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default embedding model. all-MiniLM-L6-v2 is a good balance of
# speed (80 MB, ~14k sentences/sec on CPU) and quality (384-dim).
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB collection name for SEC filings
DEFAULT_COLLECTION_NAME = "sec_filings"

# Maximum batch size for ChromaDB upserts. ChromaDB has a limit
# of ~41666 per batch, but smaller batches are safer for memory.
UPSERT_BATCH_SIZE = 500

# Embedding dimension for the default model
EMBEDDING_DIM = 384


# --------------------------------------------------------------------------- #
# ChromaStore
# --------------------------------------------------------------------------- #


class ChromaStore:
    """ChromaDB-backed vector store with local embeddings.

    Handles embedding generation, storage, and retrieval for
    chunked SEC filings. Supports filtered search by metadata
    (ticker, form_type, section, date).

    Args:
        persist_dir: Directory for ChromaDB persistent storage.
        collection_name: Name of the ChromaDB collection.
        embedding_model: sentence-transformers model name or path.
    """

    def __init__(
        self,
        persist_dir: str | Path = "./data/chroma",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """Initialize the ChromaDB store and embedding model.

        Args:
            persist_dir: Path for persistent ChromaDB storage.
            collection_name: Name for the ChromaDB collection.
            embedding_model: HuggingFace model ID for embeddings.
        """
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._collection_name = collection_name
        self._model_name = embedding_model

        # Initialize ChromaDB persistent client
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
        )

        # Get or create collection (idempotent)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

        # Lazy-load embedding model (heavy — only load when needed)
        self._model: SentenceTransformer | None = None

        logger.info(
            "chroma_store_initialized",
            persist_dir=str(self._persist_dir),
            collection=collection_name,
            embedding_model=embedding_model,
            existing_count=self._collection.count(),
        )

    @property
    def collection_name(self) -> str:
        """Name of the active ChromaDB collection."""
        return self._collection_name

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return self._collection.count()

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model.

        Loading sentence-transformers models takes 1-2 seconds and
        ~80MB memory. We defer this until the first embed call.

        Returns:
            The loaded SentenceTransformer model.
        """
        if self._model is None:
            logger.info("loading_embedding_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "embedding_model_loaded",
                model=self._model_name,
                embedding_dim=self._model.get_embedding_dimension(),
            )
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        model = self._get_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Unit vectors for cosine sim
        )
        return embeddings.tolist()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and upsert chunks into ChromaDB.

        Uses deterministic chunk_ids from Day 2 for idempotent
        ingestion. Re-adding the same chunks is a safe no-op.

        Args:
            chunks: List of Chunk objects to embed and store.

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        total_added = 0

        # Process in batches to manage memory
        for batch_start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + UPSERT_BATCH_SIZE]

            ids = [c.metadata.chunk_id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [
                {
                    "ticker": c.metadata.ticker,
                    "company_name": c.metadata.company_name,
                    "form_type": c.metadata.form_type,
                    "filing_date": c.metadata.filing_date,
                    "section_name": c.metadata.section_name,
                    "chunk_index": c.metadata.chunk_index,
                    "total_chunks_in_section": c.metadata.total_chunks_in_section,
                    "token_count": c.metadata.token_count,
                }
                for c in batch
            ]

            # Generate embeddings for this batch
            embeddings = self.embed_texts(texts)

            # Upsert (insert or update) into ChromaDB
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            total_added += len(batch)
            logger.info(
                "batch_upserted",
                batch_size=len(batch),
                total_so_far=total_added,
                total_target=len(chunks),
            )

        logger.info(
            "chunks_added",
            total_upserted=total_added,
            collection_count=self._collection.count(),
        )
        return total_added

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Semantic search with optional metadata filtering.

        Embeds the query text, searches ChromaDB for nearest neighbors,
        and optionally filters by metadata (ticker, section, date, etc.).

        Args:
            query_text: Natural language query string.
            n_results: Number of results to return.
            where: Optional ChromaDB where filter for metadata.
                Example: {"ticker": "AAPL"} or
                {"$and": [{"ticker": "AAPL"}, {"form_type": "10-K"}]}

        Returns:
            List of result dicts, each containing:
            - text: The chunk text
            - metadata: Dict of chunk metadata
            - distance: Cosine distance (lower = more similar)
            - chunk_id: The deterministic chunk ID
        """
        query_embedding = self.embed_texts([query_text])[0]

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        # Unpack ChromaDB's nested list format into flat dicts
        output: list[dict] = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                output.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })

        logger.info(
            "query_executed",
            query_preview=query_text[:80],
            n_results=len(output),
            filter=where,
        )
        return output

    def delete_by_ticker(self, ticker: str) -> None:
        """Delete all chunks for a specific ticker.

        Useful for re-ingesting updated filings.

        Args:
            ticker: Stock ticker to delete all chunks for.
        """
        self._collection.delete(where={"ticker": ticker})
        logger.info(
            "chunks_deleted",
            ticker=ticker,
            remaining_count=self._collection.count(),
        )

    def reset(self) -> None:
        """Delete and recreate the collection.

        WARNING: This is destructive. All embeddings are lost.
        """
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(
            "collection_reset",
            collection=self._collection_name,
        )

    def get_stats(self) -> dict:
        """Return collection statistics.

        Returns:
            Dict with collection metadata and counts.
        """
        count = self._collection.count()

        stats = {
            "collection_name": self._collection_name,
            "total_chunks": count,
            "persist_dir": str(self._persist_dir),
            "embedding_model": self._model_name,
        }

        # Sample to get unique tickers if collection is non-empty
        if count > 0:
            sample = self._collection.get(
                limit=min(count, 1000),
                include=["metadatas"],
            )
            if sample["metadatas"]:
                tickers = {m.get("ticker", "") for m in sample["metadatas"]}
                form_types = {m.get("form_type", "") for m in sample["metadatas"]}
                stats["unique_tickers"] = sorted(tickers)
                stats["unique_form_types"] = sorted(form_types)
                stats["sampled_count"] = len(sample["metadatas"])

        return stats
