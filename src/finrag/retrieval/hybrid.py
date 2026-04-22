"""Hybrid retrieval with Reciprocal Rank Fusion.

Combines BM25 sparse search and ChromaDB dense vector search
via Reciprocal Rank Fusion (RRF). Supports multi-query expansion
and HyDE for broader recall on financial queries.

Design decisions:
- RRF over linear combination: rank-based fusion is robust to
  score distribution differences between BM25 and cosine similarity.
  BM25 scores are unbounded; cosine distances are [0, 2]. RRF
  sidesteps normalization entirely by using ranks, not scores.
- Pluggable multi-query and HyDE strategies via callables, so the
  LLM-based versions drop in cleanly on Day 7-8.
- Metadata filtering applied at both retrieval layers for efficiency.
  Filtering at the store level avoids fetching irrelevant candidates.
- Deduplication within each source before fusion prevents a document
  appearing multiple times in one ranked list from inflating its score.

Debt: DAY-5-001 — Multi-query uses rule-based synonym expansion.
      LLM-based rephrasings will improve recall. Resolve on Day 7.
Debt: DAY-5-002 — HyDE is a no-op stub. Needs LLM to generate
      hypothetical answer documents. Resolve on Day 7-8.
"""

from collections import defaultdict
from collections.abc import Callable

import structlog

from finrag.retrieval.bm25_index import BM25Index
from finrag.vectorstore.chroma_store import ChromaStore

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# RRF constant k. Controls how much lower-ranked documents are penalized.
# k=60 is the standard value from the original RRF paper (Cormack et al. 2009).
# Higher k = more equal weighting across ranks.
# Lower k = top-ranked documents get disproportionately more weight.
RRF_K = 60

# Default number of candidates to fetch from each retriever before fusion.
# Fetching more than the final n_results gives RRF enough signal to work with.
# Too few candidates = fusion has poor recall. Too many = slower retrieval.
DEFAULT_CANDIDATES_PER_RETRIEVER = 20

# Financial term expansions for rule-based multi-query.
# Maps common abbreviations and short forms to domain-specific synonyms.
# Limited to 2 synonyms per term to keep query count manageable.
FINANCIAL_SYNONYMS: dict[str, list[str]] = {
    "revenue": ["net sales", "total revenue"],
    "profit": ["net income", "earnings"],
    "eps": ["earnings per share", "diluted eps"],
    "margin": ["gross margin", "operating margin"],
    "debt": ["long-term debt", "total debt"],
    "cash": ["cash and cash equivalents", "free cash flow"],
    "growth": ["year over year growth", "yoy growth"],
    "risk": ["risk factors", "risks and uncertainties"],
    "capex": ["capital expenditure", "capital expenditures"],
    "r&d": ["research and development"],
    "sg&a": ["selling general and administrative"],
    "ebitda": ["earnings before interest taxes depreciation amortization"],
    "roe": ["return on equity"],
    "roa": ["return on assets"],
    "fcf": ["free cash flow"],
    "goodwill": ["goodwill impairment", "intangible assets"],
}


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF computes a fused score for each document:
        score(d) = sum over r in ranked_lists of: 1 / (k + rank_r(d))

    where rank_r(d) is the 1-based rank of document d in list r,
    and k is a constant that controls rank sensitivity.

    Documents appearing in multiple lists get higher fused scores,
    which is exactly the behavior we want: a chunk that both BM25
    and vector search agree on is more likely to be relevant.

    Args:
        ranked_lists: List of ranked result lists. Each result dict
            must contain a 'chunk_id' key for deduplication.
        k: RRF constant (default 60). Higher = more equal weighting.

    Returns:
        Merged and re-ranked list of result dicts, sorted by fused
        score descending. Each result includes:
        - rrf_score: The fused relevance score
        - retrieval_sources: List of sources that found this chunk
    """
    # Accumulate RRF scores by chunk_id
    scores: dict[str, float] = defaultdict(float)
    # Track the best result dict for each chunk_id
    best_result: dict[str, dict] = {}
    # Track which sources contributed to each result
    sources: dict[str, list[str]] = defaultdict(list)

    source_names = ["dense", "sparse"]

    for list_idx, ranked_list in enumerate(ranked_lists):
        source_name = (
            source_names[list_idx]
            if list_idx < len(source_names)
            else f"source_{list_idx}"
        )
        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = result["chunk_id"]
            scores[chunk_id] += 1.0 / (k + rank)
            sources[chunk_id].append(source_name)

            # Keep the first occurrence's full result dict
            if chunk_id not in best_result:
                best_result[chunk_id] = result.copy()

    # Build fused results
    fused: list[dict] = []
    for chunk_id, rrf_score in scores.items():
        result = best_result[chunk_id]
        result["rrf_score"] = rrf_score
        result["retrieval_sources"] = sources[chunk_id]
        # Remove retriever-specific score fields to avoid confusion
        result.pop("score", None)
        result.pop("distance", None)
        fused.append(result)

    # Sort by RRF score descending
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused


# --------------------------------------------------------------------------- #
# Multi-Query Expansion
# --------------------------------------------------------------------------- #


def expand_financial_query(query: str) -> list[str]:
    """Generate query variations using financial term expansion.

    Rule-based approach that identifies financial abbreviations and
    terms in the query and generates variations with their synonyms.

    This is a Day 5 placeholder. LLM-based rephrasings will replace
    this on Day 7 for much better recall. The LLM version will
    generate semantically diverse rephrasings rather than just
    swapping synonyms.

    Args:
        query: Original query string.

    Returns:
        List of query variations including the original.
        Always returns at least the original query.
    """
    variations = [query]
    query_lower = query.lower()

    for term, synonyms in FINANCIAL_SYNONYMS.items():
        if term in query_lower:
            for synonym in synonyms[:2]:
                variation = query_lower.replace(term, synonym)
                if variation != query_lower and variation not in variations:
                    variations.append(variation)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in variations:
        normalized = v.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(v)

    logger.debug(
        "query_expanded",
        original=query[:80],
        variation_count=len(unique),
    )
    return unique


# --------------------------------------------------------------------------- #
# HyDE Stub
# --------------------------------------------------------------------------- #


def hyde_passthrough(query: str) -> str:
    """No-op HyDE stub that returns the query unchanged.

    HyDE (Hypothetical Document Embeddings) generates a hypothetical
    answer document using an LLM, then embeds THAT instead of the
    query. This gives the vector search a document-like embedding
    to match against, bridging the query-document distribution gap.

    Example: Query "What was Apple's revenue?" becomes a hypothetical
    response like "Apple reported total revenue of $X billion for
    fiscal year 2024..." which is then embedded for similarity search.

    This stub is a placeholder until LLM integration on Day 7-8.
    The strategy pattern allows dropping in the real HyDE with zero
    changes to the HybridRetriever.

    Args:
        query: Original query string.

    Returns:
        The query unchanged.
    """
    return query


# --------------------------------------------------------------------------- #
# HybridRetriever
# --------------------------------------------------------------------------- #


class HybridRetriever:
    """Hybrid retriever combining BM25 and vector search with RRF.

    Runs BM25 keyword search and ChromaDB semantic search in sequence,
    then fuses results using Reciprocal Rank Fusion. Supports
    multi-query expansion for broader recall and HyDE for improved
    vector search on question-style queries.

    The key insight: BM25 excels at exact financial terms like
    "diluted EPS" or "goodwill impairment" that embeddings miss.
    Vector search excels at semantic similarity like finding revenue
    discussion when the query says "top line performance". RRF
    combines both signals without needing to calibrate score scales.

    Args:
        chroma_store: ChromaDB vector store instance.
        bm25_index: BM25 sparse retrieval index.
        rrf_k: RRF fusion constant (default 60).
        candidates_per_retriever: Number of candidates to fetch
            from each retriever before fusion.
        multi_query_fn: Optional callable for query expansion.
            Signature: (query: str) -> list[str].
            Default: rule-based financial term expansion.
        hyde_fn: Optional callable for HyDE transformation.
            Signature: (query: str) -> str.
            Default: passthrough (no-op).
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_index: BM25Index,
        rrf_k: int = RRF_K,
        candidates_per_retriever: int = DEFAULT_CANDIDATES_PER_RETRIEVER,
        multi_query_fn: Callable[[str], list[str]] | None = None,
        hyde_fn: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            chroma_store: Initialized ChromaStore with embedded chunks.
            bm25_index: Built BM25Index with indexed chunks.
            rrf_k: RRF constant k (default 60).
            candidates_per_retriever: Candidates per source (default 20).
            multi_query_fn: Custom query expansion function.
            hyde_fn: Custom HyDE transformation function.
        """
        self._chroma = chroma_store
        self._bm25 = bm25_index
        self._rrf_k = rrf_k
        self._candidates = candidates_per_retriever
        self._multi_query_fn = multi_query_fn or expand_financial_query
        self._hyde_fn = hyde_fn or hyde_passthrough

        logger.info(
            "hybrid_retriever_initialized",
            rrf_k=rrf_k,
            candidates_per_retriever=candidates_per_retriever,
            has_custom_multi_query=multi_query_fn is not None,
            has_custom_hyde=hyde_fn is not None,
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
        use_multi_query: bool = False,
        use_hyde: bool = False,
    ) -> list[dict]:
        """Run hybrid retrieval with optional multi-query and HyDE.

        Pipeline:
        1. Optionally expand query into variations (multi-query)
        2. Optionally transform query for vector search (HyDE)
        3. Run BM25 search for each query variant
        4. Run vector search for each query variant
        5. Deduplicate within each source
        6. Fuse via RRF
        7. Return top n_results

        Args:
            query: Natural language query string.
            n_results: Maximum results to return after fusion.
            where: Optional metadata filter dict. Applied to both
                BM25 and vector search.
                Example: {"ticker": "AAPL"} or
                {"$and": [{"ticker": "AAPL"}, {"form_type": "10-K"}]}
            use_multi_query: If True, expand query into variations
                for broader recall.
            use_hyde: If True, generate hypothetical document embedding
                for vector search.

        Returns:
            List of result dicts sorted by RRF score descending,
            each containing:
            - chunk_id: The deterministic chunk ID
            - text: The chunk text
            - metadata: Dict of chunk metadata
            - rrf_score: Fused relevance score
            - retrieval_sources: List of sources ("dense", "sparse")
        """
        if not query.strip():
            return []

        # Step 1: Determine query variants
        if use_multi_query:
            queries = self._multi_query_fn(query)
            logger.info(
                "multi_query_expanded",
                original=query[:80],
                variations=len(queries),
            )
        else:
            queries = [query]

        # Step 2: Apply HyDE transformation for vector search
        if use_hyde:
            hyde_query = self._hyde_fn(query)
            logger.info(
                "hyde_applied",
                original_preview=query[:80],
                hyde_preview=hyde_query[:80],
            )
        else:
            hyde_query = query

        # Step 3: Run retrievals for all query variants
        all_dense_results: list[dict] = []
        all_sparse_results: list[dict] = []

        for q in queries:
            # Dense search: use HyDE query for the original query,
            # use the expanded variant directly for multi-query expansions.
            dense_q = hyde_query if q == query else q

            dense_results = self._chroma.query(
                query_text=dense_q,
                n_results=self._candidates,
                where=where,
            )
            sparse_results = self._bm25.query(
                query_text=q,
                n_results=self._candidates,
                where=where,
            )

            all_dense_results.extend(dense_results)
            all_sparse_results.extend(sparse_results)

        # Step 4: Deduplicate within each source (keep first = highest ranked)
        dense_deduped = self._deduplicate_results(all_dense_results)
        sparse_deduped = self._deduplicate_results(all_sparse_results)

        # Step 5: Fuse via RRF
        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_deduped, sparse_deduped],
            k=self._rrf_k,
        )

        # Step 6: Trim to requested count
        results = fused[:n_results]

        logger.info(
            "hybrid_retrieval_complete",
            query_preview=query[:80],
            queries_executed=len(queries),
            dense_candidates=len(dense_deduped),
            sparse_candidates=len(sparse_deduped),
            fused_total=len(fused),
            returned=len(results),
            filter=where,
            multi_query=use_multi_query,
            hyde=use_hyde,
        )

        return results

    def retrieve_dense_only(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Run vector search only, bypassing BM25 and RRF.

        Useful for benchmarking dense vs hybrid retrieval.

        Args:
            query: Natural language query string.
            n_results: Maximum results to return.
            where: Optional metadata filter dict.

        Returns:
            List of result dicts from vector search only.
        """
        return self._chroma.query(
            query_text=query,
            n_results=n_results,
            where=where,
        )

    def retrieve_sparse_only(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Run BM25 search only, bypassing vector search and RRF.

        Useful for benchmarking sparse vs hybrid retrieval.

        Args:
            query: Natural language query string.
            n_results: Maximum results to return.
            where: Optional metadata filter dict.

        Returns:
            List of result dicts from BM25 search only.
        """
        return self._bm25.query(
            query_text=query,
            n_results=n_results,
            where=where,
        )

    def _deduplicate_results(self, results: list[dict]) -> list[dict]:
        """Remove duplicate chunk_ids, keeping the first occurrence.

        When multi-query produces overlapping results across query
        variants, we keep only the highest-ranked occurrence. This
        prevents a single chunk from appearing multiple times in
        one ranked list, which would inflate its RRF score unfairly.

        Args:
            results: List of result dicts with chunk_id field.

        Returns:
            Deduplicated list preserving original order.
        """
        seen: set[str] = set()
        deduped: list[dict] = []
        for result in results:
            chunk_id = result["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(result)
        return deduped

    def get_stats(self) -> dict:
        """Return retriever configuration and component stats.

        Returns:
            Dict with hybrid retriever config and sub-component stats.
        """
        return {
            "rrf_k": self._rrf_k,
            "candidates_per_retriever": self._candidates,
            "bm25_stats": self._bm25.get_stats(),
            "chroma_stats": self._chroma.get_stats(),
        }
