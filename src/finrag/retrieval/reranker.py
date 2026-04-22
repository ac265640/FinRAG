"""Cross-encoder reranker for precision-focused retrieval.

Takes the top-K candidates from hybrid retrieval (Day 5) and
reranks them using a cross-encoder that sees query + document
jointly. This fixes ordering errors that bi-encoders and RRF
cannot catch, because cross-encoders compute true semantic
relevance between the query-document pair.

Pipeline position:
    Corpus → Retriever (recall) → Reranker (precision) → LLM

Design decisions:
- sentence-transformers CrossEncoder for local, offline reranking
- ms-marco-MiniLM-L-6-v2: fast (22M params), trained on MS MARCO
  passage ranking. Good balance of speed and quality for MVP.
- Configurable top_k output (default 5) to control LLM context size.
- Score normalization to [0, 1] via sigmoid for interpretable
  relevance thresholds.
- Batch inference for efficiency on larger candidate sets.

Debt: DAY-6-001 — Using ms-marco-MiniLM-L-6-v2 (general domain).
      A finance-tuned cross-encoder would improve precision on
      domain-specific queries. Resolve if retrieval eval shows gaps.
"""

import math

import structlog
from sentence_transformers import CrossEncoder

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default cross-encoder model. ms-marco-MiniLM-L-6-v2 is trained on
# MS MARCO passage ranking, fast (~22M params), and works well for
# general-purpose reranking. Returns raw logits, not probabilities.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Default number of top results to return after reranking.
DEFAULT_TOP_K = 5

# Minimum relevance score (post-sigmoid) to include a result.
# Below this threshold, chunks are likely irrelevant noise.
DEFAULT_MIN_RELEVANCE = 0.01


# --------------------------------------------------------------------------- #
# CrossEncoderReranker
# --------------------------------------------------------------------------- #


class CrossEncoderReranker:
    """Cross-encoder reranker for precision-focused retrieval.

    Takes candidate results from the hybrid retriever, scores each
    (query, document) pair with a cross-encoder, and returns the
    top-k most relevant chunks.

    The cross-encoder sees the query and document together in a
    single transformer pass, enabling deep token-level interaction
    that bi-encoders miss. This is why cross-encoders achieve
    higher precision but can't scale to full-corpus search.

    Args:
        model_name: HuggingFace cross-encoder model name.
        top_k: Number of results to return after reranking.
        min_relevance: Minimum sigmoid score to include (0-1).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        top_k: int = DEFAULT_TOP_K,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
    ) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model ID.
            top_k: Max results after reranking (default 5).
            min_relevance: Min sigmoid score threshold (default 0.01).
        """
        self._model_name = model_name
        self._top_k = top_k
        self._min_relevance = min_relevance

        # Lazy-load the model (heavy — ~90MB)
        self._model: CrossEncoder | None = None

        logger.info(
            "reranker_initialized",
            model=model_name,
            top_k=top_k,
            min_relevance=min_relevance,
        )

    @property
    def top_k(self) -> int:
        """Number of results returned after reranking."""
        return self._top_k

    def _get_model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model.

        Loading takes ~1-2 seconds and ~90MB memory.
        Deferred until first rerank call.

        Returns:
            The loaded CrossEncoder model.
        """
        if self._model is None:
            logger.info("loading_reranker_model", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("reranker_model_loaded", model=self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank candidates using cross-encoder scoring.

        Scores each (query, candidate_text) pair jointly, applies
        sigmoid normalization, filters by min_relevance, and returns
        top-k results sorted by reranker score.

        Args:
            query: The original query string.
            candidates: List of candidate dicts from hybrid retrieval.
                Each must contain a 'text' key with the chunk text.
            top_k: Override the default top_k for this call.

        Returns:
            Reranked list of result dicts, sorted by relevance
            score descending. Each result includes:
            - reranker_score: Sigmoid-normalized relevance (0-1)
            - reranker_rank: 1-based rank after reranking
            - All original fields from the candidate dict
        """
        if not candidates:
            return []

        if not query.strip():
            return []

        effective_top_k = top_k if top_k is not None else self._top_k

        model = self._get_model()

        # Build (query, document) pairs for batch inference
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs in one batch
        raw_scores = model.predict(pairs, show_progress_bar=False)

        # Normalize scores via sigmoid and attach to candidates
        scored: list[tuple[float, int]] = []
        for idx, raw_score in enumerate(raw_scores):
            sigmoid_score = _sigmoid(float(raw_score))
            if sigmoid_score >= self._min_relevance:
                scored.append((sigmoid_score, idx))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build output with top-k
        output: list[dict] = []
        for rank, (score, idx) in enumerate(scored[:effective_top_k], start=1):
            result = candidates[idx].copy()
            result["reranker_score"] = score
            result["reranker_rank"] = rank
            output.append(result)

        logger.info(
            "rerank_complete",
            query_preview=query[:80],
            candidates_in=len(candidates),
            above_threshold=len(scored),
            returned=len(output),
            top_score=output[0]["reranker_score"] if output else 0.0,
        )

        return output


def _sigmoid(x: float) -> float:
    """Apply sigmoid function to normalize scores to [0, 1].

    Cross-encoder models (like ms-marco-MiniLM) output raw logits,
    not probabilities. Sigmoid converts these to interpretable
    relevance scores where:
    - 0.0 = completely irrelevant
    - 0.5 = borderline
    - 1.0 = highly relevant

    Args:
        x: Raw logit score.

    Returns:
        Sigmoid-normalized score in [0, 1].
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # Numerically stable for large negative values
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)
