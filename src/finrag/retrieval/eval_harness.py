"""Retrieval evaluation harness for measuring search quality.

Provides metrics to evaluate retrieval independently from generation,
so we can catch retrieval failures before they pollute LLM answers.

Supported metrics:
- Precision@k: fraction of retrieved docs that are relevant
- Recall@k: fraction of relevant docs that were retrieved
- MRR (Mean Reciprocal Rank): average 1/rank of first relevant doc
- Hit Rate@k: fraction of queries where at least one relevant doc is in top-k
- NDCG@k: Normalized Discounted Cumulative Gain (position-aware relevance)

Design decisions:
- Evaluation dataset is a list of (query, relevant_chunk_ids) pairs.
  This is the minimal unit for retrieval evaluation.
- Metrics are computed per-query and then averaged (macro averaging).
- JSON-serializable evaluation dataset format for versioning in git.
- Retriever-agnostic: works with any callable that returns ranked dicts.

Debt: DAY-6-002 — Golden evaluation dataset is hand-crafted and small.
      Day 13 will build a proper 50+ Q/A pair dataset with RAGAS.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Data Structures
# --------------------------------------------------------------------------- #


@dataclass
class EvalQuery:
    """A single evaluation query with known relevant chunk IDs.

    Args:
        query: The natural language query string.
        relevant_chunk_ids: Set of chunk IDs that are relevant answers.
        metadata: Optional metadata for categorizing eval results
            (e.g., query type, difficulty).
    """

    query: str
    relevant_chunk_ids: set[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryResult:
    """Evaluation result for a single query.

    Args:
        query: The query string.
        precision_at_k: Precision@k score.
        recall_at_k: Recall@k score.
        reciprocal_rank: 1/rank of first relevant result (0 if none).
        hit: Whether any relevant doc was in top-k.
        ndcg_at_k: NDCG@k score.
        retrieved_ids: List of retrieved chunk IDs in order.
        relevant_ids: Set of relevant chunk IDs.
    """

    query: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hit: bool
    ndcg_at_k: float
    retrieved_ids: list[str]
    relevant_ids: set[str]


@dataclass
class EvalReport:
    """Aggregated evaluation report across all queries.

    Args:
        k: The k value used for evaluation.
        num_queries: Total number of queries evaluated.
        mean_precision: Mean Precision@k across all queries.
        mean_recall: Mean Recall@k across all queries.
        mrr: Mean Reciprocal Rank across all queries.
        hit_rate: Fraction of queries with at least one hit in top-k.
        mean_ndcg: Mean NDCG@k across all queries.
        per_query: Individual QueryResult for each query.
    """

    k: int
    num_queries: int
    mean_precision: float
    mean_recall: float
    mrr: float
    hit_rate: float
    mean_ndcg: float
    per_query: list[QueryResult]


# --------------------------------------------------------------------------- #
# Metric Functions
# --------------------------------------------------------------------------- #


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@k.

    Fraction of retrieved documents (top-k) that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of known relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Precision@k in [0, 1].
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for cid in top_k if cid in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@k.

    Fraction of all relevant documents that appear in top-k.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of known relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Recall@k in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for cid in top_k if cid in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Reciprocal Rank.

    1 / rank of the first relevant result. Returns 0 if no
    relevant result is found.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of known relevant chunk IDs.

    Returns:
        Reciprocal rank in (0, 1] or 0 if no hit.
    """
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Uses binary relevance (1 if relevant, 0 otherwise).
    Rewards placing relevant documents at higher ranks.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of known relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        NDCG@k in [0, 1].
    """
    if not relevant_ids or k == 0:
        return 0.0

    top_k = retrieved_ids[:k]

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for rank, cid in enumerate(top_k, start=1):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all relevant docs at top ranks
    ideal_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_k + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


# --------------------------------------------------------------------------- #
# RetrievalEvaluator
# --------------------------------------------------------------------------- #


class RetrievalEvaluator:
    """Evaluator for retrieval quality.

    Runs a set of evaluation queries against a retriever function
    and computes standard IR metrics.

    Args:
        retriever_fn: A callable that takes (query, n_results) and
            returns a list of result dicts with 'chunk_id' keys.
        k: The cutoff rank for evaluation metrics.
    """

    def __init__(
        self,
        retriever_fn: Callable[[str, int], list[dict]],
        k: int = 5,
    ) -> None:
        """Initialize the evaluator.

        Args:
            retriever_fn: Retriever callable: (query, n_results) -> results.
            k: Cutoff rank for @k metrics (default 5).
        """
        self._retriever_fn = retriever_fn
        self._k = k

    def evaluate(self, eval_queries: list[EvalQuery]) -> EvalReport:
        """Run evaluation across all queries and compute metrics.

        Args:
            eval_queries: List of EvalQuery with known relevant docs.

        Returns:
            EvalReport with aggregated and per-query metrics.
        """
        if not eval_queries:
            return EvalReport(
                k=self._k,
                num_queries=0,
                mean_precision=0.0,
                mean_recall=0.0,
                mrr=0.0,
                hit_rate=0.0,
                mean_ndcg=0.0,
                per_query=[],
            )

        per_query_results: list[QueryResult] = []

        for eq in eval_queries:
            # Run retrieval
            results = self._retriever_fn(eq.query, self._k)
            retrieved_ids = [r["chunk_id"] for r in results]

            # Compute metrics
            p_at_k = precision_at_k(retrieved_ids, eq.relevant_chunk_ids, self._k)
            r_at_k = recall_at_k(retrieved_ids, eq.relevant_chunk_ids, self._k)
            rr = reciprocal_rank(retrieved_ids, eq.relevant_chunk_ids)
            hit = rr > 0
            ndcg = ndcg_at_k(retrieved_ids, eq.relevant_chunk_ids, self._k)

            qr = QueryResult(
                query=eq.query,
                precision_at_k=p_at_k,
                recall_at_k=r_at_k,
                reciprocal_rank=rr,
                hit=hit,
                ndcg_at_k=ndcg,
                retrieved_ids=retrieved_ids,
                relevant_ids=eq.relevant_chunk_ids,
            )
            per_query_results.append(qr)

            logger.debug(
                "eval_query_complete",
                query=eq.query[:60],
                precision=f"{p_at_k:.3f}",
                recall=f"{r_at_k:.3f}",
                rr=f"{rr:.3f}",
                hit=hit,
            )

        # Aggregate
        n = len(per_query_results)
        report = EvalReport(
            k=self._k,
            num_queries=n,
            mean_precision=sum(q.precision_at_k for q in per_query_results) / n,
            mean_recall=sum(q.recall_at_k for q in per_query_results) / n,
            mrr=sum(q.reciprocal_rank for q in per_query_results) / n,
            hit_rate=sum(1 for q in per_query_results if q.hit) / n,
            mean_ndcg=sum(q.ndcg_at_k for q in per_query_results) / n,
            per_query=per_query_results,
        )

        logger.info(
            "eval_complete",
            k=self._k,
            num_queries=n,
            mean_precision=f"{report.mean_precision:.3f}",
            mean_recall=f"{report.mean_recall:.3f}",
            mrr=f"{report.mrr:.3f}",
            hit_rate=f"{report.hit_rate:.3f}",
            mean_ndcg=f"{report.mean_ndcg:.3f}",
        )

        return report


# --------------------------------------------------------------------------- #
# Dataset I/O
# --------------------------------------------------------------------------- #


def load_eval_dataset(path: Path) -> list[EvalQuery]:
    """Load evaluation queries from a JSON file.

    Expected format:
    [
        {
            "query": "What was Apple's revenue in 2024?",
            "relevant_chunk_ids": ["aapl_rev_001", "aapl_rev_002"],
            "metadata": {"category": "numerical_extraction"}
        },
        ...
    ]

    Args:
        path: Path to the JSON eval dataset.

    Returns:
        List of EvalQuery objects.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
    """
    if not path.exists():
        msg = f"Eval dataset not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        data = json.load(f)

    queries = [
        EvalQuery(
            query=item["query"],
            relevant_chunk_ids=set(item["relevant_chunk_ids"]),
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]

    logger.info("eval_dataset_loaded", path=str(path), num_queries=len(queries))
    return queries


def save_eval_report(report: EvalReport, path: Path) -> None:
    """Save an evaluation report to JSON.

    Args:
        report: The EvalReport to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "k": report.k,
        "num_queries": report.num_queries,
        "mean_precision": report.mean_precision,
        "mean_recall": report.mean_recall,
        "mrr": report.mrr,
        "hit_rate": report.hit_rate,
        "mean_ndcg": report.mean_ndcg,
        "per_query": [
            {
                "query": qr.query,
                "precision_at_k": qr.precision_at_k,
                "recall_at_k": qr.recall_at_k,
                "reciprocal_rank": qr.reciprocal_rank,
                "hit": qr.hit,
                "ndcg_at_k": qr.ndcg_at_k,
                "retrieved_ids": qr.retrieved_ids,
                "relevant_ids": sorted(qr.relevant_ids),
            }
            for qr in report.per_query
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("eval_report_saved", path=str(path))
