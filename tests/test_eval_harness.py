"""Tests for retrieval evaluation harness.

Tests cover:
- Individual metric functions (precision, recall, MRR, NDCG)
- Evaluator with mock retriever
- Dataset I/O (load/save round-trip)
- Edge cases (empty queries, no relevant docs, perfect retrieval)
- Report aggregation correctness
"""

import json
from pathlib import Path

import pytest

from finrag.retrieval.eval_harness import (
    EvalQuery,
    RetrievalEvaluator,
    load_eval_dataset,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    save_eval_report,
)


# --------------------------------------------------------------------------- #
# Metric Function Tests
# --------------------------------------------------------------------------- #


class TestPrecisionAtK:
    """Tests for precision@k metric."""

    def test_perfect_precision(self) -> None:
        """All retrieved docs are relevant."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_precision(self) -> None:
        """No retrieved docs are relevant."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_precision(self) -> None:
        """Some retrieved docs are relevant."""
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        assert abs(precision_at_k(retrieved, relevant, 3) - 2 / 3) < 1e-9

    def test_k_less_than_retrieved(self) -> None:
        """k is less than total retrieved."""
        retrieved = ["a", "b", "x", "y"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 2) == 1.0

    def test_k_zero(self) -> None:
        """k=0 returns 0."""
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestRecallAtK:
    """Tests for recall@k metric."""

    def test_perfect_recall(self) -> None:
        """All relevant docs are retrieved."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_recall(self) -> None:
        """No relevant docs are retrieved."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_recall(self) -> None:
        """Some relevant docs are retrieved."""
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b", "c"}
        assert abs(recall_at_k(retrieved, relevant, 3) - 1 / 3) < 1e-9

    def test_empty_relevant(self) -> None:
        """No relevant docs defined returns 0."""
        assert recall_at_k(["a"], set(), 1) == 0.0


class TestReciprocalRank:
    """Tests for reciprocal rank metric."""

    def test_first_position(self) -> None:
        """Relevant doc at rank 1."""
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self) -> None:
        """Relevant doc at rank 2."""
        assert reciprocal_rank(["x", "a", "c"], {"a"}) == 0.5

    def test_third_position(self) -> None:
        """Relevant doc at rank 3."""
        assert abs(reciprocal_rank(["x", "y", "a"], {"a"}) - 1 / 3) < 1e-9

    def test_no_relevant(self) -> None:
        """No relevant docs in results."""
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant(self) -> None:
        """Multiple relevant docs — uses first occurrence."""
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5


class TestNDCGAtK:
    """Tests for NDCG@k metric."""

    def test_perfect_ndcg(self) -> None:
        """All relevant docs at top positions."""
        retrieved = ["a", "b", "x"]
        relevant = {"a", "b"}
        assert abs(ndcg_at_k(retrieved, relevant, 3) - 1.0) < 1e-9

    def test_zero_ndcg(self) -> None:
        """No relevant docs retrieved."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, 3) == 0.0

    def test_lower_rank_worse_ndcg(self) -> None:
        """Relevant doc at lower rank gives worse NDCG."""
        # Relevant at rank 1
        ndcg_top = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        # Relevant at rank 3
        ndcg_bottom = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert ndcg_top > ndcg_bottom

    def test_empty_relevant(self) -> None:
        """No relevant docs defined."""
        assert ndcg_at_k(["a", "b"], set(), 2) == 0.0

    def test_k_zero(self) -> None:
        """k=0 returns 0."""
        assert ndcg_at_k(["a"], {"a"}, 0) == 0.0


# --------------------------------------------------------------------------- #
# Evaluator Tests
# --------------------------------------------------------------------------- #


class TestRetrievalEvaluator:
    """Tests for the RetrievalEvaluator."""

    def test_perfect_retrieval(self) -> None:
        """Evaluator gives perfect scores when retriever returns exact matches."""

        def mock_retriever(query: str, n: int) -> list[dict]:
            return [
                {"chunk_id": "a"},
                {"chunk_id": "b"},
            ]

        evaluator = RetrievalEvaluator(retriever_fn=mock_retriever, k=2)
        queries = [
            EvalQuery(query="test query", relevant_chunk_ids={"a", "b"}),
        ]
        report = evaluator.evaluate(queries)

        assert report.mean_precision == 1.0
        assert report.mean_recall == 1.0
        assert report.mrr == 1.0
        assert report.hit_rate == 1.0

    def test_zero_retrieval(self) -> None:
        """Evaluator gives zero scores when retriever returns wrong docs."""

        def mock_retriever(query: str, n: int) -> list[dict]:
            return [
                {"chunk_id": "x"},
                {"chunk_id": "y"},
            ]

        evaluator = RetrievalEvaluator(retriever_fn=mock_retriever, k=2)
        queries = [
            EvalQuery(query="test query", relevant_chunk_ids={"a", "b"}),
        ]
        report = evaluator.evaluate(queries)

        assert report.mean_precision == 0.0
        assert report.mean_recall == 0.0
        assert report.mrr == 0.0
        assert report.hit_rate == 0.0

    def test_mixed_retrieval(self) -> None:
        """Evaluator computes correct averages across queries."""

        call_count = 0

        def mock_retriever(query: str, n: int) -> list[dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"chunk_id": "a"}, {"chunk_id": "b"}]  # Perfect
            else:
                return [{"chunk_id": "x"}, {"chunk_id": "y"}]  # Miss

        evaluator = RetrievalEvaluator(retriever_fn=mock_retriever, k=2)
        queries = [
            EvalQuery(query="good query", relevant_chunk_ids={"a", "b"}),
            EvalQuery(query="bad query", relevant_chunk_ids={"c", "d"}),
        ]
        report = evaluator.evaluate(queries)

        assert report.num_queries == 2
        assert report.mean_precision == 0.5  # (1.0 + 0.0) / 2
        assert report.hit_rate == 0.5  # 1 hit out of 2 queries

    def test_empty_queries(self) -> None:
        """Empty query list returns zero report."""

        def mock_retriever(query: str, n: int) -> list[dict]:
            return []

        evaluator = RetrievalEvaluator(retriever_fn=mock_retriever, k=5)
        report = evaluator.evaluate([])

        assert report.num_queries == 0
        assert report.mean_precision == 0.0

    def test_report_has_per_query(self) -> None:
        """Report includes per-query results."""

        def mock_retriever(query: str, n: int) -> list[dict]:
            return [{"chunk_id": "a"}]

        evaluator = RetrievalEvaluator(retriever_fn=mock_retriever, k=1)
        queries = [
            EvalQuery(query="q1", relevant_chunk_ids={"a"}),
            EvalQuery(query="q2", relevant_chunk_ids={"b"}),
        ]
        report = evaluator.evaluate(queries)

        assert len(report.per_query) == 2
        assert report.per_query[0].query == "q1"
        assert report.per_query[1].query == "q2"


# --------------------------------------------------------------------------- #
# Dataset I/O Tests
# --------------------------------------------------------------------------- #


class TestDatasetIO:
    """Tests for load/save of eval datasets and reports."""

    def test_load_eval_dataset(self, tmp_path: Path) -> None:
        """Load eval dataset from JSON."""
        data = [
            {
                "query": "What was Apple's revenue?",
                "relevant_chunk_ids": ["aapl_rev_001"],
                "metadata": {"category": "numerical"},
            },
            {
                "query": "What are the risk factors?",
                "relevant_chunk_ids": ["aapl_risk_001", "aapl_risk_002"],
            },
        ]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data))

        queries = load_eval_dataset(path)
        assert len(queries) == 2
        assert queries[0].query == "What was Apple's revenue?"
        assert "aapl_rev_001" in queries[0].relevant_chunk_ids
        assert queries[1].metadata == {}

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Loading nonexistent dataset raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_eval_dataset(tmp_path / "nope.json")

    def test_save_and_load_report(self, tmp_path: Path) -> None:
        """Save report to JSON and verify it's valid JSON."""

        def mock_retriever(query: str, n: int) -> list[dict]:
            return [{"chunk_id": "a"}]

        evaluator = RetrievalEvaluator(retriever_fn=mock_retriever, k=1)
        queries = [EvalQuery(query="test", relevant_chunk_ids={"a"})]
        report = evaluator.evaluate(queries)

        path = tmp_path / "report.json"
        save_eval_report(report, path)
        assert path.exists()

        with open(path) as f:
            saved = json.load(f)

        assert saved["k"] == 1
        assert saved["num_queries"] == 1
        assert saved["mean_precision"] == 1.0
        assert len(saved["per_query"]) == 1
