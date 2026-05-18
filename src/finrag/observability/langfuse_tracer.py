"""Langfuse instrumentation for the FinRAG pipeline.

Provides distributed tracing, cost tracking, and production metrics
for every RAG pipeline execution. Each request produces a full trace
showing: retrieval, reranking, generation, citation enforcement, and
guard activity with timing and token counts.

Trace structure per request:
    trace (top-level)
    +-- span: input_guard       (latency, blocked?)
    +-- span: retrieval         (query, chunks found, latency)
    +-- span: reranking         (top-k scores, latency)
    +-- generation              (model, prompt, tokens, cost)
    +-- span: citation_enforce  (pass/fail, errors)
    +-- span: output_guard      (redactions, disclaimers)
    +-- score: faithfulness     (0-1, from validation)
    +-- score: citation_coverage (0-1, chunks cited / chunks used)

Production metrics tracked:
- p50/p95 latency per pipeline stage
- Cost per request (input + output tokens * model price)
- Citation coverage rate (% of answers with valid citations)
- Decline rate (% of queries refused)
- Guard block rate (input + output)

Design decisions:
- Wrapper pattern: FinRAGTracer wraps Langfuse client. If Langfuse
  is unavailable (no keys, network down), all methods are no-ops.
  The pipeline never crashes due to observability failures.
- Lazy initialization: Langfuse client created on first use, not
  at import time. Avoids startup failures in test environments.
- Metric aggregation via MetricsCollector: in-memory running stats
  (count, sum, min, max, percentiles) flushed to Langfuse scores
  periodically. Avoids per-request overhead of score computation.
- Token cost estimation: uses per-model pricing table. Not exact
  (Gemini pricing changes), but directionally correct for budgeting.

Environment variables:
    LANGFUSE_PUBLIC_KEY  -- Langfuse project public key
    LANGFUSE_SECRET_KEY  -- Langfuse project secret key
    LANGFUSE_HOST        -- Langfuse server URL (default: cloud)
    FINRAG_TRACING       -- Enable/disable tracing (default: true)
"""

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Token Cost Table (USD per 1K tokens)
# --------------------------------------------------------------------------- #

# Approximate pricing for supported models.
# Updated as of early 2025. Check provider pricing for actuals.
MODEL_COSTS = {
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-2.0-flash-lite": {"input": 0.00005, "output": 0.0002},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-flash-8b": {"input": 0.0000375, "output": 0.00015},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.0006},
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {"input": 0.0, "output": 0.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a model invocation.

    Args:
        model: Model name.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    costs = MODEL_COSTS.get(model, {"input": 0.0001, "output": 0.0004})
    return (input_tokens * costs["input"] / 1000) + (output_tokens * costs["output"] / 1000)


# --------------------------------------------------------------------------- #
# Failure Classification
# --------------------------------------------------------------------------- #


class FailureType(StrEnum):
    """Classification of pipeline failures for structured logging."""

    RETRIEVAL_EMPTY = "retrieval_empty"
    RERANKER_LOW_SCORE = "reranker_low_score"
    GENERATION_ERROR = "generation_error"
    CITATION_ENFORCEMENT = "citation_enforcement"
    INPUT_GUARD_BLOCK = "input_guard_block"
    OUTPUT_GUARD_BLOCK = "output_guard_block"
    VALIDATION_FAIL = "validation_fail"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


def classify_failure(result: dict) -> FailureType | None:
    """Classify a pipeline result into a failure type.

    Args:
        result: Final pipeline state dict.

    Returns:
        FailureType or None if no failure.
    """
    if result.get("input_guard_blocked"):
        return FailureType.INPUT_GUARD_BLOCK
    if result.get("output_guard_blocked"):
        return FailureType.OUTPUT_GUARD_BLOCK
    if result.get("error"):
        error = result["error"].lower()
        if "timeout" in error:
            return FailureType.TIMEOUT
        if "generation" in error or "llm" in error:
            return FailureType.GENERATION_ERROR
        return FailureType.UNKNOWN
    if result.get("route") == "decline":
        return FailureType.RERANKER_LOW_SCORE
    if not result.get("retrieved_chunks"):
        return FailureType.RETRIEVAL_EMPTY
    if not result.get("is_valid") and result.get("validation_errors"):
        errors = result.get("validation_errors", [])
        if any("citation" in str(e).lower() for e in errors):
            return FailureType.CITATION_ENFORCEMENT
        return FailureType.VALIDATION_FAIL
    return None


# --------------------------------------------------------------------------- #
# Metrics Collector
# --------------------------------------------------------------------------- #


@dataclass
class MetricBucket:
    """Running statistics for a single metric.

    Attributes:
        count: Number of observations.
        total: Sum of all values.
        min_val: Minimum observed value.
        max_val: Maximum observed value.
        values: Recent values for percentile calculation.
    """

    count: int = 0
    total: float = 0.0
    min_val: float = float("inf")
    max_val: float = 0.0
    values: list[float] = field(default_factory=list)

    def record(self, value: float) -> None:
        """Record a new observation.

        Args:
            value: The metric value to record.
        """
        self.count += 1
        self.total += value
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        # Keep last 1000 values for percentile calculation
        self.values.append(value)
        if len(self.values) > 1000:
            self.values = self.values[-1000:]

    @property
    def mean(self) -> float:
        """Average value."""
        return self.total / self.count if self.count else 0.0

    def percentile(self, p: float) -> float:
        """Calculate percentile from recent values.

        Args:
            p: Percentile (0-100).

        Returns:
            The p-th percentile value.
        """
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def to_dict(self) -> dict:
        """Serialize to dict.

        Returns:
            Dict with count, mean, min, max, p50, p95, p99.
        """
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "min": round(self.min_val, 4) if self.count else 0.0,
            "max": round(self.max_val, 4),
            "p50": round(self.percentile(50), 4),
            "p95": round(self.percentile(95), 4),
            "p99": round(self.percentile(99), 4),
        }


class MetricsCollector:
    """In-memory metrics aggregator for production monitoring.

    Tracks running statistics for latency, cost, and quality metrics.
    Thread-safe via simple dict operations (GIL protected for CPython).

    Attributes:
        _buckets: Dict of metric_name to MetricBucket.
        _counters: Dict of counter_name to int.
    """

    def __init__(self) -> None:
        """Initialize empty metrics collector."""
        self._buckets: dict[str, MetricBucket] = defaultdict(MetricBucket)
        self._counters: dict[str, int] = defaultdict(int)

    def record_latency(self, stage: str, latency_ms: float) -> None:
        """Record latency for a pipeline stage.

        Args:
            stage: Pipeline stage name (retrieval, rerank, generate, etc).
            latency_ms: Latency in milliseconds.
        """
        self._buckets[f"latency_{stage}"].record(latency_ms)

    def record_cost(self, cost_usd: float) -> None:
        """Record cost for a request.

        Args:
            cost_usd: Cost in USD.
        """
        self._buckets["cost_per_request"].record(cost_usd)

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage.

        Args:
            input_tokens: Input token count.
            output_tokens: Output token count.
        """
        self._buckets["input_tokens"].record(input_tokens)
        self._buckets["output_tokens"].record(output_tokens)

    def increment(self, counter: str) -> None:
        """Increment a counter.

        Args:
            counter: Counter name (e.g. total_requests, declines, blocks).
        """
        self._counters[counter] += 1

    def get_rates(self) -> dict:
        """Calculate operational rates.

        Returns:
            Dict with decline_rate, citation_coverage, block_rate.
        """
        total = self._counters.get("total_requests", 0)
        if total == 0:
            return {
                "decline_rate": 0.0,
                "citation_coverage": 0.0,
                "input_block_rate": 0.0,
                "output_block_rate": 0.0,
            }

        return {
            "decline_rate": round(self._counters.get("declines", 0) / total, 4),
            "citation_coverage": round(self._counters.get("cited_responses", 0) / total, 4),
            "input_block_rate": round(self._counters.get("input_blocks", 0) / total, 4),
            "output_block_rate": round(self._counters.get("output_blocks", 0) / total, 4),
        }

    def get_summary(self) -> dict:
        """Full metrics summary.

        Returns:
            Dict with all buckets and rates.
        """
        return {
            "latencies": {
                name.replace("latency_", ""): bucket.to_dict()
                for name, bucket in self._buckets.items()
                if name.startswith("latency_")
            },
            "costs": self._buckets["cost_per_request"].to_dict(),
            "tokens": {
                "input": self._buckets["input_tokens"].to_dict(),
                "output": self._buckets["output_tokens"].to_dict(),
            },
            "counters": dict(self._counters),
            "rates": self.get_rates(),
        }

    def reset(self) -> None:
        """Reset all metrics. Used in tests."""
        self._buckets.clear()
        self._counters.clear()


# Singleton metrics collector
metrics = MetricsCollector()


# --------------------------------------------------------------------------- #
# Langfuse Tracer
# --------------------------------------------------------------------------- #


class FinRAGTracer:
    """Wrapper around Langfuse client for FinRAG pipeline tracing.

    All methods are no-ops if Langfuse is not configured or unavailable.
    The pipeline never crashes due to tracing failures.

    Usage:
        tracer = FinRAGTracer()
        trace = tracer.start_trace(query="What was AAPL revenue?")
        span = tracer.start_span(trace, "retrieval")
        # ... do retrieval ...
        tracer.end_span(span, metadata={...})
        tracer.end_trace(trace, result={...})

    Attributes:
        _client: Langfuse client instance (or None).
        _enabled: Whether tracing is active.
    """

    def __init__(self) -> None:
        """Initialize tracer. Connects to Langfuse if keys are set."""
        self._client = None
        self._enabled = os.environ.get("FINRAG_TRACING", "true").lower() == "true"
        self._initialized = False

    def _ensure_client(self) -> bool:
        """Lazily initialize Langfuse client.

        Returns:
            True if client is available.
        """
        if self._initialized:
            return self._client is not None

        self._initialized = True

        if not self._enabled:
            logger.info("tracing_disabled", reason="FINRAG_TRACING=false")
            return False

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

        if not public_key or not secret_key:
            logger.info("tracing_disabled", reason="LANGFUSE keys not set")
            return False

        try:
            from langfuse import Langfuse

            host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            logger.info("langfuse_connected", host=host)
            return True

        except ImportError:
            logger.warning("langfuse_not_installed", hint="pip install langfuse")
            return False
        except Exception as e:
            logger.error("langfuse_init_failed", error=str(e))
            return False

    @property
    def enabled(self) -> bool:
        """Whether tracing is currently active and Langfuse is connected."""
        return self._ensure_client()

    def start_trace(
        self,
        query: str,
        session_id: str = "",
        request_id: str = "",
        metadata: dict | None = None,
    ) -> dict:
        """Start a new trace for a pipeline execution.

        Args:
            query: The user's query.
            session_id: Session identifier.
            request_id: HTTP request identifier.
            metadata: Additional trace metadata.

        Returns:
            Trace context dict (opaque, pass to other methods).
        """
        trace_ctx = {
            "trace_id": request_id or str(time.time()),
            "start_time": time.time(),
            "query": query,
            "session_id": session_id,
            "spans": {},
            "_langfuse_trace": None,
        }

        metrics.increment("total_requests")

        if self._ensure_client() and self._client:
            try:
                trace = self._client.trace(
                    id=trace_ctx["trace_id"],
                    name="finrag_query",
                    session_id=session_id or None,
                    input={"query": query},
                    metadata={
                        "request_id": request_id,
                        **(metadata or {}),
                    },
                )
                trace_ctx["_langfuse_trace"] = trace
            except Exception as e:
                logger.debug("trace_start_failed", error=str(e))

        return trace_ctx

    def start_span(
        self,
        trace_ctx: dict,
        name: str,
        metadata: dict | None = None,
    ) -> dict:
        """Start a span within a trace.

        Args:
            trace_ctx: Trace context from start_trace.
            name: Span name (e.g. retrieval, rerank, generate).
            metadata: Span-level metadata.

        Returns:
            Span context dict.
        """
        span_ctx = {
            "name": name,
            "start_time": time.time(),
            "_langfuse_span": None,
        }

        langfuse_trace = trace_ctx.get("_langfuse_trace")
        if langfuse_trace:
            try:
                span = langfuse_trace.span(
                    name=name,
                    metadata=metadata,
                )
                span_ctx["_langfuse_span"] = span
            except Exception as e:
                logger.debug("span_start_failed", name=name, error=str(e))

        trace_ctx["spans"][name] = span_ctx
        return span_ctx

    def end_span(
        self,
        span_ctx: dict,
        output: dict | None = None,
        metadata: dict | None = None,
        level: str = "DEFAULT",
    ) -> float:
        """End a span and record latency.

        Args:
            span_ctx: Span context from start_span.
            output: Span output data.
            metadata: Additional metadata.
            level: Log level (DEFAULT, DEBUG, WARNING, ERROR).

        Returns:
            Latency in milliseconds.
        """
        latency_ms = round((time.time() - span_ctx["start_time"]) * 1000, 2)
        metrics.record_latency(span_ctx["name"], latency_ms)

        langfuse_span = span_ctx.get("_langfuse_span")
        if langfuse_span:
            try:
                langfuse_span.end(
                    output=output,
                    metadata={
                        "latency_ms": latency_ms,
                        **(metadata or {}),
                    },
                    level=level,
                )
            except Exception as e:
                logger.debug("span_end_failed", error=str(e))

        return latency_ms

    def record_generation(
        self,
        trace_ctx: dict,
        model: str,
        prompt: str,
        completion: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        metadata: dict | None = None,
    ) -> None:
        """Record an LLM generation event.

        Args:
            trace_ctx: Trace context.
            model: Model name.
            prompt: System/user prompt sent.
            completion: LLM response text.
            input_tokens: Input token count.
            output_tokens: Output token count.
            metadata: Additional metadata.
        """
        cost = estimate_cost(model, input_tokens, output_tokens)
        metrics.record_cost(cost)
        metrics.record_tokens(input_tokens, output_tokens)

        langfuse_trace = trace_ctx.get("_langfuse_trace")
        if langfuse_trace:
            try:
                langfuse_trace.generation(
                    name="rag_generation",
                    model=model,
                    input=prompt[:2000],
                    output=completion[:2000],
                    usage={
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens,
                    },
                    metadata={
                        "cost_usd": cost,
                        **(metadata or {}),
                    },
                )
            except Exception as e:
                logger.debug("generation_record_failed", error=str(e))

    def score_trace(
        self,
        trace_ctx: dict,
        name: str,
        value: float,
        comment: str = "",
    ) -> None:
        """Add a score to the trace.

        Args:
            trace_ctx: Trace context.
            name: Score name (faithfulness, citation_coverage, etc).
            value: Score value (0-1).
            comment: Optional comment.
        """
        langfuse_trace = trace_ctx.get("_langfuse_trace")
        if langfuse_trace:
            try:
                langfuse_trace.score(
                    name=name,
                    value=value,
                    comment=comment,
                )
            except Exception as e:
                logger.debug("score_failed", name=name, error=str(e))

    def end_trace(
        self,
        trace_ctx: dict,
        result: dict | None = None,
    ) -> dict:
        """End a trace and record final metrics.

        Args:
            trace_ctx: Trace context.
            result: Final pipeline result dict.

        Returns:
            Summary dict with latency, cost, failure info.
        """
        total_latency_ms = round((time.time() - trace_ctx["start_time"]) * 1000, 2)
        metrics.record_latency("total", total_latency_ms)

        result = result or {}
        summary: dict = {
            "trace_id": trace_ctx["trace_id"],
            "total_latency_ms": total_latency_ms,
            "route": result.get("route", "unknown"),
        }

        # Classify and count failures
        failure = classify_failure(result)
        if failure:
            summary["failure_type"] = failure.value
            metrics.increment(f"failure_{failure.value}")

            if failure == FailureType.INPUT_GUARD_BLOCK:
                metrics.increment("input_blocks")
            elif failure == FailureType.OUTPUT_GUARD_BLOCK:
                metrics.increment("output_blocks")

        # Track declines
        if result.get("route") == "decline":
            metrics.increment("declines")

        # Track citation coverage
        citations = result.get("citations", [])
        if citations and result.get("is_valid"):
            metrics.increment("cited_responses")

        # End Langfuse trace
        langfuse_trace = trace_ctx.get("_langfuse_trace")
        if langfuse_trace:
            try:
                langfuse_trace.update(
                    output={
                        "answer": result.get("answer", "")[:500],
                        "route": result.get("route"),
                        "is_valid": result.get("is_valid"),
                        "citation_count": len(citations),
                    },
                    metadata={
                        "total_latency_ms": total_latency_ms,
                        "failure_type": summary.get("failure_type"),
                    },
                )
            except Exception as e:
                logger.debug("trace_end_failed", error=str(e))

        logger.info(
            "trace_completed",
            trace_id=trace_ctx["trace_id"],
            latency_ms=total_latency_ms,
            route=summary["route"],
            failure=summary.get("failure_type"),
        )

        return summary

    def flush(self) -> None:
        """Flush pending Langfuse events."""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.debug("flush_failed", error=str(e))


# Singleton tracer instance
tracer = FinRAGTracer()


# --------------------------------------------------------------------------- #
# Pipeline Instrumentation Helper
# --------------------------------------------------------------------------- #


def instrument_pipeline_result(
    result: dict,
    request_id: str = "",
    session_id: str = "",
    query: str = "",
) -> dict:
    """One-shot instrumentation for a completed pipeline result.

    Convenience function for instrumenting after the pipeline runs.
    Creates a trace, records metrics, and returns the summary.

    Args:
        result: Final pipeline state dict.
        request_id: HTTP request ID.
        session_id: Session ID.
        query: Original query.

    Returns:
        Trace summary dict.
    """
    trace_ctx = tracer.start_trace(
        query=query,
        session_id=session_id,
        request_id=request_id,
        metadata={
            "route": result.get("route"),
            "prompt_version": result.get("prompt_version", "unknown"),
        },
    )

    # Record generation if we have token info
    model = result.get("generation_model", "gemini-2.0-flash")
    if model and model not in ("decline", "error_handler", "input_guard", "stub"):
        tracer.record_generation(
            trace_ctx,
            model=model,
            prompt=result.get("query", query)[:500],
            completion=result.get("answer", "")[:500],
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
        )

    # Score citation coverage
    citations = result.get("citations", [])
    reranked = result.get("reranked_chunks", [])
    if reranked:
        coverage = len(citations) / len(reranked) if reranked else 0.0
        tracer.score_trace(
            trace_ctx,
            name="citation_coverage",
            value=min(coverage, 1.0),
        )

    summary = tracer.end_trace(trace_ctx, result=result)
    tracer.flush()
    return summary
