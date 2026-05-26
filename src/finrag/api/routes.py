"""API routes for the FinRAG pipeline.

Endpoints:
    POST /api/v1/query         -- Synchronous JSON response
    POST /api/v1/query/stream  -- SSE streaming response
    GET  /api/v1/sessions/{id} -- Session state inspection
    DELETE /api/v1/sessions/{id} -- Clear a session
    GET  /api/v1/config/prompts -- Active prompt versions

Design decisions:
- SSE over WebSocket: simpler, works through proxies, sufficient
  for server-to-client streaming. Client never streams back.
- Session ID auto-generation: if client omits session_id, we
  generate one. Simplifies client while enabling multi-turn.
- Streaming granularity: structured events at each pipeline stage
  so clients can show progressive UI. Token-level streaming needs
  LangChain streaming callbacks (tracked as DAY-11-002).

Debt: DAY-11-002 -- SSE simulates chunking of final answer. True
      token streaming requires LangChain callbacks. Add Day 12.
"""

import asyncio
import json
import uuid
import time
import hashlib

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, BackgroundTasks
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from finrag.api.dependencies import get_redis_cache
from finrag.core.cache import RedisCache
from finrag.observability.langfuse_tracer import instrument_pipeline_result, metrics
from finrag.orchestration.memory import SessionStore
from finrag.orchestration.prompt_config import get_active_prompt_version

import os
from finrag.api.limiter import limiter

logger = structlog.get_logger(__name__)


def hash_api_key(api_key: str) -> str:
    """Helper to return SHA256 of the API key for secure storage."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


async def log_query(
    api_key_hash: str,
    query_text: str,
    ticker: str | None,
    filing_type: str | None,
    fiscal_period: str | None,
    confidence_score: float | None,
    response_time_ms: int,
    was_declined: bool,
    cache_hit: bool,
    chunk_count: int | None,
) -> None:
    """Background task to asynchronously record query logs and upsert session stats."""
    from finrag.database.connection import AsyncSessionLocal
    from finrag.database.repository import QueryLogRepository, SessionRepository

    try:
        async with AsyncSessionLocal() as db:
            log_repo = QueryLogRepository()
            await log_repo.create_log(
                session=db,
                api_key_hash=api_key_hash,
                query_text=query_text,
                ticker=ticker,
                filing_type=filing_type,
                fiscal_period=fiscal_period,
                confidence_score=confidence_score,
                response_time_ms=response_time_ms,
                was_declined=was_declined,
                cache_hit=cache_hit,
                chunk_count=chunk_count,
            )

            session_repo = SessionRepository()
            await session_repo.upsert_session(
                session=db,
                api_key_hash=api_key_hash,
            )
            await db.commit()
            logger.info("db_query_log_success", api_key_hash=api_key_hash[:8])
    except Exception as e:
        logger.error("db_query_log_failed", error=str(e))



# --------------------------------------------------------------------------- #
# Request/Response Models
# --------------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    """Request body for /query endpoint.

    Attributes:
        query: Natural language financial question.
        session_id: Optional session ID for multi-turn context.
        metadata_filter: Optional retrieval filter.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Financial research question",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for multi-turn. Auto-generated if missing.",
    )
    metadata_filter: dict | None = Field(
        default=None,
        alias="filters",
        description="Metadata filter for retrieval, e.g. {ticker: AAPL}",
    )

    model_config = {
        "populate_by_name": True
    }


class CitationResponse(BaseModel):
    """Single citation in the response.

    Attributes:
        chunk_id: Source chunk identifier.
        filing_reference: Human-readable filing reference.
        section: Filing section.
        page: Page number in filing.
        relevance_score: Reranker score.
    """

    chunk_id: str = ""
    filing_reference: str = ""
    section: str = ""
    page: int | None = None
    relevance_score: float = 0.0
    document_url: str | None = None
    ticker: str = ""
    filing_type: str = ""
    filing_date: str = ""


class QueryResponse(BaseModel):
    """Response body for /query endpoint.

    Attributes:
        answer: Generated answer text.
        citations: Supporting citations.
        session_id: Session ID used.
        confidence: Overall confidence score.
        route: Pipeline route taken.
        prompt_version: Prompt config version used.
        metadata: Pipeline metadata.
    """

    answer: str = ""
    citations: list[CitationResponse] = []
    session_id: str = ""
    confidence: float = 0.0
    route: str = ""
    prompt_version: str = ""
    metadata: dict = {}
    cached: bool = False


class SessionResponse(BaseModel):
    """Response body for session inspection.

    Attributes:
        session_id: The session identifier.
        turn_count: Conversation turns so far.
        entities: Entities discussed across turns.
        filings: Filing types referenced.
        periods: Time periods mentioned.
        cited_chunks: Unique chunks cited count.
    """

    session_id: str
    turn_count: int = 0
    entities: list[str] = []
    filings: list[str] = []
    periods: list[str] = []
    cited_chunks: int = 0


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/api/v1", tags=["FinRAG"])


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


def get_session_store(request: Request) -> SessionStore:
    """Get SessionStore from app state.

    Args:
        request: Incoming request.

    Returns:
        Shared SessionStore instance.
    """
    return request.app.state.session_store


def get_compiled_graph(request: Request):
    """Get compiled RAG graph from app state.

    Args:
        request: Incoming request.

    Returns:
        Compiled LangGraph pipeline or None.
    """
    return getattr(request.app.state, "compiled_graph", None)


# --------------------------------------------------------------------------- #
# POST /query
# --------------------------------------------------------------------------- #


@router.post("/query", response_model=QueryResponse)
@limiter.limit(os.getenv("RATE_LIMIT_PER_MINUTE", "10") + "/minute")
async def query_endpoint(
    body: QueryRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    session_store: SessionStore = Depends(get_session_store),
    compiled_graph=Depends(get_compiled_graph),
    cache: RedisCache = Depends(get_redis_cache),
) -> QueryResponse:
    """Run a financial research query through the RAG pipeline.

    Executes the full pipeline synchronously. If session_id provided,
    conversation history is injected for multi-turn context.

    Args:
        body: Query request body.
        request: HTTP request.
        response: HTTP response.
        background_tasks: Background tasks orchestrator.
        session_store: Shared session store.
        compiled_graph: Compiled RAG graph.
        cache: Shared Redis cache.

    Returns:
        QueryResponse with answer, citations, metadata.
    """
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    session_id = body.session_id or str(uuid.uuid4())

    logger.info(
        "query_received",
        request_id=request_id,
        query_preview=body.query[:80],
        session_id=session_id,
        has_filter=body.metadata_filter is not None,
    )

    # 1. Check Redis cache
    cached_response = await cache.get(body.query, body.metadata_filter)
    if cached_response is not None:
        response.headers["X-Cache"] = "HIT"
        session = session_store.get_or_create(session_id)
        
        # Keep raw citations as dicts for add_turn
        raw_citations = cached_response.get("citations", [])
        session.add_turn(
            query=body.query,
            answer=cached_response.get("answer", ""),
            citations=raw_citations,
            metadata_filter=body.metadata_filter,
        )
        
        # Build CitationResponse objects for QueryResponse returning
        citations = []
        for c in raw_citations:
            citations.append(
                CitationResponse(
                    chunk_id=c.get("chunk_id", ""),
                    filing_reference=c.get("filing_reference", ""),
                    section=c.get("section", ""),
                    page=c.get("page"),
                    relevance_score=c.get("relevance_score", 0.0),
                )
            )
            
        res = QueryResponse(
            answer=cached_response.get("answer", ""),
            citations=citations,
            session_id=session_id,
            confidence=cached_response.get("confidence", 0.0),
            route=cached_response.get("route", "unknown"),
            prompt_version=cached_response.get("prompt_version", "unknown"),
            metadata=cached_response.get("metadata", {}),
            cached=True,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        api_key = getattr(request.state, "api_key", None)
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]
            else:
                api_key = request.headers.get("X-API-Key", "anonymous")
        background_tasks.add_task(
            log_query,
            api_key_hash=hash_api_key(api_key),
            query_text=body.query,
            ticker=body.metadata_filter.get("ticker") if body.metadata_filter else None,
            filing_type=body.metadata_filter.get("filing_type") if body.metadata_filter else None,
            fiscal_period=body.metadata_filter.get("fiscal_period") if body.metadata_filter else None,
            confidence_score=res.confidence,
            response_time_ms=elapsed_ms,
            was_declined=res.route == "decline",
            cache_hit=True,
            chunk_count=len(res.citations),
        )
        return res

    # Cache miss
    response.headers["X-Cache"] = "MISS"

    session = session_store.get_or_create(session_id)
    resolved_query = session.resolve_references(body.query)
    conversation_history = session.get_conversation_history(max_turns=5)

    if compiled_graph is not None:
        try:
            result = await asyncio.to_thread(
                compiled_graph.invoke,
                {
                    "query": resolved_query,
                    "metadata_filter": body.metadata_filter,
                    "conversation_history": conversation_history,
                    "step_count": 0,
                    "max_steps": 15,
                    "messages": [],
                },
            )
        except Exception as e:
            logger.error("pipeline_error", error=str(e), request_id=request_id)
            raise HTTPException(status_code=500, detail=f"Pipeline error: {e!s}") from e
    else:
        result = {
            "answer": "Pipeline not initialized. This is a stub response.",
            "citations": [],
            "route": "stub",
            "is_valid": True,
        }

    # Instrument trace (no-op if Langfuse not configured)
    trace_summary = instrument_pipeline_result(
        result=result,
        request_id=request_id,
        session_id=session_id,
        query=body.query,
    )

    answer = result.get("answer", "")
    citations = result.get("citations", [])
    session.add_turn(
        query=body.query,
        answer=answer,
        citations=citations if isinstance(citations, list) else [],
        metadata_filter=body.metadata_filter,
    )

    citation_responses = []
    for c in citations:
        if isinstance(c, dict):
            citation_responses.append(
                CitationResponse(
                    chunk_id=c.get("chunk_id", ""),
                    filing_reference=c.get("filing_reference", ""),
                    section=c.get("section", ""),
                    page=c.get("page"),
                    relevance_score=c.get("relevance_score", 0.0),
                    document_url=c.get("document_url"),
                    ticker=c.get("ticker", ""),
                    filing_type=c.get("filing_type", ""),
                    filing_date=c.get("filing_date", ""),
                )
            )

    prompt_versions = get_active_prompt_version()

    res = QueryResponse(
        answer=answer,
        citations=citation_responses,
        session_id=session_id,
        confidence=result.get("confidence", result.get("route_confidence", 0.0)),
        route=result.get("route", "unknown"),
        prompt_version=prompt_versions.get("generation", "unknown"),
        metadata={
            "request_id": request_id,
            "step_count": result.get("step_count", 0),
            "is_valid": result.get("is_valid", False),
            "input_blocked": result.get("input_guard_blocked", False),
            "output_blocked": result.get("output_guard_blocked", False),
            "trace_id": trace_summary.get("trace_id", ""),
            "total_latency_ms": trace_summary.get("total_latency_ms", 0),
        },
        cached=False,
    )

    await cache.set(body.query, body.metadata_filter, res.model_dump())

    elapsed_ms = int((time.time() - start_time) * 1000)
    api_key = getattr(request.state, "api_key", None)
    if not api_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
        else:
            api_key = request.headers.get("X-API-Key", "anonymous")
    background_tasks.add_task(
        log_query,
        api_key_hash=hash_api_key(api_key),
        query_text=body.query,
        ticker=body.metadata_filter.get("ticker") if body.metadata_filter else None,
        filing_type=body.metadata_filter.get("filing_type") if body.metadata_filter else None,
        fiscal_period=body.metadata_filter.get("fiscal_period") if body.metadata_filter else None,
        confidence_score=res.confidence,
        response_time_ms=elapsed_ms,
        was_declined=res.route == "decline",
        cache_hit=False,
        chunk_count=len(res.citations),
    )

    return res


# --------------------------------------------------------------------------- #
# POST /query/stream -- SSE
# --------------------------------------------------------------------------- #


@router.post("/query/stream")
async def query_stream_endpoint(
    body: QueryRequest,
    request: Request,
    session_store: SessionStore = Depends(get_session_store),
    compiled_graph=Depends(get_compiled_graph),
):
    """Stream a query response via Server-Sent Events.

    Emits structured events at each pipeline stage:
    retrieval_start, chunks_found, rerank_done,
    generation_start, answer_chunk, citation, done.

    Args:
        body: Query request body.
        request: HTTP request.
        session_store: Shared session store.
        compiled_graph: Compiled RAG graph.

    Returns:
        EventSourceResponse with SSE event stream.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    session_id = body.session_id or str(uuid.uuid4())

    logger.info(
        "stream_query_received",
        request_id=request_id,
        query_preview=body.query[:80],
        session_id=session_id,
    )

    async def event_generator():
        """Generate SSE events for the query pipeline."""
        session = session_store.get_or_create(session_id)
        resolved_query = session.resolve_references(body.query)
        conversation_history = session.get_conversation_history(max_turns=5)

        yield {
            "event": "retrieval_start",
            "data": json.dumps({"query": body.query, "session_id": session_id}),
        }

        if compiled_graph is not None:
            try:
                result = await asyncio.to_thread(
                    compiled_graph.invoke,
                    {
                        "query": resolved_query,
                        "metadata_filter": body.metadata_filter,
                        "conversation_history": conversation_history,
                        "step_count": 0,
                        "max_steps": 15,
                        "messages": [],
                    },
                )
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)}),
                }
                return
        else:
            result = {
                "answer": "Pipeline not initialized. Stub response.",
                "citations": [],
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "route": "stub",
            }

        retrieved = result.get("retrieved_chunks", [])
        yield {
            "event": "chunks_found",
            "data": json.dumps({"count": len(retrieved)}),
        }

        reranked = result.get("reranked_chunks", [])
        yield {
            "event": "rerank_done",
            "data": json.dumps({"count": len(reranked)}),
        }

        yield {
            "event": "generation_start",
            "data": json.dumps({"route": result.get("route", "unknown")}),
        }

        answer = result.get("answer", "")
        logger.info(
            "stream_answer_preview",
            answer_len=len(answer),
            answer_preview=answer[:120] if answer else "<empty>",
            route=result.get("route", "unknown"),
            citations_count=len(result.get("citations", [])),
        )

        # Detect quota/rate-limit answers and surface them as proper errors
        # instead of silently empty answers that trigger "Insufficient Evidence"
        _answer_lower = answer.lower()
        _is_quota_answer = any(kw in _answer_lower for kw in (
            "rate-limited", "quota exceeded", "resource_exhausted",
            "generation failed: missing api key", "generation failed after retries",
        ))
        if _is_quota_answer:
            yield {
                "event": "error",
                "data": json.dumps({"error": answer}),
            }
            return

        chunk_size = 80
        if answer:  # Only send chunks if there is actual content
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i : i + chunk_size]
                yield {
                    "event": "answer_chunk",
                    "data": json.dumps({"text": chunk, "index": i // chunk_size}),
                }
                await asyncio.sleep(0.02)

        citations = result.get("citations", [])
        for c in citations:
            if isinstance(c, dict):
                yield {"event": "citation", "data": json.dumps(c)}

        session.add_turn(
            query=body.query,
            answer=answer,
            citations=citations if isinstance(citations, list) else [],
            metadata_filter=body.metadata_filter,
        )

        # Instrument trace for the streaming endpoint
        instrument_pipeline_result(
            result=result,
            request_id=request_id,
            session_id=session_id,
            query=body.query,
        )

        # Determine is_valid for the frontend:
        # - A real answer with citations = valid (even if enforcement had minor issues)
        # - route == "decline" or "stub" with empty answer = not valid
        route = result.get("route", "unknown")
        has_real_answer = bool(answer.strip()) and route not in ("decline", "stub", "blocked")
        effective_is_valid = has_real_answer or (route == "decline")  # decline is intentional, not an error

        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "session_id": session_id,
                    "route": route,
                    "is_valid": effective_is_valid,
                    "is_declined": route == "decline",
                    "total_citations": len(citations),
                    "request_id": request_id,
                    "confidence": result.get("confidence", result.get("route_confidence", 0.0)),
                }
            ),
        }

    return EventSourceResponse(event_generator())


# --------------------------------------------------------------------------- #
# GET /sessions/{session_id}
# --------------------------------------------------------------------------- #


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
) -> SessionResponse:
    """Get the state of a conversation session.

    Args:
        session_id: The session identifier.
        session_store: Shared session store.

    Returns:
        SessionResponse with session state.

    Raises:
        HTTPException: 404 if session not found.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    state = session.to_dict()
    return SessionResponse(
        session_id=state["session_id"],
        turn_count=state["turn_count"],
        entities=state["entities"],
        filings=state["filings"],
        periods=state["periods"],
        cited_chunks=state["cited_chunks"],
    )


# --------------------------------------------------------------------------- #
# DELETE /sessions/{session_id}
# --------------------------------------------------------------------------- #


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
) -> dict:
    """Delete a conversation session.

    Args:
        session_id: The session identifier.
        session_store: Shared session store.

    Returns:
        Confirmation dict.

    Raises:
        HTTPException: 404 if session not found.
    """
    deleted = session_store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return {"detail": f"Session '{session_id}' deleted", "session_id": session_id}


# --------------------------------------------------------------------------- #
# GET /available-filings — what data is actually in ChromaDB
# --------------------------------------------------------------------------- #


@router.get("/available-filings")
async def get_available_filings(request: Request) -> dict:
    """Return available ticker/form_type/filing_date combinations from ChromaDB.

    Introspects the vector store so the frontend can show only options
    that have real data — no more phantom period selections.

    Returns:
        Dict mapping (ticker, form_type) to list of available filing_dates.
        Example: {"AAPL": {"10-K": ["2025-10-30"], "10-Q": ["2026-01-30", "2026-05-01"]}}
    """
    chroma_store = getattr(request.app.state, "chroma_store", None)
    if chroma_store is None:
        return {"available": {}, "total_chunks": 0}

    try:
        # Sample a large number of chunks to get all unique combinations
        result = chroma_store._collection.get(
            include=["metadatas"],
            limit=5000,
        )
        metadatas = result.get("metadatas", [])

        # Build available filings map: {ticker: {form_type: set(filing_dates)}}
        available: dict = {}
        for meta in metadatas:
            ticker = meta.get("ticker", "")
            form_type = meta.get("form_type", "")
            filing_date = meta.get("filing_date", "")
            if not ticker or not form_type:
                continue
            if ticker not in available:
                available[ticker] = {}
            if form_type not in available[ticker]:
                available[ticker][form_type] = set()
            if filing_date:
                available[ticker][form_type].add(filing_date)

        # Convert sets to sorted lists (most recent first)
        serializable = {
            ticker: {
                form_type: sorted(list(dates), reverse=True)
                for form_type, dates in form_types.items()
            }
            for ticker, form_types in available.items()
        }

        return {
            "available": serializable,
            "total_chunks": len(metadatas),
        }

    except Exception as e:
        logger.warning("available_filings_error", error=str(e))
        return {"available": {}, "total_chunks": 0, "error": str(e)}


# --------------------------------------------------------------------------- #
# GET /config/prompts
# --------------------------------------------------------------------------- #


@router.get("/config/prompts")
async def get_prompt_config() -> dict:
    """Return active prompt configuration versions.

    Returns:
        Dict with generation and retrieval prompt versions.
    """
    versions = get_active_prompt_version()
    return {
        "prompt_versions": versions,
        "status": "loaded" if versions.get("generation") != "not_loaded" else "not_loaded",
    }


# --------------------------------------------------------------------------- #
# GET /metrics -- Production metrics
# --------------------------------------------------------------------------- #


@router.get("/metrics")
async def get_metrics() -> dict:
    """Return production metrics summary.

    Includes p50/p95 latency, cost, token usage, and operational
    rates (decline, citation coverage, guard blocks).

    Returns:
        Dict with full metrics summary.
    """
    return metrics.get_summary()


# --------------------------------------------------------------------------- #
# GET /analytics/queries
# --------------------------------------------------------------------------- #


@router.get("/analytics/queries")
async def get_analytics_endpoint(
    ticker: str | None = None,
    days: int = 7,
) -> dict:
    """Return aggregated query analytics for a given ticker or duration.

    Args:
        ticker: Optional company ticker to filter by.
        days: Number of past days to aggregate over (default 7).

    Returns:
        Aggregated analytics summary.
    """
    from finrag.database.connection import AsyncSessionLocal
    from finrag.database.repository import QueryLogRepository

    async with AsyncSessionLocal() as session:
        log_repo = QueryLogRepository()
        return await log_repo.get_analytics(session=session, ticker=ticker, days=days)
