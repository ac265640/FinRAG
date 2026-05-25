"""FastAPI application factory for the FinRAG pipeline.

Creates and configures the FastAPI app with middleware, routes,
and shared resources. Factory pattern supports different configs
for test vs production.

Startup sequence:
    1. Initialize SessionStore for conversation memory
    2. Load versioned prompt configs from YAML
    3. (Optional) Initialize HybridRetriever, Reranker, RAGGenerator
    4. Compile the LangGraph pipeline
    5. Store everything in app.state for route access

Heavy initialization (embeddings, indexes, LLM clients) only
happens when FINRAG_INIT_PIPELINE=true. In test mode, we skip
pipeline init and use stub responses.

Usage:
    uvicorn finrag.api.app:create_app --factory --reload
"""

import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from finrag.api.mcp_server import mcp_router
from finrag.api.middleware import (
    AuthMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
)
from finrag.api.routes import router as api_router
from finrag.api.ingest_routes import router as ingest_router
from finrag.orchestration.memory import SessionStore
from finrag.orchestration.prompt_config import load_generation_config, load_retrieval_config

from finrag.api.limiter import limiter, custom_rate_limit_handler
from slowapi.errors import RateLimitExceeded
from finrag.core.middleware import RequestIDMiddleware as CoreRequestIDMiddleware
from finrag.core.logging import configure_logging

configure_logging(os.getenv("LOG_LEVEL", "INFO"))

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Application Lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic.

    Startup:
        - Create SessionStore
        - Load prompt configs
        - Optionally initialize the full RAG pipeline

    Shutdown:
        - Log shutdown

    Args:
        app: The FastAPI application instance.
    """
    logger.info("finrag_api_starting")

    # --- Session Store ---
    max_sessions = int(os.environ.get("FINRAG_MAX_SESSIONS", "1000"))
    app.state.session_store = SessionStore(max_sessions=max_sessions)

    # --- Redis Cache ---
    try:
        from finrag.api.dependencies import get_redis_cache_instance
        app.state.redis_cache = get_redis_cache_instance()
        logger.info("redis_cache_initialized")
    except Exception as e:
        logger.error("redis_cache_init_failed", error=str(e))
        app.state.redis_cache = None

    # --- Prompt Configs ---
    prompt_version = os.environ.get("FINRAG_PROMPT_VERSION", "v1")
    try:
        gen_config = load_generation_config(version=prompt_version)
        ret_config = load_retrieval_config(version=prompt_version)
        # Store in app.state so routes can access prompt versions
        app.state.gen_config = gen_config
        app.state.ret_config = ret_config
        logger.info(
            "prompt_configs_loaded",
            generation_version=gen_config.version,
            retrieval_version=ret_config.version,
        )
    except Exception as e:
        logger.warning("prompt_config_load_failed", error=str(e))
        app.state.gen_config = None
        app.state.ret_config = None

    # --- Pipeline Initialization (optional) ---
    init_pipeline = os.environ.get("FINRAG_INIT_PIPELINE", "false").lower() == "true"

    if init_pipeline:
        try:
            from pathlib import Path

            from finrag.ingestion.chunker import chunk_filing_directory
            from finrag.orchestration.generator import RAGGenerator
            from finrag.orchestration.graph import compile_rag_graph
            from finrag.retrieval.bm25_index import BM25Index
            from finrag.retrieval.hybrid import HybridRetriever
            from finrag.retrieval.reranker import CrossEncoderReranker
            from finrag.vectorstore.chroma_store import ChromaStore

            # Build ChromaDB vector store
            chroma_store = ChromaStore()

            # Build BM25 index from all raw filing directories
            raw_dir = Path("./data/raw")
            bm25_index = BM25Index()
            if raw_dir.exists():
                filing_dirs = [d for d in sorted(raw_dir.iterdir()) if d.is_dir() and (d / "metadata.json").exists()]
                if filing_dirs:
                    all_chunks = []
                    for filing_dir in filing_dirs:
                        all_chunks.extend(chunk_filing_directory(filing_dir))
                    if all_chunks:
                        bm25_index.add_chunks(all_chunks)
                        logger.info(
                            "bm25_index_built_from_raw",
                            filing_count=len(filing_dirs),
                            chunk_count=len(all_chunks),
                        )
                    else:
                        logger.warning("no_chunks_for_bm25", raw_dir=str(raw_dir))
                else:
                    logger.warning("no_filing_dirs_found", raw_dir=str(raw_dir))
            else:
                logger.warning("raw_dir_not_found", raw_dir=str(raw_dir))

            # Build hybrid retriever with both indexes
            hybrid_retriever = HybridRetriever(
                chroma_store=chroma_store,
                bm25_index=bm25_index,
            )
            reranker = CrossEncoderReranker()
            rag_generator = RAGGenerator()

            app.state.compiled_graph = compile_rag_graph(
                hybrid_retriever=hybrid_retriever,
                reranker=reranker,
                rag_generator=rag_generator,
            )
            app.state.chroma_store = chroma_store  # Expose for /available-filings
            logger.info("rag_pipeline_initialized")

        except Exception as e:
            logger.error("pipeline_init_failed", error=str(e))
            app.state.compiled_graph = None
    else:
        app.state.compiled_graph = None
        logger.info("pipeline_init_skipped", reason="FINRAG_INIT_PIPELINE != true")

    logger.info(
        "finrag_api_ready",
        pipeline_active=app.state.compiled_graph is not None,
        max_sessions=max_sessions,
        prompt_version=prompt_version,
    )

    yield

    logger.info("finrag_api_shutdown")


# --------------------------------------------------------------------------- #
# Application Factory
# --------------------------------------------------------------------------- #


def create_app(
    api_key: str | None = None,
    max_requests_per_minute: int = 60,
    enable_auth: bool = True,
    enable_rate_limit: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        api_key: API key for auth. Reads FINRAG_API_KEY if None.
        max_requests_per_minute: Rate limit per client IP.
        enable_auth: Enable auth middleware.
        enable_rate_limit: Enable rate limiting.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="FinRAG",
        description=(
            "Citation-enforced financial research assistant over SEC filings. "
            "Every answer is grounded in specific paragraphs from specific filings."
        ),
        version="0.11.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)

    # --- Root Path Rewrite Middleware for QA/Compat ---
    @app.middleware("http")
    async def rewrite_paths_middleware(request: Request, call_next):
        path = request.url.path
        # Avoid double-prefixing if already starts with /api/v1
        if not path.startswith("/api/v1"):
            if (
                path == "/query"
                or path.startswith("/query/")
                or path == "/ingest"
                or path.startswith("/ingest/")
                or path == "/analytics"
                or path.startswith("/analytics/")
                or path == "/available-filings"
            ):
                request.scope["path"] = f"/api/v1{path}"
        return await call_next(request)

    # --- Middleware Stack ---
    # Applied in reverse: last add_middleware is outermost.
    if enable_rate_limit:
        app.add_middleware(
            RateLimitMiddleware,
            max_requests=max_requests_per_minute,
            window_seconds=60,
        )

    if enable_auth:
        app.add_middleware(AuthMiddleware, api_key=api_key)

    app.add_middleware(CoreRequestIDMiddleware)

    # --- CORS ---
    # Add CORS last so it is the outermost middleware.
    allowed_origins = os.environ.get("FINRAG_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routes ---
    app.include_router(api_router)
    app.include_router(ingest_router)
    app.include_router(mcp_router)

    # --- Health Check ---
    @app.get("/healthz", tags=["Health"])
    async def health_check() -> dict:
        """Health check for load balancers.

        Returns:
            Status dict with pipeline state.
        """
        chromadb_status = "error"
        if hasattr(app.state, "chroma_store") and app.state.chroma_store is not None:
            try:
                app.state.chroma_store._client.heartbeat()
                chromadb_status = "ok"
            except Exception:
                chromadb_status = "error"

        redis_status = "error"
        if hasattr(app.state, "redis_cache") and app.state.redis_cache is not None:
            if await app.state.redis_cache.health_check():
                redis_status = "ok"

        postgres_status = "error"
        try:
            from finrag.database.connection import engine
            from sqlalchemy import text
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            postgres_status = "ok"
        except Exception:
            postgres_status = "error"

        pipeline_active = hasattr(app.state, "compiled_graph") and app.state.compiled_graph is not None
        session_count = app.state.session_store.active_count if hasattr(app.state, "session_store") else 0

        overall_status = "healthy"
        if chromadb_status == "error" or redis_status == "error" or postgres_status == "error":
            overall_status = "degraded"

        return {
            "status": overall_status,
            "pipeline_active": pipeline_active,
            "active_sessions": session_count,
            "dependencies": {
                "chromadb": chromadb_status,
                "redis": redis_status,
                "postgres": postgres_status,
            },
            "version": "0.11.0",
        }

    logger.info(
        "fastapi_app_created",
        auth_enabled=enable_auth,
        rate_limit_enabled=enable_rate_limit,
        max_rpm=max_requests_per_minute,
    )

    return app
