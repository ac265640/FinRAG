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
import pathlib
from contextlib import asynccontextmanager

# Load .env from project root before anything else reads env vars.
# override=False means real environment variables always win over .env values.
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_path = pathlib.Path(__file__).parent.parent.parent.parent / ".env"
    if _env_path.exists():
        _load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on environment variables

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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
            
            # Use active prompt config to initialize RAGGenerator
            gen_config = getattr(app.state, "gen_config", None)
            if gen_config:
                rag_generator = RAGGenerator(
                    model_name=gen_config.model.name,
                    temperature=gen_config.model.temperature,
                    max_retries=gen_config.model.max_retries,
                )
            else:
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

    # --- Landing Page / Root Dashboard ---
    @app.get("/", tags=["Dashboard"], response_class=HTMLResponse)
    async def landing_page(request: Request) -> str:
        """Render a production-grade status dashboard and API usage documentation.

        This serves as the landing page inside Hugging Face Spaces.
        """
        # Determine status of databases
        chromadb_status = "error"
        total_chunks = 0
        available_filings = {}
        
        if hasattr(app.state, "chroma_store") and app.state.chroma_store is not None:
            try:
                app.state.chroma_store._client.heartbeat()
                chromadb_status = "ok"
                collection = app.state.chroma_store._collection
                if collection:
                    total_chunks = collection.count()
            except Exception:
                chromadb_status = "error"

        redis_status = "error"
        if hasattr(app.state, "redis_cache") and app.state.redis_cache is not None:
            try:
                if await app.state.redis_cache.health_check():
                    redis_status = "ok"
            except Exception:
                redis_status = "error"

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

        # Fetch filings lists using available filings logic directly
        if chromadb_status == "ok" and hasattr(app.state, "chroma_store"):
            try:
                result = app.state.chroma_store._collection.get(
                    include=["metadatas"],
                    limit=5000,
                )
                metadatas = result.get("metadatas", [])
                for meta in metadatas:
                    ticker = meta.get("ticker", "")
                    form_type = meta.get("form_type", "")
                    filing_date = meta.get("filing_date", "")
                    if not ticker or not form_type:
                        continue
                    if ticker not in available_filings:
                        available_filings[ticker] = {}
                    if form_type not in available_filings[ticker]:
                        available_filings[ticker][form_type] = set()
                    if filing_date:
                        available_filings[ticker][form_type].add(filing_date)
                
                # Convert to serializable format
                available_filings = {
                    ticker: {
                        form_type: sorted(list(dates), reverse=True)
                        for form_type, dates in form_types.items()
                    }
                    for ticker, form_types in available_filings.items()
                }
            except Exception:
                pass

        # Build list of filings HTML
        filings_rows_html = ""
        if available_filings:
            for ticker, form_types in available_filings.items():
                for form_type, dates in form_types.items():
                    dates_str = ", ".join(dates)
                    filings_rows_html += f"""
                    <li class="filing-row">
                        <span class="filing-ticker">{ticker}</span>
                        <div class="filing-details">
                            <span class="filing-type">{form_type}</span>
                            <span class="filing-date">Date: {dates_str}</span>
                        </div>
                    </li>
                    """
        else:
            filings_rows_html = """
            <li class="filing-row" style="justify-content: center; color: var(--text-muted); font-style: italic;">
                No SEC filings ingested yet. Use the POST /api/v1/ingest API to load files.
            </li>
            """

        pipeline_status_badge = '<span class="status-badge status-ok"><span class="pulse"></span>Active</span>' if pipeline_active else '<span class="status-badge status-warning">Inactive</span>'
        chroma_status_badge = '<span class="status-badge status-ok"><span class="pulse"></span>Online</span>' if chromadb_status == "ok" else '<span class="status-badge status-error">Offline</span>'
        redis_status_badge = '<span class="status-badge status-ok"><span class="pulse"></span>Online</span>' if redis_status == "ok" else '<span class="status-badge status-error">Offline</span>'
        postgres_status_badge = '<span class="status-badge status-ok"><span class="pulse"></span>Online</span>' if postgres_status == "ok" else '<span class="status-badge status-error">Offline</span>'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinRAG - Live Production Engine</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #080c14;
            --card-bg: rgba(15, 23, 42, 0.6);
            --card-border: rgba(255, 255, 255, 0.08);
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.15);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.15);
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --text-accent: #60a5fa;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.04) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.04) 0%, transparent 40%);
            background-attachment: fixed;
            padding: 2.5rem 1.5rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
            flex: 1;
        }}

        header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
        }}

        .logo-group {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .logo-icon {{
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.3rem;
            color: #ffffff;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.25);
        }}

        .logo-text h1 {{
            font-size: 1.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.025em;
        }}

        .logo-text p {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .header-links {{
            display: flex;
            gap: 0.75rem;
        }}

        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1.25rem;
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.85rem;
            text-decoration: none;
            transition: all 0.2s ease;
            cursor: pointer;
        }}

        .btn-primary {{
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .btn-primary:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
        }}

        .btn-outline {{
            background: rgba(255, 255, 255, 0.04);
            color: var(--text-main);
            border: 1px solid var(--card-border);
        }}

        .btn-outline:hover {{
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-1px);
        }}

        .grid {{
            display: grid;
            grid-template-columns: 7fr 5fr;
            gap: 2rem;
        }}

        @media (max-width: 900px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.75rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        }}

        .card-title {{
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            color: #ffffff;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.75rem;
        }}

        .card-title svg {{
            width: 18px;
            height: 18px;
            color: var(--text-accent);
        }}

        .system-status {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }}

        @media (max-width: 500px) {{
            .system-status {{
                grid-template-columns: 1fr;
            }}
        }}

        .status-item {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .status-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
        }}

        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.25rem 0.625rem;
            border-radius: 9999px;
            text-transform: uppercase;
        }}

        .status-ok {{
            background-color: var(--success-glow);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}

        .status-error {{
            background-color: rgba(239, 68, 68, 0.1);
            color: var(--danger);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }}

        .status-warning {{
            background-color: rgba(245, 158, 11, 0.1);
            color: var(--warning);
            border: 1px solid rgba(245, 158, 11, 0.2);
        }}

        .pulse {{
            width: 6px;
            height: 6px;
            background-color: currentColor;
            border-radius: 50%;
            display: inline-block;
            animation: pulse-animation 2s infinite;
        }}

        @keyframes pulse-animation {{
            0% {{
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            }}
            70% {{
                transform: scale(1);
                box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
            }}
            100% {{
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }}
        }}

        .filings-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}

        .filing-row {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: border-color 0.2s ease;
        }}

        .filing-row:hover {{
            border-color: rgba(255, 255, 255, 0.15);
        }}

        .filing-ticker {{
            font-weight: 700;
            font-size: 0.9rem;
            background: rgba(59, 130, 246, 0.1);
            color: var(--text-accent);
            padding: 0.125rem 0.5rem;
            border-radius: 6px;
            border: 1px solid rgba(59, 130, 246, 0.2);
        }}

        .filing-details {{
            display: flex;
            align-items: center;
            gap: 1.5rem;
            font-size: 0.85rem;
        }}

        .filing-type {{
            color: #ffffff;
            font-weight: 600;
        }}

        .filing-date {{
            color: var(--text-muted);
        }}

        .api-routes {{
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }}

        .route-item {{
            border: 1px solid var(--card-border);
            border-radius: 12px;
            overflow: hidden;
            background: rgba(0, 0, 0, 0.15);
        }}

        .route-header {{
            padding: 0.875rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--card-border);
        }}

        .method {{
            padding: 0.25rem 0.625rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .method-get {{
            background-color: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}

        .method-post {{
            background-color: rgba(59, 130, 246, 0.1);
            color: var(--text-accent);
            border: 1px solid rgba(59, 130, 246, 0.2);
        }}

        .route-path {{
            color: #ffffff;
        }}

        .route-desc {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-left: auto;
        }}

        .route-body {{
            padding: 1.25rem;
            border-top: 1px solid rgba(255, 255, 255, 0.02);
        }}

        .code-container {{
            position: relative;
            background: #05070e;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            overflow-x: auto;
            color: #cbd5e1;
            margin-top: 0.5rem;
        }}

        .code-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 0.375rem;
            display: block;
        }}

        .copy-badge {{
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 0.125rem 0.375rem;
            font-size: 0.65rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .copy-badge:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
        }}

        footer {{
            margin-top: 5rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--card-border);
            padding-top: 1.5rem;
        }}

        .auth-banner {{
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: flex-start;
            gap: 0.875rem;
        }}

        .auth-banner-icon {{
            color: var(--text-accent);
            flex-shrink: 0;
            margin-top: 0.125rem;
        }}

        .auth-banner-content h4 {{
            font-weight: 700;
            font-size: 0.95rem;
            color: #ffffff;
            margin-bottom: 0.25rem;
        }}

        .auth-banner-content p {{
            font-size: 0.82rem;
            color: var(--text-muted);
            line-height: 1.45;
        }}

        .summary-stats {{
            display: flex;
            gap: 2rem;
            margin-bottom: 1.5rem;
        }}

        .stat-card {{
            flex: 1;
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        }}

        .stat-value {{
            font-size: 1.75rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.25rem;
        }}

        .stat-desc {{
            font-size: 0.8rem;
            color: var(--text-muted);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-group">
                <div class="logo-icon">FR</div>
                <div class="logo-text">
                    <h1>FinRAG Engine</h1>
                    <p>Citation-Enforced SEC Research Assistant API</p>
                </div>
            </div>
            <div class="header-links">
                <a href="/docs" class="btn btn-primary">
                    Interactive Docs
                </a>
            </div>
        </header>

        <div class="auth-banner">
            <div class="auth-banner-icon">
                <svg style="width:20px;height:20px;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
            </div>
            <div class="auth-banner-content">
                <h4>API Authentication Protocol Active</h4>
                <p>This API engine enforces security tokens in production. For all API endpoints (excluding <code>/healthz</code> and <code>/docs</code>), you must present your configured <code>FINRAG_API_KEY</code> token inside the <code>X-API-Key</code> or <code>Authorization: Bearer &lt;token&gt;</code> headers.</p>
            </div>
        </div>

        <div class="grid">
            <div class="left-col">
                <div class="card">
                    <h2 class="card-title">
                        Core API Endpoints
                    </h2>
                    
                    <div class="api-routes">
                        <div class="route-item">
                            <div class="route-header">
                                <span class="method method-get">GET</span>
                                <span class="route-path">/api/v1/available-filings</span>
                                <span class="route-desc">List ingested SEC filings</span>
                            </div>
                            <div class="route-body">
                                <span class="code-label">Example Request</span>
                                <div class="code-container">
                                    <div class="copy-badge" onclick="navigator.clipboard.writeText(this.nextSibling.textContent.trim()); this.textContent='Copied!'; setTimeout(() => this.textContent='Copy', 1500)">Copy</div><pre>curl -X GET https://acxxxy-finrag.hf.space/api/v1/available-filings \\
  -H "X-API-Key: YOUR_API_KEY"</pre>
                                </div>
                            </div>
                        </div>

                        <div class="route-item">
                            <div class="route-header">
                                <span class="method method-post">POST</span>
                                <span class="route-path">/api/v1/query</span>
                                <span class="route-desc">Trigger a citation-enforced RAG query</span>
                            </div>
                            <div class="route-body">
                                <span class="code-label">Example Request</span>
                                <div class="code-container">
                                    <div class="copy-badge" onclick="navigator.clipboard.writeText(this.nextSibling.textContent.trim()); this.textContent='Copied!'; setTimeout(() => this.textContent='Copy', 1500)">Copy</div><pre>curl -X POST https://acxxxy-finrag.hf.space/api/v1/query \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -d '{{
    "query": "What is Apples total net sales for fiscal year 2025?",
    "filters": {{
      "ticker": "AAPL",
      "filing_type": "10-K"
    }}
  }}'</pre>
                                </div>
                            </div>
                        </div>

                        <div class="route-item">
                            <div class="route-header">
                                <span class="method method-post">POST</span>
                                <span class="route-path">/api/v1/ingest</span>
                                <span class="route-desc">Asynchronously ingest SEC filings</span>
                            </div>
                            <div class="route-body">
                                <span class="code-label">Example Request</span>
                                <div class="code-container">
                                    <div class="copy-badge" onclick="navigator.clipboard.writeText(this.nextSibling.textContent.trim()); this.textContent='Copied!'; setTimeout(() => this.textContent='Copy', 1500)">Copy</div><pre>curl -X POST https://acxxxy-finrag.hf.space/api/v1/ingest \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -d '{{
    "ticker": "AAPL",
    "filing_type": "10-K"
  }}'</pre>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="right-col">
                <div class="card">
                    <h2 class="card-title">
                        System Status
                    </h2>
                    
                    <div class="summary-stats">
                        <div class="stat-card">
                            <div class="stat-value">{total_chunks}</div>
                            <div class="stat-desc">Ingested Chunks</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{len(available_filings)}</div>
                            <div class="stat-desc">Active Tickers</div>
                        </div>
                    </div>

                    <div class="system-status">
                        <div class="status-item">
                            <span class="status-label">Pipeline Engine</span>
                            {pipeline_status_badge}
                        </div>
                        <div class="status-item">
                            <span class="status-label">ChromaDB Store</span>
                            {chroma_status_badge}
                        </div>
                        <div class="status-item">
                            <span class="status-label">Upstash Cache</span>
                            {redis_status_badge}
                        </div>
                        <div class="status-item">
                            <span class="status-label">Neon Postgres</span>
                            {postgres_status_badge}
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2 class="card-title">
                        Ingested Tickers & Filings
                    </h2>
                    <ul class="filings-list">
                        {filings_rows_html}
                    </ul>
                </div>
            </div>
        </div>

        <footer>
            <p>&copy; 2026 FinRAG Production Engine. Structured, Citation-Grounded Financial Intelligence.</p>
        </footer>
    </div>
</body>
</html>"""
        return html_content

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
