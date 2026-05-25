import structlog
from finrag.core.job_store import JobStore
from finrag.ingestion.edgar_client import ingest_filing
from finrag.config import get_settings
from finrag.vectorstore.chroma_store import ChromaStore
from finrag.ingestion.chunker import chunk_filing_directory
from finrag.api.dependencies import get_redis_cache_instance

logger = structlog.get_logger(__name__)

async def run_ingestion(
    job_id: str,
    ticker: str,
    filing_type: str,
    fiscal_period: str,
    job_store: JobStore
) -> None:
    """Async background task that downloads filings, chunks them, embeds them, and invalidates cache."""
    try:
        await job_store.update_job(
            job_id=job_id,
            status="running",
            progress=10,
            message="Downloading filings from SEC EDGAR..."
        )

        settings = get_settings()
        
        # Download filing
        saved_paths = await ingest_filing(
            ticker=ticker,
            filing_type=filing_type,
            settings=settings,
            count=1
        )

        if not saved_paths:
            raise ValueError(f"No filings found or downloaded for ticker {ticker} with type {filing_type}")

        await job_store.update_job(
            job_id=job_id,
            status="running",
            progress=50,
            message=f"Downloaded {len(saved_paths)} filings. Chunking and embedding..."
        )

        # Chunk and embed filings
        store = ChromaStore()
        total_chunks = 0
        for path in saved_paths:
            chunks = chunk_filing_directory(path)
            added = store.add_chunks(chunks)
            total_chunks += added

        await job_store.update_job(
            job_id=job_id,
            status="completed",
            progress=100,
            message=f"Ingestion complete. Embedded {total_chunks} chunks."
        )

        # Invalidate Redis cache for this ticker
        cache = get_redis_cache_instance()
        if cache is not None:
            await cache.invalidate_ticker(ticker)

        logger.info("ingestion_job_success", job_id=job_id, ticker=ticker, chunks=total_chunks)

    except Exception as e:
        logger.error("ingestion_failed", job_id=job_id, ticker=ticker, error=str(e))
        await job_store.update_job(
            job_id=job_id,
            status="failed",
            progress=0,
            message="Ingestion failed",
            error=str(e)
        )
