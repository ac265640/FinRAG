from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from finrag.core.job_store import JobStore
from finrag.core.ingestion_worker import run_ingestion

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])

class IngestRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    filing_type: str = Field(..., description="Filing type (e.g. 10-K, 10-Q, 8-K)")
    fiscal_period: str = Field(..., description="Fiscal period (e.g. FY2025)")

class IngestResponse(BaseModel):
    job_id: str
    status: str
    message: str
    status_url: str

# Standard SEC filings tickers allowed for sandbox/demonstration
ALLOWED_TICKERS = {"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"}

@router.post("/ingest", response_model=IngestResponse)
async def post_ingest(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """Queue an asynchronous ingestion job for SEC filings in the background."""
    ticker_upper = body.ticker.upper().strip()
    if ticker_upper not in ALLOWED_TICKERS and not (ticker_upper.isalpha() and 1 <= len(ticker_upper) <= 5):
        raise HTTPException(
            status_code=400,
            detail=f"Ticker '{body.ticker}' is not in the allowed list."
        )
        
    filing_type_upper = body.filing_type.upper().strip()
    if filing_type_upper not in {"10-K", "10-Q", "8-K"}:
        raise HTTPException(
            status_code=400,
            detail="Filing type must be one of 10-K, 10-Q, or 8-K"
        )

    job_store = JobStore()
    job_id = await job_store.create_job(ticker=ticker_upper, filing_type=filing_type_upper)
    
    background_tasks.add_task(
        run_ingestion,
        job_id=job_id,
        ticker=ticker_upper,
        filing_type=filing_type_upper,
        fiscal_period=body.fiscal_period,
        job_store=job_store
    )
    
    return IngestResponse(
        job_id=job_id,
        status="pending",
        message="Ingestion job queued",
        status_url=f"/api/v1/ingest/{job_id}/status"
    )

@router.get("/ingest/{job_id}/status")
async def get_ingest_status(job_id: str) -> dict:
    """Retrieve current status of a queued or active ingestion job."""
    job_store = JobStore()
    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found")
    return job
