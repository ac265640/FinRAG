from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse
import os

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour"]
)

def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom exception handler for rate limiting to return structured JSON."""
    limit_val = os.getenv("RATE_LIMIT_PER_MINUTE", "10")
    retry_after = 30
    if hasattr(exc, "retry_after") and exc.retry_after is not None:
        retry_after = exc.retry_after
    return JSONResponse(
        status_code=429,
        content={
            "error": "RATE_LIMIT_EXCEEDED",
            "message": f"Too many requests. Limit: {limit_val}/minute.",
            "retry_after_seconds": retry_after
        },
        headers={"Retry-After": str(retry_after)}
    )
