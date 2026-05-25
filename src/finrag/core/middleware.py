import uuid
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import time

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to inject an 8-character request ID into every request context for structured logging."""
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())[:8]  # short 8-char ID
        
        # Bind request_id to structlog context
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method
        )
        
        start = time.time()
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        
        # Log every request completion
        structlog.get_logger().info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=duration_ms
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
