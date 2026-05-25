"""Middleware stack for the FinRAG API layer.

Provides layered middleware for request tracing, authentication,
rate limiting, and structured logging. Each middleware is a
Starlette BaseHTTPMiddleware subclass.

Middleware execution order (outermost to innermost):
    LoggingMiddleware, RequestIDMiddleware, AuthMiddleware, RateLimitMiddleware

Why this order:
- Logging wraps everything: captures total latency including auth/rate checks.
- RequestID runs early: downstream middleware can read request_id from state.
- Auth before rate limiting: unauthenticated requests rejected before
  consuming rate limit budget.
- Rate limiting innermost: only valid requests consume tokens.

Design decisions:
- In-memory rate limiter: sliding window counter per client IP.
  Zero deps, good for single-instance. For multi-instance, swap to Redis.
- Auth is bearer token: single shared secret via FINRAG_API_KEY env var.
- Skip paths: /healthz, /docs, /openapi.json bypass auth entirely.

Debt: DAY-11-001 -- Rate limiter is in-memory, resets on restart.
      Use Redis for multi-instance deployment (Day 15).
"""

import os
import time
import uuid
from collections import defaultdict

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

SKIP_AUTH_PATHS = {"/", "/healthz", "/docs", "/openapi.json", "/redoc"}


# --------------------------------------------------------------------------- #
# Request ID Middleware
# --------------------------------------------------------------------------- #


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request/response.

    Generates a UUID4 per request. Stored in request.state.request_id
    and returned as X-Request-ID response header. Uses client-provided
    X-Request-ID if present for end-to-end tracing.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with ID injection.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response with X-Request-ID header.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# --------------------------------------------------------------------------- #
# Auth Middleware
# --------------------------------------------------------------------------- #


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate Bearer token against FINRAG_API_KEY env var.

    Rejects requests without valid token with 401. Skips
    /healthz and /docs paths entirely.
    """

    def __init__(self, app, api_key: str | None = None) -> None:
        """Initialize with API key.

        Args:
            app: ASGI application.
            api_key: API key to validate. If None, reads FINRAG_API_KEY.
                If env var unset, auth is disabled (dev mode).
        """
        super().__init__(app)
        self.api_key = api_key or os.environ.get("FINRAG_API_KEY", "")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Validate auth token.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response or 401 JSONResponse if unauthorized.
        """
        if request.url.path in SKIP_AUTH_PATHS:
            return await call_next(request)

        if not self.api_key:
            request.state.api_key = ""
            return await call_next(request)

        token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            x_api_key = request.headers.get("X-API-Key", "")
            if x_api_key:
                token = x_api_key

        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid API key. Use: Bearer <token> in Authorization header or X-API-Key header"},
            )

        if token != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )

        request.state.api_key = token
        return await call_next(request)


# --------------------------------------------------------------------------- #
# Rate Limit Middleware
# --------------------------------------------------------------------------- #


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter per client IP.

    Tracks request timestamps per IP in a rolling window.
    Returns 429 Too Many Requests when limit exceeded.

    Attributes:
        max_requests: Maximum requests per window.
        window_seconds: Window duration in seconds.
        _requests: Dict mapping IP to list of timestamps.
    """

    def __init__(
        self,
        app,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        """Initialize rate limiter.

        Args:
            app: ASGI application.
            max_requests: Max requests per window (default 60).
            window_seconds: Window size in seconds (default 60).
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Check rate limit for client IP.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response or 429 JSONResponse if rate limited.
        """
        if request.url.path in SKIP_AUTH_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self.window_seconds

        self._requests[client_ip] = [ts for ts in self._requests[client_ip] if ts > cutoff]

        if len(self._requests[client_ip]) >= self.max_requests:
            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                count=len(self._requests[client_ip]),
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s."),
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


# --------------------------------------------------------------------------- #
# Logging Middleware
# --------------------------------------------------------------------------- #


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured JSON logging for every HTTP request.

    Logs method, path, status code, latency, and request ID.
    Uses structlog for consistent JSON output.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Log request and response details.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response (unmodified).
        """
        start = time.time()

        response = await call_next(request)

        latency_ms = round((time.time() - start) * 1000, 2)
        request_id = getattr(request.state, "request_id", "unknown")

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
            request_id=request_id,
        )

        return response
