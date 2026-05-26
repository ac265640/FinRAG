"""Tests for Day 11: FastAPI layer, middleware, routes, MCP server.

Covers:
- App creation and health check
- RequestID middleware (UUID injection)
- Auth middleware (bearer token validation)
- Rate limit middleware (sliding window)
- Query endpoint (sync, stub pipeline)
- SSE streaming endpoint (event sequence)
- Session endpoints (GET, DELETE)
- Config/prompt version endpoint
- MCP tool listing and execution
"""

import pytest
from fastapi.testclient import TestClient

from finrag.api.app import create_app
from finrag.api.mcp_server import MCPToolCallRequest
from finrag.api.routes import QueryRequest, QueryResponse, SessionResponse

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def app_no_auth():
    """Create app with auth and rate limiting disabled."""
    app = create_app(enable_auth=False, enable_rate_limit=False)
    return app


@pytest.fixture
def client_no_auth(app_no_auth):
    """TestClient with auth disabled."""
    with TestClient(app_no_auth) as c:
        yield c


@pytest.fixture
def app_with_auth():
    """Create app with auth enabled using test key."""
    app = create_app(api_key="test-secret-key", enable_rate_limit=False)
    return app


@pytest.fixture
def client_with_auth(app_with_auth):
    """TestClient with auth enabled."""
    with TestClient(app_with_auth) as c:
        yield c


@pytest.fixture
def app_with_rate_limit():
    """Create app with rate limiting (3 req/60s for fast testing)."""
    app = create_app(
        enable_auth=False,
        enable_rate_limit=True,
        max_requests_per_minute=3,
    )
    return app


@pytest.fixture
def client_rate_limited(app_with_rate_limit):
    """TestClient with rate limiting."""
    with TestClient(app_with_rate_limit) as c:
        yield c


# --------------------------------------------------------------------------- #
# App Creation Tests
# --------------------------------------------------------------------------- #


class TestAppCreation:
    """Test FastAPI app creation."""

    def test_create_app_returns_fastapi(self):
        app = create_app(enable_auth=False, enable_rate_limit=False)
        assert app is not None
        assert app.title == "FinRAG"
        assert app.version == "0.11.0"

    def test_create_app_with_auth(self):
        app = create_app(api_key="test-key")
        assert app is not None


# --------------------------------------------------------------------------- #
# Health Check Tests
# --------------------------------------------------------------------------- #


class TestHealthCheck:
    """Test /healthz endpoint."""

    def test_health_returns_200(self, client_no_auth):
        resp = client_no_auth.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["version"] == "0.11.0"
        assert "pipeline_active" in data
        assert "active_sessions" in data

    def test_health_bypasses_auth(self, client_with_auth):
        # No auth header, but healthz should still work
        resp = client_with_auth.get("/healthz")
        assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# RequestID Middleware Tests
# --------------------------------------------------------------------------- #


class TestRequestIDMiddleware:
    """Test X-Request-ID injection."""

    def test_response_has_request_id(self, client_no_auth):
        resp = client_no_auth.get("/healthz")
        assert "X-Request-ID" in resp.headers
        # Should be a valid UUID format
        request_id = resp.headers["X-Request-ID"]
        assert len(request_id) == 36  # UUID4 length

    def test_client_provided_id_preserved(self, client_no_auth):
        resp = client_no_auth.get(
            "/healthz",
            headers={"X-Request-ID": "my-custom-id-123"},
        )
        assert resp.headers["X-Request-ID"] == "my-custom-id-123"


# --------------------------------------------------------------------------- #
# Auth Middleware Tests
# --------------------------------------------------------------------------- #


class TestAuthMiddleware:
    """Test bearer token authentication."""

    def test_no_auth_returns_401(self, client_with_auth):
        resp = client_with_auth.get("/api/v1/config/prompts")
        assert resp.status_code == 401
        assert "Authorization" in resp.json()["detail"]

    def test_invalid_token_returns_401(self, client_with_auth):
        resp = client_with_auth.get(
            "/api/v1/config/prompts",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_valid_token_passes(self, client_with_auth):
        resp = client_with_auth.get(
            "/api/v1/config/prompts",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 200

    def test_missing_bearer_prefix_returns_401(self, client_with_auth):
        resp = client_with_auth.get(
            "/api/v1/config/prompts",
            headers={"Authorization": "test-secret-key"},
        )
        assert resp.status_code == 401


# --------------------------------------------------------------------------- #
# Rate Limit Middleware Tests
# --------------------------------------------------------------------------- #


class TestRateLimitMiddleware:
    """Test sliding window rate limiting."""

    def test_under_limit_passes(self, client_rate_limited):
        resp = client_rate_limited.get("/api/v1/config/prompts")
        assert resp.status_code == 200

    def test_over_limit_returns_429(self, client_rate_limited):
        # 3 requests allowed, 4th should be rejected
        for _ in range(3):
            client_rate_limited.get("/api/v1/config/prompts")

        resp = client_rate_limited.get("/api/v1/config/prompts")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_health_bypasses_rate_limit(self, client_rate_limited):
        # Exhaust limit
        for _ in range(5):
            client_rate_limited.get("/api/v1/config/prompts")

        # Health should still work
        resp = client_rate_limited.get("/healthz")
        assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Query Endpoint Tests
# --------------------------------------------------------------------------- #


class TestQueryEndpoint:
    """Test POST /api/v1/query."""

    def test_query_returns_response(self, client_no_auth):
        resp = client_no_auth.post(
            "/api/v1/query",
            json={"query": "What was AAPL revenue in FY2024?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "session_id" in data
        assert "citations" in data
        assert "route" in data
        assert "metadata" in data

    def test_query_with_session_id(self, client_no_auth):
        resp = client_no_auth.post(
            "/api/v1/query",
            json={
                "query": "What was revenue?",
                "session_id": "test-session-1",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "test-session-1"

    def test_query_with_metadata_filter(self, client_no_auth):
        resp = client_no_auth.post(
            "/api/v1/query",
            json={
                "query": "What was revenue?",
                "metadata_filter": {"ticker": "AAPL"},
            },
        )
        assert resp.status_code == 200

    def test_query_too_short_returns_422(self, client_no_auth):
        resp = client_no_auth.post(
            "/api/v1/query",
            json={"query": "Hi"},
        )
        assert resp.status_code == 422

    def test_query_missing_body_returns_422(self, client_no_auth):
        resp = client_no_auth.post("/api/v1/query", json={})
        assert resp.status_code == 422

    def test_stub_response_when_no_pipeline(self, client_no_auth):
        resp = client_no_auth.post(
            "/api/v1/query",
            json={"query": "What was AAPL revenue?"},
        )
        data = resp.json()
        assert "stub" in data["answer"].lower() or "not initialized" in data["answer"].lower()


# --------------------------------------------------------------------------- #
# SSE Streaming Tests
# --------------------------------------------------------------------------- #


class TestStreamEndpoint:
    """Test POST /api/v1/query/stream SSE."""

    def test_stream_returns_events(self, client_no_auth):
        with client_no_auth.stream(
            "POST",
            "/api/v1/query/stream",
            json={"query": "What was AAPL revenue in FY2024?"},
        ) as resp:
            assert resp.status_code == 200
            content_type = resp.headers.get("content-type", "")
            assert "text/event-stream" in content_type

            events = []
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())

            # Should contain expected event types
            assert "retrieval_start" in events
            assert "chunks_found" in events
            assert "done" in events

    def test_stream_with_session(self, client_no_auth):
        with client_no_auth.stream(
            "POST",
            "/api/v1/query/stream",
            json={
                "query": "Revenue question?",
                "session_id": "stream-test-1",
            },
        ) as resp:
            assert resp.status_code == 200
            # Consume the stream
            for _ in resp.iter_lines():
                pass


# --------------------------------------------------------------------------- #
# Session Endpoint Tests
# --------------------------------------------------------------------------- #


class TestSessionEndpoints:
    """Test session management endpoints."""

    def test_get_session_not_found(self, client_no_auth):
        resp = client_no_auth.get("/api/v1/sessions/nonexistent")
        assert resp.status_code == 404

    def test_get_session_after_query(self, client_no_auth):
        # Create session via query
        client_no_auth.post(
            "/api/v1/query",
            json={
                "query": "What was AAPL revenue?",
                "session_id": "session-get-test",
            },
        )

        resp = client_no_auth.get("/api/v1/sessions/session-get-test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "session-get-test"
        assert data["turn_count"] == 1

    def test_delete_session(self, client_no_auth):
        # Create session
        client_no_auth.post(
            "/api/v1/query",
            json={
                "query": "What was revenue?",
                "session_id": "session-del-test",
            },
        )

        resp = client_no_auth.delete("/api/v1/sessions/session-del-test")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["detail"]

        # Verify gone
        resp = client_no_auth.get("/api/v1/sessions/session-del-test")
        assert resp.status_code == 404

    def test_delete_nonexistent_session(self, client_no_auth):
        resp = client_no_auth.delete("/api/v1/sessions/nope")
        assert resp.status_code == 404

    def test_multi_turn_session(self, client_no_auth):
        sid = "multi-turn-test"
        client_no_auth.post(
            "/api/v1/query",
            json={"query": "What was AAPL revenue?", "session_id": sid},
        )
        client_no_auth.post(
            "/api/v1/query",
            json={"query": "How about MSFT?", "session_id": sid},
        )

        resp = client_no_auth.get(f"/api/v1/sessions/{sid}")
        data = resp.json()
        assert data["turn_count"] == 2


# --------------------------------------------------------------------------- #
# Config Endpoint Tests
# --------------------------------------------------------------------------- #


class TestConfigEndpoint:
    """Test prompt config endpoint."""

    def test_get_prompt_config(self, client_no_auth):
        resp = client_no_auth.get("/api/v1/config/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt_versions" in data
        assert "status" in data


# --------------------------------------------------------------------------- #
# MCP Tool Tests
# --------------------------------------------------------------------------- #


class TestMCPTools:
    """Test MCP tool server."""

    def test_list_tools(self, client_no_auth):
        resp = client_no_auth.get("/mcp/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        tool_names = [t["name"] for t in data["tools"]]
        assert "query_financial_data" in tool_names
        assert "get_session_context" in tool_names
        assert "list_available_tickers" in tool_names

    def test_tool_has_schema(self, client_no_auth):
        resp = client_no_auth.get("/mcp/tools")
        tools = resp.json()["tools"]
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_call_query_tool(self, client_no_auth):
        resp = client_no_auth.post(
            "/mcp/call",
            json={
                "name": "query_financial_data",
                "arguments": {"query": "What was AAPL revenue?"},
                "call_id": "test-call-1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["call_id"] == "test-call-1"
        assert "answer" in data["result"]
        assert data["error"] is None

    def test_call_session_context_tool(self, client_no_auth):
        # First create a session via query
        client_no_auth.post(
            "/mcp/call",
            json={
                "name": "query_financial_data",
                "arguments": {
                    "query": "AAPL 10-K revenue?",
                    "session_id": "mcp-session-1",
                },
            },
        )

        resp = client_no_auth.post(
            "/mcp/call",
            json={
                "name": "get_session_context",
                "arguments": {"session_id": "mcp-session-1"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["session_id"] == "mcp-session-1"
        assert data["result"]["turn_count"] == 1

    def test_call_list_tickers_tool(self, client_no_auth):
        resp = client_no_auth.post(
            "/mcp/call",
            json={"name": "list_available_tickers", "arguments": {}},
        )
        assert resp.status_code == 200
        assert "tickers" in resp.json()["result"]

    def test_call_unknown_tool(self, client_no_auth):
        resp = client_no_auth.post(
            "/mcp/call",
            json={"name": "nonexistent_tool", "arguments": {}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is not None
        assert "Unknown tool" in data["error"]

    def test_call_session_context_not_found(self, client_no_auth):
        resp = client_no_auth.post(
            "/mcp/call",
            json={
                "name": "get_session_context",
                "arguments": {"session_id": "nope"},
            },
        )
        data = resp.json()
        assert "not found" in data["result"].get("error", "")


# --------------------------------------------------------------------------- #
# Pydantic Model Tests
# --------------------------------------------------------------------------- #


class TestRequestModels:
    """Test request/response model validation."""

    def test_query_request_valid(self):
        req = QueryRequest(query="What was revenue?")
        assert req.query == "What was revenue?"
        assert req.session_id is None
        assert req.metadata_filter is None

    def test_query_request_with_all_fields(self):
        req = QueryRequest(
            query="AAPL revenue?",
            session_id="s1",
            metadata_filter={"ticker": "AAPL"},
        )
        assert req.session_id == "s1"
        assert req.metadata_filter == {"ticker": "AAPL"}

    def test_query_response_defaults(self):
        resp = QueryResponse()
        assert resp.answer == ""
        assert resp.citations == []
        assert resp.session_id == ""

    def test_session_response(self):
        resp = SessionResponse(
            session_id="s1",
            turn_count=3,
            entities=["AAPL"],
            filings=["10-K"],
        )
        assert resp.session_id == "s1"
        assert resp.turn_count == 3

    def test_mcp_request_model(self):
        req = MCPToolCallRequest(
            name="query_financial_data",
            arguments={"query": "test"},
            call_id="c1",
        )
        assert req.name == "query_financial_data"
        assert req.call_id == "c1"
