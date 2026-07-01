"""Tests for middleware and global exception handlers introduced in this PR.

Covers:
- DevalignException → 4xx/5xx handler registered in create_app()
- RequestLoggingMiddleware: successful request flow (X-Request-ID header)
- RequestLoggingMiddleware: unhandled exception is re-raised (not swallowed)
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.shared.exceptions import (
    AuthorizationError,
    DevalignException,
    NotFoundError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_with_routes() -> FastAPI:
    """Create a minimal FastAPI app that exercises our handler and middleware."""
    from src.main import create_app

    application = create_app()

    # Add test-only routes so we can trigger specific behaviours without
    # touching the real authentication / DB layers.
    from fastapi import APIRouter

    test_router = APIRouter(prefix="/test-only")

    @test_router.get("/raise-not-found")
    async def _raise_not_found() -> None:
        raise NotFoundError("Test resource not found")

    @test_router.get("/raise-forbidden")
    async def _raise_forbidden() -> None:
        raise AuthorizationError()

    @test_router.get("/raise-base")
    async def _raise_base() -> None:
        raise DevalignException("generic failure")

    @test_router.get("/raise-runtime")
    async def _raise_runtime() -> None:
        raise RuntimeError("boom")

    @test_router.get("/ok")
    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    application.include_router(test_router)
    return application


# ---------------------------------------------------------------------------
# Exception handler tests (new code: main.py lines 101-106)
# ---------------------------------------------------------------------------


@pytest.fixture
def test_client() -> TestClient:
    """Sync test client — enough for exception handler smoke-tests."""
    return TestClient(_make_app_with_routes(), raise_server_exceptions=False)


def test_not_found_error_returns_404(test_client: TestClient) -> None:
    """DevalignException subclass with 404 status → handler returns 404."""
    resp = test_client.get("/test-only/raise-not-found")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Test resource not found"


def test_authorization_error_returns_403(test_client: TestClient) -> None:
    """AuthorizationError (403 subclass) is handled by the global handler."""
    resp = test_client.get("/test-only/raise-forbidden")
    assert resp.status_code == 403
    assert "detail" in resp.json()


def test_base_devalign_exception_returns_500(test_client: TestClient) -> None:
    """Base DevalignException defaults to 500."""
    resp = test_client.get("/test-only/raise-base")
    assert resp.status_code == 500
    assert resp.json()["detail"] == "generic failure"


def test_unhandled_runtime_error_returns_500(test_client: TestClient) -> None:
    """A plain RuntimeError bypasses the DevalignException handler → 500."""
    resp = test_client.get("/test-only/raise-runtime")
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Middleware tests (new code: middleware.py lines 33-40)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_injects_request_id_header() -> None:
    """Successful requests must carry X-Request-ID in the response."""
    app = _make_app_with_routes()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/test-only/ok")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    # Must be a valid UUID (36 chars with dashes)
    assert len(resp.headers["x-request-id"]) == 36


@pytest.mark.asyncio
async def test_middleware_reraises_unhandled_exception() -> None:
    """Middleware must re-raise (not swallow) unexpected exceptions.

    When the middleware re-raises, ASGITransport propagates the exception to
    the test client. We confirm that the RuntimeError bubbles up — this is the
    expected behaviour after the refactor (log + raise, not return 500 JSON).
    """
    app = _make_app_with_routes()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        with pytest.raises(RuntimeError, match="boom"):
            await ac.get("/test-only/raise-runtime")
