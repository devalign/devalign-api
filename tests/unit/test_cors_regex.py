"""Tests for CORS configuration including regex pattern matching."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_cors_allowed_exact_origin(client: AsyncClient) -> None:
    """Explicit origins in CORS_ORIGINS should be allowed."""
    headers = {"Origin": "http://localhost:3000"}
    response = await client.get("/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


@pytest.mark.asyncio
async def test_cors_allowed_regex_preview_origin(client: AsyncClient) -> None:
    """Dynamic Vercel preview URLs matching CORS_ORIGIN_REGEX should be allowed."""
    origin = "https://devalign-web-git-feat-market-topology-compact-view-jackaranaram.vercel.app"
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "GET",
    }
    response = await client.options("/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
    assert response.headers.get("access-control-allow-credentials") == "true"


@pytest.mark.asyncio
async def test_cors_disallowed_origin(client: AsyncClient) -> None:
    """Origins not in CORS_ORIGINS nor matching CORS_ORIGIN_REGEX should not get CORS headers."""
    headers = {"Origin": "https://unauthorized-domain.com"}
    response = await client.get("/health", headers=headers)
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
