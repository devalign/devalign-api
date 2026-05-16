"""Tests for health check and basic application startup."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """Health check endpoint must return 200 with expected payload."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "env" in data


@pytest.mark.asyncio
async def test_openapi_docs_available(client: AsyncClient, api_prefix: str) -> None:
    """OpenAPI docs must be accessible."""
    response = await client.get(f"{api_prefix}/docs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_openapi_json_available(client: AsyncClient, api_prefix: str) -> None:
    """OpenAPI JSON schema must be accessible."""
    response = await client.get(f"{api_prefix}/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Devalign API"
