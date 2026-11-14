"""pytest configuration and shared fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest.fixture
async def client() -> AsyncClient:
    """Async HTTP test client for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def api_prefix() -> str:
    """API version prefix."""
    return "/api/v1"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Automatically mark test items based on their directory location."""
    for item in items:
        fspath = str(item.fspath)
        if "tests/unit" in fspath or "tests\\unit" in fspath:
            item.add_marker(pytest.mark.unit)
        elif "tests/integration" in fspath or "tests\\integration" in fspath:
            item.add_marker(pytest.mark.integration)
