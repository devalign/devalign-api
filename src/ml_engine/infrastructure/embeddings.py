"""Embedding services using HTTP APIs (Voyage AI and OpenAI)."""

from typing import Any, cast

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.ml_engine.domain.ports import EmbeddingService
from src.shared.exceptions import ExternalServiceError

logger = structlog.get_logger(__name__)


class VoyageEmbeddingService(EmbeddingService):
    """Embedding service using Voyage AI HTTP API."""

    def __init__(self, api_key: str, model: str = "voyage-4-lite") -> None:
        self._api_key = api_key
        self._model = model
        self._api_url = "https://api.voyageai.com/v1/embeddings"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_text(self, text: str) -> list[float]:
        if not self._api_key:
            raise ExternalServiceError("Voyage AI API key is not configured")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    self._api_url,
                    headers=headers,
                    json={"input": [text], "model": self._model},
                )
                if response.status_code != 200:
                    logger.error(
                        "Voyage AI API failed",
                        status=response.status_code,
                        body=response.text,
                    )
                    raise ExternalServiceError("Voyage AI embedding service failed")
                data = response.json()
                return cast("list[float]", data["data"][0]["embedding"])
            except Exception as exc:
                if isinstance(exc, ExternalServiceError):
                    raise exc
                logger.error("Voyage AI embedding failed", error=str(exc))
                raise ExternalServiceError("Voyage AI embedding service unavailable") from exc

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._api_key:
            raise ExternalServiceError("Voyage AI API key is not configured")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    self._api_url,
                    headers=headers,
                    json={"input": texts, "model": self._model},
                )
                if response.status_code != 200:
                    logger.error(
                        "Voyage AI API batch failed",
                        status=response.status_code,
                        body=response.text,
                    )
                    raise ExternalServiceError("Voyage AI embedding service failed")
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            except Exception as exc:
                if isinstance(exc, ExternalServiceError):
                    raise exc
                logger.error("Voyage AI batch embedding failed", error=str(exc))
                raise ExternalServiceError("Voyage AI embedding service unavailable") from exc


class OpenAIEmbeddingService(EmbeddingService):
    """Embedding service using OpenAI Embeddings HTTP API."""

    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", dimensions: int = 1024
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._api_url = "https://api.openai.com/v1/embeddings"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_text(self, text: str) -> list[float]:
        if not self._api_key:
            raise ExternalServiceError("OpenAI API key is not configured")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        json_payload: dict[str, Any] = {"input": [text], "model": self._model}
        if "text-embedding-3" in self._model:
            json_payload["dimensions"] = self._dimensions

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    self._api_url,
                    headers=headers,
                    json=json_payload,
                )
                if response.status_code != 200:
                    logger.error(
                        "OpenAI API failed",
                        status=response.status_code,
                        body=response.text,
                    )
                    raise ExternalServiceError("OpenAI embedding service failed")
                data = response.json()
                return cast("list[float]", data["data"][0]["embedding"])
            except Exception as exc:
                if isinstance(exc, ExternalServiceError):
                    raise exc
                logger.error("OpenAI embedding failed", error=str(exc))
                raise ExternalServiceError("OpenAI embedding service unavailable") from exc

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._api_key:
            raise ExternalServiceError("OpenAI API key is not configured")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        json_payload: dict[str, Any] = {"input": texts, "model": self._model}
        if "text-embedding-3" in self._model:
            json_payload["dimensions"] = self._dimensions

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    self._api_url,
                    headers=headers,
                    json=json_payload,
                )
                if response.status_code != 200:
                    logger.error(
                        "OpenAI API batch failed",
                        status=response.status_code,
                        body=response.text,
                    )
                    raise ExternalServiceError("OpenAI embedding service failed")
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            except Exception as exc:
                if isinstance(exc, ExternalServiceError):
                    raise exc
                logger.error("OpenAI batch embedding failed", error=str(exc))
                raise ExternalServiceError("OpenAI embedding service unavailable") from exc


def get_embedding_service() -> EmbeddingService:
    """Dependency provider factory for EmbeddingService."""
    if settings.EMBEDDING_PROVIDER == "openai":
        from src.ml_engine.infrastructure.models import EMBEDDING_DIM

        return OpenAIEmbeddingService(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIM,
        )
    elif settings.EMBEDDING_PROVIDER == "voyage":
        return VoyageEmbeddingService(
            api_key=settings.VOYAGE_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )
    raise ValueError(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}")
