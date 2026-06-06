"""Embedding service implementations.

Strategy:
- Development: sentence-transformers (local, free)
- Production: OpenAI text-embedding-3-small (higher quality)

Both implement EmbeddingService port — swappable via config.
"""

from typing import Any, cast

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.ml_engine.domain.ports import EmbeddingService
from src.shared.exceptions import ExternalServiceError, MLPipelineError

logger = structlog.get_logger(__name__)


class LocalEmbeddingService(EmbeddingService):
    """Embedding service using sentence-transformers (runs locally, no API cost)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # Lazy load to avoid startup delay

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading sentence-transformer model", model=self._model_name)
                self._model = SentenceTransformer(self._model_name)
            except ImportError as exc:
                raise MLPipelineError(
                    "sentence-transformers not installed. Run: uv add sentence-transformers"
                ) from exc
        return self._model

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        model = self._get_model()
        # sentence-transformers encode is synchronous — wrap for async compatibility
        import asyncio

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, convert_to_numpy=True).tolist(),
        )
        return cast("list[float]", embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        model = self._get_model()
        import asyncio

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_numpy=True).tolist(),
        )
        return cast("list[list[float]]", embeddings)


class OpenAIEmbeddingService(EmbeddingService):
    """Embedding service using OpenAI text-embedding-3-small API."""

    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", dimensions: int = 384
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions

    async def embed_text(self, text: str) -> list[float]:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self._api_key)
            kwargs: dict[str, Any] = {"input": text, "model": self._model}
            if "text-embedding-3" in self._model:
                kwargs["dimensions"] = self._dimensions
            response = await client.embeddings.create(**kwargs)
            return response.data[0].embedding
        except Exception as exc:
            logger.error("OpenAI embedding failed", error=str(exc))
            raise ExternalServiceError("OpenAI embedding service unavailable") from exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self._api_key)
            kwargs: dict[str, Any] = {"input": texts, "model": self._model}
            if "text-embedding-3" in self._model:
                kwargs["dimensions"] = self._dimensions
            response = await client.embeddings.create(**kwargs)
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.error("OpenAI batch embedding failed", error=str(exc))
            raise ExternalServiceError("OpenAI embedding service unavailable") from exc


class GroqEmbeddingService(EmbeddingService):
    """Embedding service using Groq's nomic-embed-text-v1.5 API."""

    def __init__(
        self, api_key: str, model: str = "nomic-embed-text-v1.5", dimensions: int = 384
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions

    def _truncate_and_normalize(self, vector: list[float]) -> list[float]:
        truncated = vector[: self._dimensions]
        import math

        norm = math.sqrt(sum(x**2 for x in truncated))
        if norm > 0:
            return [x / norm for x in truncated]
        return truncated

    async def embed_text(self, text: str) -> list[float]:
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self._api_key)
            response = await client.embeddings.create(input=text, model=self._model)
            return self._truncate_and_normalize(cast("list[float]", response.data[0].embedding))
        except Exception as exc:
            logger.error("Groq embedding failed", error=str(exc))
            raise ExternalServiceError("Groq embedding service unavailable") from exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self._api_key)
            response = await client.embeddings.create(input=texts, model=self._model)
            return cast(
                "list[list[float]]",
                [
                    self._truncate_and_normalize(cast("list[float]", item.embedding))
                    for item in response.data
                ],
            )
        except Exception as exc:
            logger.error("Groq batch embedding failed", error=str(exc))
            raise ExternalServiceError("Groq embedding service unavailable") from exc


class HuggingFaceAPIEmbeddingService(EmbeddingService):
    """Embedding service using Hugging Face's free Inference API."""

    def __init__(self, api_key: str, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._api_key = api_key
        self._model = model if "/" in model else f"sentence-transformers/{model}"
        self._api_url = f"https://api-inference.huggingface.co/models/{self._model}"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(ExternalServiceError),
        reraise=True,
    )
    async def embed_text(self, text: str) -> list[float]:
        import httpx

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    self._api_url,
                    headers=headers,
                    json={"inputs": [text]},
                )
                if response.status_code != 200:
                    logger.error(
                        "Hugging Face API failed",
                        status=response.status_code,
                        body=response.text,
                    )
                    raise ExternalServiceError("Hugging Face embedding service failed")
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list):
                        return cast("list[float]", data[0])
                    elif isinstance(data[0], float):
                        return cast("list[float]", data)
                raise ValueError("Unexpected Hugging Face API response format")
            except Exception as exc:
                logger.error("Hugging Face embedding failed", error=str(exc))
                raise ExternalServiceError("Hugging Face embedding service unavailable") from exc

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(ExternalServiceError),
        reraise=True,
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    self._api_url,
                    headers=headers,
                    json={"inputs": texts},
                )
                if response.status_code != 200:
                    logger.error(
                        "Hugging Face API failed",
                        status=response.status_code,
                        body=response.text,
                    )
                    raise ExternalServiceError("Hugging Face embedding service failed")
                data = response.json()
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    return cast("list[list[float]]", data)
                raise ValueError("Unexpected Hugging Face API response format")
            except Exception as exc:
                logger.error("Hugging Face batch embedding failed", error=str(exc))
                raise ExternalServiceError("Hugging Face embedding service unavailable") from exc


def get_embedding_service() -> EmbeddingService:
    """Factory: returns the configured embedding service."""
    if settings.EMBEDDING_PROVIDER == "openai":
        from src.ml_engine.infrastructure.models import EMBEDDING_DIM

        return OpenAIEmbeddingService(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIM,
        )
    elif settings.EMBEDDING_PROVIDER == "groq":
        from src.ml_engine.infrastructure.models import EMBEDDING_DIM

        return GroqEmbeddingService(
            api_key=settings.GROQ_API_KEY,
            model=settings.EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIM,
        )
    elif settings.EMBEDDING_PROVIDER == "huggingface":
        return HuggingFaceAPIEmbeddingService(
            api_key=settings.HF_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )
    # Default: local sentence-transformers
    return LocalEmbeddingService(model_name=settings.EMBEDDING_MODEL)
