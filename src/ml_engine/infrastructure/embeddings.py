"""Embedding service implementations.

Strategy:
- Development: sentence-transformers (local, free)
- Production: OpenAI text-embedding-3-small (higher quality)

Both implement EmbeddingService port — swappable via config.
"""

import structlog

from src.config import settings
from src.ml_engine.domain.ports import EmbeddingService
from src.shared.exceptions import ExternalServiceError, MLPipelineError

logger = structlog.get_logger(__name__)


class LocalEmbeddingService(EmbeddingService):
    """Embedding service using sentence-transformers (runs locally, no API cost)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # Lazy load to avoid startup delay

    def _get_model(self) -> object:
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
            lambda: model.encode(text, convert_to_numpy=True).tolist(),  # type: ignore[union-attr]
        )
        return embedding  # type: ignore[return-value]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        model = self._get_model()
        import asyncio

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_numpy=True).tolist(),  # type: ignore[union-attr]
        )
        return embeddings  # type: ignore[return-value]


class OpenAIEmbeddingService(EmbeddingService):
    """Embedding service using OpenAI text-embedding-3-small API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._api_key = api_key
        self._model = model

    async def embed_text(self, text: str) -> list[float]:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self._api_key)
            response = await client.embeddings.create(input=text, model=self._model)
            return response.data[0].embedding
        except Exception as exc:
            logger.error("OpenAI embedding failed", error=str(exc))
            raise ExternalServiceError("OpenAI embedding service unavailable") from exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self._api_key)
            response = await client.embeddings.create(input=texts, model=self._model)
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.error("OpenAI batch embedding failed", error=str(exc))
            raise ExternalServiceError("OpenAI embedding service unavailable") from exc


def get_embedding_service() -> EmbeddingService:
    """Factory: returns the configured embedding service."""
    if settings.EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddingService(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )
    # Default: local sentence-transformers
    return LocalEmbeddingService(model_name=settings.EMBEDDING_MODEL)
