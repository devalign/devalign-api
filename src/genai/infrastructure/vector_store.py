"""pgvector implementation of VectorStorePort using LangChain."""

from typing import Any, cast

import structlog
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

from src.config import settings
from src.genai.domain.ports import VectorStorePort
from src.shared.exceptions import ExternalServiceError

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "standards_knowledge_base"


class PGVectorStore(VectorStorePort):
    """
    Vector store backed by PostgreSQL pgvector extension.
    Stores SFIA 9 and IEEE SWECOM documents for RAG retrieval.
    """

    def __init__(self) -> None:
        self._store: PGVector | None = None

    def _get_store(self) -> PGVector:
        """Lazy initialize PGVector store."""
        if self._store is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            from src.config import settings as cfg

            embeddings: Any = None
            if cfg.EMBEDDING_PROVIDER == "openai":
                from langchain_openai import OpenAIEmbeddings

                from src.ml_engine.infrastructure.models import EMBEDDING_DIM

                embeddings = OpenAIEmbeddings(
                    openai_api_key=cfg.OPENAI_API_KEY,
                    model=cfg.EMBEDDING_MODEL,
                    dimensions=EMBEDDING_DIM,
                )
            elif cfg.EMBEDDING_PROVIDER == "groq":
                from langchain_core.embeddings import Embeddings

                from src.ml_engine.infrastructure.models import EMBEDDING_DIM

                class TruncatedGroqEmbeddings(Embeddings):
                    def __init__(self, api_key: str, model: str, dimensions: int):
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

                    def embed_documents(self, texts: list[str]) -> list[list[float]]:
                        from groq import Groq

                        client = Groq(api_key=self._api_key)
                        response = client.embeddings.create(input=texts, model=self._model)
                        return [
                            self._truncate_and_normalize(cast("list[float]", item.embedding))
                            for item in response.data
                        ]

                    def embed_query(self, text: str) -> list[float]:
                        from groq import Groq

                        client = Groq(api_key=self._api_key)
                        response = client.embeddings.create(input=text, model=self._model)
                        return self._truncate_and_normalize(
                            cast("list[float]", response.data[0].embedding)
                        )

                embeddings = TruncatedGroqEmbeddings(
                    api_key=cfg.GROQ_API_KEY,
                    model=cfg.EMBEDDING_MODEL,
                    dimensions=EMBEDDING_DIM,
                )
            elif cfg.EMBEDDING_PROVIDER == "huggingface":
                from langchain_core.embeddings import Embeddings

                class HuggingFaceAPIEmbeddings(Embeddings):
                    def __init__(self, api_key: str, model_name: str):
                        self._api_key = api_key
                        self._model = (
                            model_name
                            if "/" in model_name
                            else f"sentence-transformers/{model_name}"
                        )
                        self._api_url = f"https://api-inference.huggingface.co/models/{self._model}"

                    def embed_documents(self, texts: list[str]) -> list[list[float]]:
                        import httpx

                        headers = {}
                        if self._api_key:
                            headers["Authorization"] = f"Bearer {self._api_key}"
                        response = httpx.post(
                            self._api_url,
                            headers=headers,
                            json={"inputs": texts},
                            timeout=60.0,
                        )
                        if response.status_code != 200:
                            raise RuntimeError(
                                f"Hugging Face Inference API failed: {response.text}"
                            )
                        return cast("list[list[float]]", response.json())

                    def embed_query(self, text: str) -> list[float]:
                        import httpx

                        headers = {}
                        if self._api_key:
                            headers["Authorization"] = f"Bearer {self._api_key}"
                        response = httpx.post(
                            self._api_url,
                            headers=headers,
                            json={"inputs": [text]},
                            timeout=30.0,
                        )
                        if response.status_code != 200:
                            raise RuntimeError(
                                f"Hugging Face Inference API failed: {response.text}"
                            )
                        data = response.json()
                        if isinstance(data, list) and len(data) > 0:
                            if isinstance(data[0], list):
                                return cast("list[float]", data[0])
                            return cast("list[float]", data)
                        raise ValueError("Unexpected Hugging Face API response format")

                embeddings = HuggingFaceAPIEmbeddings(
                    api_key=cfg.HF_API_KEY,
                    model_name=cfg.EMBEDDING_MODEL,
                )
            else:
                embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL)

            self._store = PGVector(
                connection_string=settings.DATABASE_URL.replace("+asyncpg", ""),
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
        return self._store

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, str] | None = None,
    ) -> list[str]:
        """Return k most similar document texts from the vector store."""
        try:
            import asyncio

            store = self._get_store()
            loop = asyncio.get_event_loop()
            docs: list[Document] = await loop.run_in_executor(
                None,
                lambda: store.similarity_search(query=query, k=k, filter=filter_metadata),
            )
            return [doc.page_content for doc in docs]
        except Exception as exc:
            logger.error("Vector similarity search failed", query=query[:100], error=str(exc))
            raise ExternalServiceError("Vector store search failed") from exc

    async def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, str]],
    ) -> None:
        """Add documents to the vector store (used during seeding)."""
        try:
            import asyncio

            store = self._get_store()
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadatas, strict=True)
            ]
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: store.add_documents(documents))
            logger.info("Documents added to vector store", count=len(documents))
        except Exception as exc:
            logger.error("Failed to add documents to vector store", error=str(exc))
            raise ExternalServiceError("Vector store insertion failed") from exc
