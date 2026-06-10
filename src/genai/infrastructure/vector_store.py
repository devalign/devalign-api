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
            cfg = settings
            embeddings: Any = None
            if cfg.EMBEDDING_PROVIDER == "openai":
                from langchain_openai import OpenAIEmbeddings

                from src.ml_engine.infrastructure.models import EMBEDDING_DIM

                embeddings = OpenAIEmbeddings(
                    openai_api_key=cfg.OPENAI_API_KEY,
                    model=cfg.EMBEDDING_MODEL,
                    dimensions=EMBEDDING_DIM,
                )
            elif cfg.EMBEDDING_PROVIDER == "voyage":
                from langchain_core.embeddings import Embeddings

                class VoyageAIEmbeddings(Embeddings):
                    def __init__(self, api_key: str, model_name: str):
                        self._api_key = api_key
                        self._model = model_name
                        self._api_url = "https://api.voyageai.com/v1/embeddings"

                    def embed_documents(self, texts: list[str]) -> list[list[float]]:
                        import httpx

                        headers = {
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        }
                        response = httpx.post(
                            self._api_url,
                            headers=headers,
                            json={"input": texts, "model": self._model},
                            timeout=60.0,
                        )
                        if response.status_code != 200:
                            raise RuntimeError(f"Voyage AI Inference API failed: {response.text}")
                        data = response.json()
                        return cast(
                            "list[list[float]]", [item["embedding"] for item in data["data"]]
                        )

                    def embed_query(self, text: str) -> list[float]:
                        import httpx

                        headers = {
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        }
                        response = httpx.post(
                            self._api_url,
                            headers=headers,
                            json={"input": [text], "model": self._model},
                            timeout=30.0,
                        )
                        if response.status_code != 200:
                            raise RuntimeError(f"Voyage AI Inference API failed: {response.text}")
                        data = response.json()
                        return cast("list[float]", data["data"][0]["embedding"])

                embeddings = VoyageAIEmbeddings(
                    api_key=cfg.VOYAGE_API_KEY,
                    model_name=cfg.EMBEDDING_MODEL,
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {cfg.EMBEDDING_PROVIDER}")

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
