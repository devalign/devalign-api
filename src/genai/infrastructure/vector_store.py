"""pgvector implementation of VectorStorePort using LangChain."""

from typing import Any

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

            if cfg.EMBEDDING_PROVIDER == "openai":
                from langchain_openai import OpenAIEmbeddings

                embeddings: Any = OpenAIEmbeddings(
                    openai_api_key=cfg.OPENAI_API_KEY,
                    model=cfg.EMBEDDING_MODEL,
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
