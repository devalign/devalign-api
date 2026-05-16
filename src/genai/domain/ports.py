"""GenAI module domain ports (interfaces)."""

from abc import ABC, abstractmethod
from uuid import UUID

from src.genai.domain.entities import Roadmap


class LLMService(ABC):
    """Port for LLM text generation — decoupled from provider."""

    @abstractmethod
    async def generate(self, prompt: str, context: list[str]) -> str:
        """Generate text from a prompt with optional context documents."""
        ...


class VectorStorePort(ABC):
    """Port for vector database operations."""

    @abstractmethod
    async def similarity_search(
        self, query: str, k: int = 5, filter_metadata: dict[str, str] | None = None
    ) -> list[str]:
        """Return the k most similar documents to the query."""
        ...

    @abstractmethod
    async def add_documents(self, texts: list[str], metadatas: list[dict[str, str]]) -> None:
        """Add documents with metadata to the vector store."""
        ...


class RoadmapRepository(ABC):
    """Port for roadmap persistence."""

    @abstractmethod
    async def save(self, roadmap: Roadmap) -> Roadmap:
        """Persist a generated roadmap."""
        ...

    @abstractmethod
    async def get_by_user_id(self, user_id: UUID) -> list[Roadmap]:
        """Retrieve all roadmaps for a user."""
        ...

    @abstractmethod
    async def get_latest_by_user_id(self, user_id: UUID) -> Roadmap | None:
        """Retrieve the most recent roadmap for a user."""
        ...
