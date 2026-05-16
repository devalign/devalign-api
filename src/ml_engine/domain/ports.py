"""ML Engine domain ports (interfaces)."""

from abc import ABC, abstractmethod
from uuid import UUID

from src.ml_engine.domain.entities import TechCluster, UserProfile


class EmbeddingService(ABC):
    """Port for generating text embeddings."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


class CVParserService(ABC):
    """Port for parsing CV documents into plain text."""

    @abstractmethod
    async def extract_text(self, content: bytes, content_type: str) -> str:
        """Extract plain text from a PDF or DOCX document."""
        ...


class ClusterRepository(ABC):
    """Port for cluster data persistence."""

    @abstractmethod
    async def get_all_active(self) -> list[TechCluster]:
        """Retrieve all active tech clusters."""
        ...

    @abstractmethod
    async def get_by_id(self, cluster_id: UUID) -> TechCluster | None:
        """Retrieve a cluster by ID."""
        ...


class UserProfileRepository(ABC):
    """Port for user profile persistence."""

    @abstractmethod
    async def save(self, profile: UserProfile) -> UserProfile:
        """Persist a user profile."""
        ...

    @abstractmethod
    async def get_by_user_id(self, user_id: UUID) -> UserProfile | None:
        """Retrieve the latest profile for a user."""
        ...
