"""ML Engine domain ports (interfaces)."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from src.ml_engine.domain.entities import Skill, SkillRelationType, TechCluster, UserProfile


class LLMService(ABC):
    """Port for interacting with LLM APIs."""

    @abstractmethod
    async def generate(self, prompt: str, context: list[Any] | None = None) -> str:
        """Generate text from a prompt with optional context."""
        ...


class EmbeddingService(ABC):
    """Port for generating embeddings from text."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of text strings."""
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

    @abstractmethod
    async def delete_by_user_id(self, user_id: UUID) -> None:
        """Delete the user profile and all associated data."""
        ...


class SkillRepository(ABC):
    """Port for skill persistence and retrieval."""

    @abstractmethod
    async def get_all_skills(self) -> list[Skill]:
        """Retrieve all canonical skills with aliases and outgoing relations."""
        ...

    @abstractmethod
    async def get_skill_graph(self) -> dict[UUID, Skill]:
        """Load the full skill graph as a {skill_id: Skill} map for O(1) lookups."""
        ...

    @abstractmethod
    async def save_skills(self, skills: list[Skill]) -> list[Skill]:
        """Save new skills to the database and return them with populated IDs."""
        ...

    @abstractmethod
    async def add_relations(
        self, relations: list[tuple[UUID, UUID, SkillRelationType]]
    ) -> None:
        """Persist skill-to-skill knowledge graph edges (idempotent)."""
        ...


class MLJobOfferRepository(ABC):
    """Port for accessing job offers from the ML Engine context."""

    @abstractmethod
    async def get_unnormalized_offers(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve job offers that have not yet been normalized.
        Returns dictionaries or raw data rather than domain entities, since the ML
        engine only needs id and raw_skills.
        """
        ...

    @abstractmethod
    async def save_offer_skills(self, offer_skills: list[dict[str, Any]]) -> None:
        """Bulk save offer_skills relations."""
        ...

    @abstractmethod
    async def mark_as_normalized(self, job_offer_ids: list[UUID]) -> None:
        """Mark job offers as normalized so they aren't processed again."""
        ...
