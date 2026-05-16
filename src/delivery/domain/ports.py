"""Delivery module domain ports (interfaces)."""

from abc import ABC, abstractmethod
from uuid import UUID

from src.delivery.domain.entities import CVDocument, User


class UserRepository(ABC):
    """Port for user data persistence."""

    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> User | None:
        """Retrieve a user by their ID."""
        ...

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None:
        """Retrieve a user by email."""
        ...

    @abstractmethod
    async def upsert(self, user: User) -> User:
        """Create or update a user (used on auth callback)."""
        ...


class CVRepository(ABC):
    """Port for CV document persistence."""

    @abstractmethod
    async def save(self, cv: CVDocument) -> CVDocument:
        """Persist a CV document record."""
        ...

    @abstractmethod
    async def get_by_user_id(self, user_id: UUID) -> list[CVDocument]:
        """Get all CVs uploaded by a user."""
        ...

    @abstractmethod
    async def get_latest_by_user_id(self, user_id: UUID) -> CVDocument | None:
        """Get the most recent CV for a user."""
        ...


class StorageService(ABC):
    """Port for file storage operations."""

    @abstractmethod
    async def upload_cv(
        self,
        user_id: UUID,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        """Upload a CV file and return its storage path."""
        ...

    @abstractmethod
    async def get_signed_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """Generate a signed URL for temporary file access."""
        ...
