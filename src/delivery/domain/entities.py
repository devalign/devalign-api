"""Delivery module domain entities."""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class User:
    """Represents an authenticated developer user."""

    id: UUID
    email: str
    full_name: str | None = None
    avatar_url: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not self.email or "@" not in self.email:
            raise ValueError(f"Invalid email: {self.email}")


@dataclass(frozen=True)
class CVDocument:
    """Represents an uploaded CV document."""

    id: UUID
    user_id: UUID
    storage_path: str  # Supabase Storage path
    original_filename: str
    content_type: str
    size_bytes: int
    status: str = "processing"
    uploaded_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_pdf(self) -> bool:
        return self.content_type == "application/pdf"

    @property
    def is_docx(self) -> bool:
        return self.content_type in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
