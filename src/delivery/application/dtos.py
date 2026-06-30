"""Delivery module application DTOs."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr


class UserProfileDTO(BaseModel):
    """User profile data returned to API consumers."""

    id: UUID
    email: EmailStr
    full_name: str | None = None
    avatar_url: str | None = None


class CVUploadResultDTO(BaseModel):
    """Result of a CV upload operation."""

    cv_id: UUID
    user_id: UUID
    storage_path: str
    original_filename: str
    size_bytes: int
    status: str = "processing"
    download_url: str | None = None
    uploaded_at: datetime | None = None
    message: str = "CV uploaded successfully"


class CVListDTO(BaseModel):
    """List of CVs for a user."""

    user_id: UUID
    cvs: list[CVUploadResultDTO]
    total: int


class CVStatusDTO(BaseModel):
    """Status of active CV processing."""

    cv_id: UUID | None = None
    status: str | None = None
    uploaded_at: datetime | None = None
    error_message: str | None = None

