"""Delivery module application DTOs."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr

from src.ml_engine.application.dtos import SkillDTO


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
    extracted_skills: list[SkillDTO] | None = None


class FinalizeRequestDTO(BaseModel):
    """Request body for finalizing CV analysis (Phase 2)."""

    skills: list[SkillDTO] | None = None


class FinalizeResponseDTO(BaseModel):
    """Response after triggering Phase 2 diagnosis."""

    cv_id: UUID
    status: str = "processing"
    message: str = "Diagnóstico en proceso..."
