"""ML Engine application DTOs."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ProfileRequestDTO(BaseModel):
    """Request to profile a user from their latest CV."""

    user_id: UUID
    cv_id: UUID


class SkillDTO(BaseModel):
    name: str
    skill_type: str
    market_importance: str | None = None
    sfia_reference: str | None = None


class ClusterAffinityDTO(BaseModel):
    cluster_id: UUID
    cluster_name: str
    affinity_score: float = Field(ge=0.0, le=1.0)
    is_primary: bool


class UserProfileDTO(BaseModel):
    """User profile result returned after CV analysis."""

    user_id: UUID
    cv_id: UUID
    seniority: str
    primary_specialty: str
    alignment_score: float = Field(ge=0.0, le=1.0)
    secondary_affinities: list[ClusterAffinityDTO] = []
    detected_skills: list[SkillDTO] = []
    skill_gaps: list[SkillDTO] = []
    full_name: str | None = None
    current_job_role: str | None = None
    years_experience: int | None = None
    preferred_modality: str | None = None
    location: str | None = None
    availability: str | None = None
    work_experience: list[dict[str, Any]] = []
    education: list[dict[str, Any]] = []
    certifications: list[dict[str, Any]] = []
    message: str = "Profile generated successfully"


class ClusterDTO(BaseModel):
    """Public representation of a tech cluster."""

    id: UUID
    name: str
    description: str
    top_skills: list[str]
    job_offer_count: int


class ProfileUpdateDTO(BaseModel):
    full_name: str | None = None
    current_job_role: str | None = None
    years_experience: int | None = None
    preferred_modality: str | None = None
    location: str | None = None
    availability: str | None = None
    work_experience: list[dict[str, Any]] | None = None
    education: list[dict[str, Any]] | None = None
    certifications: list[dict[str, Any]] | None = None


class SkillsUpdateDTO(BaseModel):
    skills: list[SkillDTO]
