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
    market_demand_percentage: int | None = None


class DomainAffinityDTO(BaseModel):
    domain: str
    affinity_score: float


class ClusterAffinityDTO(BaseModel):
    cluster_id: UUID
    cluster_name: str
    affinity_score: float = Field(ge=0.0, le=1.0)
    is_primary: bool
    market_insights: dict[str, Any] | None = None
    compatible_roles: list[dict[str, Any]] | None = None
    ai_insight: str | None = None


class UserProfileDTO(BaseModel):
    """User profile result returned after CV analysis."""

    user_id: UUID
    cv_id: UUID | None = None
    seniority: str
    primary_specialty: str
    alignment_score: float = Field(ge=0.0, le=1.0)
    secondary_affinities: list[ClusterAffinityDTO] = []
    all_affinities: list[ClusterAffinityDTO] = []
    domain_affinities: list[DomainAffinityDTO] = []
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


class GraphNodeDTO(BaseModel):
    id: str
    label: str
    group: str
    domains: list[str] = []
    status: str = "neutral"  # "acquired", "gap", "neutral"


class GraphLinkDTO(BaseModel):
    source: str
    target: str
    value: float = 1.0
    type: str = "implicit"  # "explicit_relation", "implicit_domain"


class GraphResponseDTO(BaseModel):
    nodes: list[GraphNodeDTO]
    links: list[GraphLinkDTO]
