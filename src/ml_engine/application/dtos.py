"""ML Engine application DTOs."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ProfileRequestDTO(BaseModel):
    """Request to profile a user from their latest CV."""

    user_id: UUID
    cv_id: UUID


class SkillDTO(BaseModel):
    """Represents a skill in the API response.

    The `inferred_from` field is non-empty only for skills that were derived
    from the knowledge graph (e.g. "SQL" inferred because the candidate has
    "PostgreSQL"). Use this field in the UI to render a tooltip explaining
    the inference rather than showing the skill as explicitly stated.
    """

    name: str
    skill_type: str
    market_importance: str | None = None
    sfia_reference: str | None = None
    market_demand_percentage: int | None = None
    inferred_from: list[str] = []  # Names of child skills that triggered this inference
    # Evidence and proficiency level
    self_taught: bool = False
    personal_projects: bool = False
    years_of_experience: int = 0
    has_certification: bool = False
    ict_score: float = 0.0
    trend: str | None = None


class DomainAffinityDTO(BaseModel):
    domain: str
    affinity_score: float
    market_demand: float = 0.5  # Relative market demand based on job postings


class ClusterAffinityDTO(BaseModel):
    cluster_id: UUID
    cluster_name: str
    affinity_score: float = Field(ge=0.0, le=1.0)
    is_primary: bool
    market_insights: dict[str, Any] | None = None
    compatible_roles: list[dict[str, Any]] | None = None
    ai_insight: str | None = None
    detected_skills: list[SkillDTO] = []
    skill_gaps: list[SkillDTO] = []
    job_offer_count: int = 0
    top_skills: list[str] = []


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
    last_analysis_date: datetime | None = None
    message: str = "Profile generated successfully"


class DiagnosticDetailDTO(BaseModel):
    user_id: UUID
    full_name: str | None = None
    current_job_role: str | None = None
    seniority: str
    last_analysis_date: datetime | None = None
    cluster_name: str
    affinity_score: float
    job_offer_count: int
    top_skills: list[str]
    market_insights: dict[str, Any] | None = None
    compatible_roles: list[dict[str, Any]] | None = None
    ai_insight: str | None = None
    detected_skills: list[SkillDTO] = []
    skill_gaps: list[SkillDTO] = []
    domain_affinities: list[DomainAffinityDTO] = []
    total_profile_skills: int


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
