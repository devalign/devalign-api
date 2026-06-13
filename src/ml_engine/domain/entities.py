"""ML Engine domain entities."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID


class SeniorityLevel(StrEnum):
    """Developer seniority levels mapped to SFIA 9 responsibility levels."""

    JUNIOR = "junior"  # SFIA levels 1-2
    MID = "mid"  # SFIA level 3
    SENIOR = "senior"  # SFIA levels 4-5
    STAFF = "staff"  # SFIA level 6+


class SkillType(StrEnum):
    HARD_SKILL = "hard_skill"
    SOFT_SKILL = "soft_skill"
    METHODOLOGY = "methodology"
    TOOL = "tool"


@dataclass(frozen=True)
class Skill:
    """A technical or soft skill extracted from job offers or CVs."""

    name: str
    skill_type: SkillType
    normalized_name: str  # lowercase, no spaces (e.g. "react.js" → "reactjs")
    weight: float = 1.0
    frequency: float = 1.0  # Relative frequency in a cluster (if applicable)
    domain: str | None = None
    id: UUID | None = None


@dataclass(frozen=True)
class TechCluster:
    """A cluster of co-occurring technologies representing a specialty."""

    id: UUID
    name: str  # e.g. "Backend Cloud-Native Java"
    description: str
    centroid_skills: list[Skill]  # Most representative skills
    job_offer_count: int  # How many offers belong to this cluster
    cluster_index: int  # K-Prototypes cluster number
    market_insights: dict[str, Any] = field(default_factory=dict)
    compatible_roles: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ClusterAffinity:
    """Affinity score between a user profile and a cluster."""

    cluster_id: UUID
    cluster_name: str
    affinity_score: float  # Cosine similarity [0, 1]
    is_primary: bool  # Highest score = primary specialty
    market_insights: dict[str, Any] = field(default_factory=dict)
    compatible_roles: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SkillGap:
    """A skill that a user is missing compared to their target specialty."""

    skill: Skill
    market_importance: str  # "critical" | "high" | "medium"
    sfia_reference: str | None = None  # SFIA 9 skill ID (e.g. "PROG")


@dataclass(frozen=True)
class UserProfile:
    """The computed profile of a developer based on their CV."""

    user_id: UUID
    cv_id: UUID
    embedding: list[float]  # Vector representation of CV
    detected_skills: list[Skill]
    seniority: SeniorityLevel
    primary_affinity: ClusterAffinity
    secondary_affinities: list[ClusterAffinity] = field(default_factory=list)
    skill_gaps: list[SkillGap] = field(default_factory=list)
    full_name: str | None = None
    current_job_role: str | None = None
    years_experience: int | None = None
    preferred_modality: str | None = None
    location: str | None = None
    availability: str | None = None
    work_experience: list[dict[str, Any]] = field(default_factory=list)
    education: list[dict[str, Any]] = field(default_factory=list)
    certifications: list[dict[str, Any]] = field(default_factory=list)

    @property
    def primary_specialty(self) -> str:
        return self.primary_affinity.cluster_name

    @property
    def alignment_score(self) -> float:
        return self.primary_affinity.affinity_score
