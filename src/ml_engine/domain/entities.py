"""ML Engine domain entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID


class SeniorityLevel(StrEnum):
    """Developer seniority levels mapped to SFIA 9 responsibility levels."""

    JUNIOR = "junior"  # SFIA levels 1-2
    MID = "mid"  # SFIA level 3
    SENIOR = "senior"  # SFIA levels 4-5
    STAFF = "staff"  # SFIA level 6+


class SkillNature(StrEnum):
    """The fundamental nature of a skill."""

    CONCEPT = "concept"
    TECH = "tech"
    SOFT = "soft"


class SkillRelationType(StrEnum):
    """Types of edges in the knowledge graph."""

    REQUIRES = "requires"
    ALTERNATIVE_TO = "alternative_to"
    BELONGS_TO = "belongs_to"
    ESSENTIAL = "essential"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class SkillRelation:
    """An edge between two skills in the knowledge graph."""

    target_skill_id: UUID
    target_skill_name: str
    relation_type: SkillRelationType


@dataclass(frozen=True)
class Skill:
    """A technical or soft skill extracted from job offers or CVs."""

    name: str
    nature: SkillNature
    normalized_name: str  # lowercase, no spaces (e.g. "react.js" → "reactjs")
    domain_tags: list[str] = field(default_factory=list)
    core_domains: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    relations: list[SkillRelation] = field(default_factory=list)
    weight: float = 1.0
    frequency: float = 1.0  # Relative frequency in a cluster (if applicable)
    embedding: list[float] | None = None
    id: UUID | None = None
    # Non-empty only for skills inferred via the knowledge graph.
    # Contains the canonical names of child skills that triggered this inference
    # (e.g. ["PostgreSQL"] when SQL is inferred because the CV mentions PostgreSQL).
    inferred_from: list[str] = field(default_factory=list)
    # Evidence and proficiency level
    self_taught: bool = False
    personal_projects: bool = False
    years_of_experience: int = 0
    has_certification: bool = False
    ict_score: float = 0.0

    def calculate_ict(self) -> float:
        exp_points = 3 * self.years_of_experience
        cert_points = 4 if self.has_certification else 0
        projects_points = 2 if self.personal_projects else 0
        self_taught_points = 1 if self.self_taught else 0
        return float(min(10.0, self_taught_points + projects_points + exp_points + cert_points))


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
    is_primary: bool = False
    market_insights: dict[str, Any] | None = None
    compatible_roles: list[dict[str, Any]] | None = None
    ai_insight: str | None = None
    detected_skills: list[Skill] = field(default_factory=list)
    skill_gaps: list[SkillGap] = field(default_factory=list)
    job_offer_count: int = 0
    top_skills: list[str] = field(default_factory=list)


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
    cv_id: UUID | None
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
    cv_raw_text: str | None = None
    last_analysis_date: datetime | None = None

    @property
    def primary_specialty(self) -> str:
        return self.primary_affinity.cluster_name

    @property
    def alignment_score(self) -> float:
        return self.primary_affinity.affinity_score
