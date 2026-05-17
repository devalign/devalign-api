"""GenAI module domain entities."""

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID


class PhaseComplexity(str, Enum):
    FOUNDATIONAL = "foundational"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass(frozen=True)
class LearningResource:
    """A recommended learning resource for a roadmap phase."""

    title: str
    resource_type: str  # "course" | "book" | "documentation" | "project"
    url: str | None = None
    estimated_hours: int | None = None


@dataclass(frozen=True)
class RoadmapPhase:
    """A phase in the learning roadmap."""

    phase_number: int
    title: str
    description: str
    skills_to_acquire: list[str]
    complexity: PhaseComplexity
    estimated_weeks: int
    sfia_reference: str | None = None  # SFIA 9 skill ID this phase maps to
    swecom_reference: str | None = None  # SWECOM area reference
    resources: list[LearningResource] = field(default_factory=list)


@dataclass(frozen=True)
class Roadmap:
    """A personalized learning roadmap for a developer."""

    id: UUID
    user_id: UUID
    specialty: str
    seniority: str
    phases: list[RoadmapPhase]
    total_estimated_weeks: int
    generated_by_model: str  # LLM model identifier

    @classmethod
    def compute_total_weeks(cls, phases: list[RoadmapPhase]) -> int:
        return sum(p.estimated_weeks for p in phases)
