"""GenAI application DTOs."""

from uuid import UUID

from pydantic import BaseModel


class RoadmapRequestDTO(BaseModel):
    """Request to generate a personalized roadmap."""

    user_id: UUID
    specialty: str
    seniority: str
    skill_gaps: list[str]


class LearningResourceDTO(BaseModel):
    title: str
    resource_type: str
    url: str | None = None
    estimated_hours: int | None = None


class RoadmapPhaseDTO(BaseModel):
    phase_number: int
    title: str
    description: str
    skills_to_acquire: list[str]
    complexity: str
    estimated_weeks: int
    sfia_reference: str | None = None
    swecom_reference: str | None = None
    resources: list[LearningResourceDTO] = []


class RoadmapDTO(BaseModel):
    """Full roadmap response returned to the frontend."""

    id: UUID
    user_id: UUID
    specialty: str
    seniority: str
    phases: list[RoadmapPhaseDTO]
    total_estimated_weeks: int
    generated_by_model: str
    message: str = "Roadmap generated successfully"
