"""SQLAlchemy ORM models for the GenAI module.

Maps to the roadmaps table. The full roadmap content is stored as JSONB
to avoid premature normalization of the LLM-generated structure.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 — required by SQLAlchemy Mapped[] at runtime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    pass

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.genai.domain.entities import PhaseComplexity, Roadmap, RoadmapPhase
from src.shared.database import Base


class RoadmapModel(Base):
    """ORM model for the roadmaps table.

    Persists the output of the RAG + LLM pipeline.

    Design decision: roadmap_json stores the full structured roadmap
    as JSONB. This avoids creating a complex relational schema for
    phases/resources that would complicate LLM output parsing.
    The JSONB can be queried by Postgres if needed.

    Status lifecycle: "generating" → "completed" | "failed"
    """

    __tablename__ = "roadmaps"

    roadmap_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    diagnostic_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("diagnostics.diagnostic_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Full roadmap content from the LLM — phases, skills, resources, SFIA references
    roadmap_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # "generating" | "completed" | "failed"
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="generating", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def to_entity(self) -> Roadmap:
        """Convert ORM model to domain entity.

        Reconstructs the Roadmap entity from the JSONB blob.
        """
        phases_data = self.roadmap_json.get("phases", [])
        phases = [
            RoadmapPhase(
                phase_number=p.get("phase_number", i + 1),
                title=p.get("title", ""),
                description=p.get("description", ""),
                skills_to_acquire=p.get("skills_to_acquire", []),
                complexity=PhaseComplexity(p.get("complexity", "foundational")),
                estimated_weeks=p.get("estimated_weeks", 0),
                sfia_reference=p.get("sfia_reference"),
                swecom_reference=p.get("swecom_reference"),
            )
            for i, p in enumerate(phases_data)
        ]
        return Roadmap(
            id=self.roadmap_id,
            user_id=self.roadmap_json.get("user_id", ""),  # type: ignore[arg-type]
            specialty=self.roadmap_json.get("specialty", ""),
            seniority=self.roadmap_json.get("seniority", ""),
            phases=phases,
            total_estimated_weeks=sum(p.estimated_weeks for p in phases),
            generated_by_model=self.roadmap_json.get("generated_by_model", ""),
        )
