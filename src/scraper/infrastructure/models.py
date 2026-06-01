"""SQLAlchemy ORM models for the scraper module.

Maps to the job_offers and offer_skills tables in the database.
The scraper (external repo) populates job_offers via Supabase upsert.
The ml_engine normalizes raw_hard_skills/raw_soft_skills into offer_skills.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — required by SQLAlchemy Mapped[] at runtime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.scraper.domain.entities import JobOffer, JobPlatform
from src.shared.database import Base


class JobOfferModel(Base):
    """ORM model for the job_offers table.

    Produced by devalign-scraping (Supabase upsert).
    Consumed by devalign-api (ML Engine for clustering).

    Notes:
        - salary is stored as raw text (e.g. "S/. 3,500", "A convenir", "USD 2,500 - 4,000")
          because the scraper cannot always extract a clean numeric value.
        - experience_years is stored as raw text (e.g. "2 a 4 años de experiencia", "No especificado").
        - raw_hard_skills / raw_soft_skills are JSONB staging columns. They hold the
          raw list-of-strings extracted by the scraper. The ML engine reads them to
          normalize and populate the offer_skills table.
        - cluster_id is FK to clusters — populated by the ML Engine after clustering.
    """

    __tablename__ = "job_offers"

    job_offer_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    cluster_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("clusters.cluster_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    job_title: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    company: Mapped[str | None] = mapped_column(String(150), nullable=True)
    location: Mapped[str | None] = mapped_column(String(100), nullable=True)
    modality: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Raw salary text from scraper — "S/. 3,500", "A convenir", "USD 2,500 - 4,000"
    salary: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Raw experience text — "2 a 4 años de experiencia", "5+", "No especificado"
    experience_years: Mapped[str | None] = mapped_column(String(100), nullable=True)
    education_level: Mapped[str | None] = mapped_column(String(100), nullable=True)
    full_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    portal: Mapped[str | None] = mapped_column(String(100), nullable=True)
    date_posted: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # JSONB staging: raw skill lists from the scraper (e.g. ["python", "docker"])
    raw_hard_skills: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    raw_soft_skills: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    is_normalized: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false", index=True
    )
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    offer_skills: Mapped[list[OfferSkillModel]] = relationship(
        "OfferSkillModel", back_populates="job_offer", lazy="select", cascade="all, delete-orphan"
    )

    def to_entity(self) -> JobOffer:
        return JobOffer(
            id=self.job_offer_id,
            platform=JobPlatform(self.portal) if self.portal else JobPlatform.COMPUTRABAJO,
            title=self.job_title,
            company=self.company or "",
            location=self.location or "",
            raw_description=self.full_description or "",
            salary=self.salary or "",
            modality=self.modality or "",
            experience_years=self.experience_years or "",
            education_level=self.education_level or "",
            source_url=self.source_url,
            date_posted=self.date_posted or "",
            hard_skills=self.raw_hard_skills or [],
            soft_skills=self.raw_soft_skills or [],
        )


class OfferSkillModel(Base):
    """ORM model for the offer_skills table.

    Junction table linking a job_offer to a normalized skill.
    Populated by the ML Engine skill normalization pipeline.
    """

    __tablename__ = "offer_skills"

    offer_skill_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    job_offer_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("job_offers.job_offer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    skill_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("skills.skill_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # "hard_skill" | "soft_skill" | "methodology" | "tool"
    skill_type: Mapped[str] = mapped_column(String(50), nullable=False)
    importance_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Relationships
    job_offer: Mapped[JobOfferModel] = relationship("JobOfferModel", back_populates="offer_skills")
