"""SQLAlchemy ORM models for the ML Engine module.

Maps to: profiles, skills, clusters, cluster_skills, diagnostics, diagnostic_skills.

Architecture notes:
    - clusters.centroid_vec uses pgvector's Vector(384) type — matches all-MiniLM-L6-v2.
    - profiles.cv_embedding uses Vector(384) for semantic similarity search.
    - skill normalization pipeline reads job_offers.raw_hard_skills and populates skills + offer_skills.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — required by SQLAlchemy Mapped[] at runtime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import Base

# Embedding dimension for voyage-4-lite (Voyage AI)
# Change this constant if switching embedding models — also update Alembic migration.
EMBEDDING_DIM = 1024


class SkillModel(Base):
    """ORM model for the skills table.

    Central catalog of normalized technical and soft skills.
    Shared across: offer_skills, cluster_skills, diagnostic_skills.
    """

    __tablename__ = "skills"
    __table_args__ = (UniqueConstraint("name", name="uq_skills_name"),)

    skill_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    # Normalized skill name — lowercase, canonical form (e.g. "react.js", "kubernetes")
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # "hard_skill" | "soft_skill" | "methodology" | "tool"
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    cluster_skills: Mapped[list[ClusterSkillModel]] = relationship(
        "ClusterSkillModel", back_populates="skill", lazy="select"
    )
    diagnostic_skills: Mapped[list[DiagnosticSkillModel]] = relationship(
        "DiagnosticSkillModel", back_populates="skill", lazy="select"
    )


class ClusterModel(Base):
    """ORM model for the clusters table.

    Represents a K-Prototypes cluster of co-occurring technologies.
    centroid_vec is a pgvector Vector(384) column — the centroid of all
    CV embeddings assigned to this cluster.
    """

    __tablename__ = "clusters"

    cluster_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    # Human-readable cluster name (e.g. "Backend Cloud-Native Java")
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # pgvector column — centroid of all embeddings in this cluster
    # Dimensioned for all-MiniLM-L6-v2 (384 dims)
    centroid_vec: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    cluster_skills: Mapped[list[ClusterSkillModel]] = relationship(
        "ClusterSkillModel", back_populates="cluster", lazy="select", cascade="all, delete-orphan"
    )
    diagnostics: Mapped[list[DiagnosticModel]] = relationship(
        "DiagnosticModel", back_populates="detected_cluster", lazy="select"
    )


class ClusterSkillModel(Base):
    """ORM model for the cluster_skills table.

    Junction table linking a cluster to its representative skills,
    with an importance score reflecting how central each skill is.
    """

    __tablename__ = "cluster_skills"

    cluster_skill_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    cluster_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("clusters.cluster_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    skill_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("skills.skill_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    importance_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Relationships
    cluster: Mapped[ClusterModel] = relationship("ClusterModel", back_populates="cluster_skills")
    skill: Mapped[SkillModel] = relationship("SkillModel", back_populates="cluster_skills")


class ProfileModel(Base):
    """ORM model for the profiles table.

    Stores the developer's self-reported data + computed CV embedding.
    cv_embedding is a pgvector Vector(384) for cosine similarity search
    against cluster centroids.
    """

    __tablename__ = "profiles"

    profile_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    full_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    current_job_role: Mapped[str | None] = mapped_column(String(100), nullable=True)
    years_experience: Mapped[int | None] = mapped_column(nullable=True)
    preferred_modality: Mapped[str | None] = mapped_column(String(50), nullable=True)
    cv_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    cv_raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    cv_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    # pgvector embedding of the CV raw text — used for cosine similarity vs cluster centroids
    cv_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    work_experience: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    education: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    certifications: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    location: Mapped[str | None] = mapped_column(String(100), nullable=True)
    availability: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    diagnostics: Mapped[list[DiagnosticModel]] = relationship(
        "DiagnosticModel", back_populates="profile", lazy="select", cascade="all, delete-orphan"
    )


class DiagnosticModel(Base):
    """ORM model for the diagnostics table.

    Result of comparing a user profile against a cluster.
    Stores the detected specialty and affinity score.
    """

    __tablename__ = "diagnostics"

    diagnostic_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    profile_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("profiles.profile_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    detected_cluster_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("clusters.cluster_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    affinity_score: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    profile: Mapped[ProfileModel] = relationship("ProfileModel", back_populates="diagnostics")
    detected_cluster: Mapped[ClusterModel] = relationship(
        "ClusterModel", back_populates="diagnostics"
    )
    diagnostic_skills: Mapped[list[DiagnosticSkillModel]] = relationship(
        "DiagnosticSkillModel",
        back_populates="diagnostic",
        lazy="select",
        cascade="all, delete-orphan",
    )


class DiagnosticSkillModel(Base):
    """ORM model for the diagnostic_skills table.

    Maps each skill detected in a diagnostic to its status:
    - "consolidated" — user already has this skill
    - "gap"          — user is missing this skill vs the cluster standard
    - "emerging"     — skill is growing in demand but not yet critical
    """

    __tablename__ = "diagnostic_skills"

    diagnostic_skill_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    diagnostic_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("diagnostics.diagnostic_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    skill_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("skills.skill_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # "consolidated" | "gap" | "emerging"
    skill_status: Mapped[str] = mapped_column(String(50), nullable=False)
    importance_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Relationships
    diagnostic: Mapped[DiagnosticModel] = relationship(
        "DiagnosticModel", back_populates="diagnostic_skills"
    )
    skill: Mapped[SkillModel] = relationship("SkillModel", back_populates="diagnostic_skills")
