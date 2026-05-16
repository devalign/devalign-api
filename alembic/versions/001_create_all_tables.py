"""001 — Create all tables for Devalign schema.

Revision: 001_create_all_tables
Creates: users, cv_documents, profiles, skills, clusters, cluster_skills,
         job_offers, offer_skills, diagnostics, diagnostic_skills, roadmaps

Notes:
    - Enables pgvector extension for Vector(384) columns.
    - users.user_id references auth.users(id) from Supabase Auth.
    - job_offers.source_url has a UNIQUE constraint (upsert key for the scraper).
    - cluster_skills and offer_skills are junction tables.
    - Roadmap content stored as JSONB to accommodate dynamic LLM output.
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# Revision identifiers
revision = "001_create_all_tables"
down_revision = None
branch_labels = None
depends_on = None

# Embedding dimension — must match EMBEDDING_DIM in ml_engine/infrastructure/models.py
EMBEDDING_DIM = 384


def upgrade() -> None:
    # ── Enable pgvector extension ────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── 1. users ─────────────────────────────────────────────────────────────
    # Public application table. user_id references Supabase Auth's auth.users(id).
    # No password_hash — Supabase Auth manages credentials.
    op.create_table(
        "users",
        sa.Column("user_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("avatar_url", sa.String(512), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── 2. cv_documents ───────────────────────────────────────────────────────
    op.create_table(
        "cv_documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("storage_path", sa.String(512), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(128), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_cv_documents_user_id", "cv_documents", ["user_id"])

    # ── 3. skills ─────────────────────────────────────────────────────────────
    # Central normalized skill catalog — shared by offer_skills, cluster_skills,
    # and diagnostic_skills. Populated by the ML Engine normalization pipeline.
    op.create_table(
        "skills",
        sa.Column("skill_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_skills_name", "skills", ["name"])
    op.create_unique_constraint("uq_skills_name", "skills", ["name"])

    # ── 4. clusters ───────────────────────────────────────────────────────────
    # K-Prototypes clusters of co-occurring technologies.
    # centroid_vec is a pgvector Vector(384) — cosine similarity target.
    op.create_table(
        "clusters",
        sa.Column("cluster_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(150), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("centroid_vec", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ── 5. cluster_skills ─────────────────────────────────────────────────────
    op.create_table(
        "cluster_skills",
        sa.Column("cluster_skill_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "cluster_id",
            UUID(as_uuid=True),
            sa.ForeignKey("clusters.cluster_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "skill_id",
            UUID(as_uuid=True),
            sa.ForeignKey("skills.skill_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("importance_score", sa.Numeric(5, 2), nullable=True),
    )
    op.create_index("ix_cluster_skills_cluster_id", "cluster_skills", ["cluster_id"])
    op.create_index("ix_cluster_skills_skill_id", "cluster_skills", ["skill_id"])

    # ── 6. job_offers ─────────────────────────────────────────────────────────
    # Produced by devalign-scraping (Supabase upsert on source_url).
    # raw_hard_skills / raw_soft_skills: JSONB staging columns for ML normalization.
    # salary / experience_years: varchar — raw text as extracted by scraper.
    op.create_table(
        "job_offers",
        sa.Column("job_offer_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "cluster_id",
            UUID(as_uuid=True),
            sa.ForeignKey("clusters.cluster_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("job_title", sa.String(150), nullable=False),
        sa.Column("company", sa.String(150), nullable=True),
        sa.Column("location", sa.String(100), nullable=True),
        sa.Column("modality", sa.String(50), nullable=True),
        sa.Column("salary", sa.String(100), nullable=True),
        sa.Column("experience_years", sa.String(100), nullable=True),
        sa.Column("education_level", sa.String(100), nullable=True),
        sa.Column("full_description", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("portal", sa.String(100), nullable=True),
        sa.Column("date_posted", sa.String(50), nullable=True),
        sa.Column("raw_hard_skills", JSONB, nullable=True),
        sa.Column("raw_soft_skills", JSONB, nullable=True),
        sa.Column(
            "scraped_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_job_offers_job_title", "job_offers", ["job_title"])
    op.create_index("ix_job_offers_cluster_id", "job_offers", ["cluster_id"])
    op.create_unique_constraint("uq_job_offers_source_url", "job_offers", ["source_url"])

    # ── 7. offer_skills ───────────────────────────────────────────────────────
    # Populated by ML Engine after normalizing raw_hard_skills / raw_soft_skills.
    op.create_table(
        "offer_skills",
        sa.Column("offer_skill_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_offer_id",
            UUID(as_uuid=True),
            sa.ForeignKey("job_offers.job_offer_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "skill_id",
            UUID(as_uuid=True),
            sa.ForeignKey("skills.skill_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("skill_type", sa.String(50), nullable=False),
        sa.Column("importance_score", sa.Numeric(5, 2), nullable=True),
    )
    op.create_index("ix_offer_skills_job_offer_id", "offer_skills", ["job_offer_id"])
    op.create_index("ix_offer_skills_skill_id", "offer_skills", ["skill_id"])

    # ── 8. profiles ───────────────────────────────────────────────────────────
    # Developer profile — 1:1 with users.
    # cv_embedding: Vector(384) for cosine similarity vs cluster centroids.
    op.create_table(
        "profiles",
        sa.Column("profile_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("full_name", sa.String(150), nullable=True),
        sa.Column("current_job_role", sa.String(100), nullable=True),
        sa.Column("years_experience", sa.Integer(), nullable=True),
        sa.Column("preferred_modality", sa.String(50), nullable=True),
        sa.Column("cv_url", sa.Text(), nullable=True),
        sa.Column("cv_raw_text", sa.Text(), nullable=True),
        sa.Column("cv_embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_profiles_user_id", "profiles", ["user_id"], unique=True)

    # ── 9. diagnostics ────────────────────────────────────────────────────────
    # Result of comparing a user profile against a cluster.
    op.create_table(
        "diagnostics",
        sa.Column("diagnostic_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "profile_id",
            UUID(as_uuid=True),
            sa.ForeignKey("profiles.profile_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "detected_cluster_id",
            UUID(as_uuid=True),
            sa.ForeignKey("clusters.cluster_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("affinity_score", sa.Numeric(5, 2), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_diagnostics_profile_id", "diagnostics", ["profile_id"])
    op.create_index("ix_diagnostics_cluster_id", "diagnostics", ["detected_cluster_id"])

    # ── 10. diagnostic_skills ─────────────────────────────────────────────────
    # Skills per diagnostic with status: "consolidated" | "gap" | "emerging"
    op.create_table(
        "diagnostic_skills",
        sa.Column("diagnostic_skill_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "diagnostic_id",
            UUID(as_uuid=True),
            sa.ForeignKey("diagnostics.diagnostic_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "skill_id",
            UUID(as_uuid=True),
            sa.ForeignKey("skills.skill_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("skill_status", sa.String(50), nullable=False),
        sa.Column("importance_score", sa.Numeric(5, 2), nullable=True),
    )
    op.create_index("ix_diagnostic_skills_diagnostic_id", "diagnostic_skills", ["diagnostic_id"])
    op.create_index("ix_diagnostic_skills_skill_id", "diagnostic_skills", ["skill_id"])

    # ── 11. roadmaps ──────────────────────────────────────────────────────────
    # LLM-generated roadmap stored as JSONB. Status: "generating" | "completed" | "failed"
    op.create_table(
        "roadmaps",
        sa.Column("roadmap_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "diagnostic_id",
            UUID(as_uuid=True),
            sa.ForeignKey("diagnostics.diagnostic_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("roadmap_json", JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(50), nullable=False, server_default="generating"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_roadmaps_diagnostic_id", "roadmaps", ["diagnostic_id"])
    op.create_index("ix_roadmaps_status", "roadmaps", ["status"])


def downgrade() -> None:
    op.drop_table("roadmaps")
    op.drop_table("diagnostic_skills")
    op.drop_table("diagnostics")
    op.drop_table("profiles")
    op.drop_table("offer_skills")
    op.drop_table("job_offers")
    op.drop_table("cluster_skills")
    op.drop_table("clusters")
    op.drop_table("skills")
    op.drop_table("cv_documents")
    op.drop_table("users")
    op.execute("DROP EXTENSION IF EXISTS vector")
