"""add_structured_market_fields_to_job_offers

Revision ID: 8a7b6c5d4e3f
Revises: 29c892425359
Create Date: 2026-09-17 15:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8a7b6c5d4e3f'
down_revision: Union[str, None] = '29c892425359'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add structured salary and currency columns
    op.add_column(
        "job_offers",
        sa.Column("min_salary_usd", sa.Numeric(10, 2), nullable=True),
    )
    op.add_column(
        "job_offers",
        sa.Column("max_salary_usd", sa.Numeric(10, 2), nullable=True),
    )
    op.add_column(
        "job_offers",
        sa.Column("currency", sa.String(length=10), nullable=True),
    )
    op.add_column(
        "job_offers",
        sa.Column("is_salary_negotiable", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )

    # 2. Add structured experience years columns
    op.add_column(
        "job_offers",
        sa.Column("min_experience_years", sa.Integer(), nullable=True),
    )
    op.add_column(
        "job_offers",
        sa.Column("max_experience_years", sa.Integer(), nullable=True),
    )

    # 3. Add absolute publication timestamp
    op.add_column(
        "job_offers",
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 4. Create indexes for performance
    op.create_index(
        op.f("ix_job_offers_min_salary_usd"),
        "job_offers",
        ["min_salary_usd"],
        unique=False,
    )
    op.create_index(
        op.f("ix_job_offers_min_experience_years"),
        "job_offers",
        ["min_experience_years"],
        unique=False,
    )
    op.create_index(
        op.f("ix_job_offers_published_at"),
        "job_offers",
        ["published_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_job_offers_published_at"), table_name="job_offers")
    op.drop_index(op.f("ix_job_offers_min_experience_years"), table_name="job_offers")
    op.drop_index(op.f("ix_job_offers_min_salary_usd"), table_name="job_offers")

    op.drop_column("job_offers", "published_at")
    op.drop_column("job_offers", "max_experience_years")
    op.drop_column("job_offers", "min_experience_years")
    op.drop_column("job_offers", "is_salary_negotiable")
    op.drop_column("job_offers", "currency")
    op.drop_column("job_offers", "max_salary_usd")
    op.drop_column("job_offers", "min_salary_usd")
