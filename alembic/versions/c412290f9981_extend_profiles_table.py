"""extend_profiles_table

Revision ID: c412290f9981
Revises: 002_add_is_normalized
Create Date: 2026-05-29 19:04:51.931428

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c412290f9981"
down_revision: str | None = "002_add_is_normalized"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    from sqlalchemy.dialects.postgresql import JSONB

    op.add_column(
        "profiles", sa.Column("work_experience", JSONB(), nullable=False, server_default="[]")
    )
    op.add_column("profiles", sa.Column("education", JSONB(), nullable=False, server_default="[]"))
    op.add_column(
        "profiles", sa.Column("certifications", JSONB(), nullable=False, server_default="[]")
    )
    op.add_column("profiles", sa.Column("location", sa.String(length=100), nullable=True))
    op.add_column("profiles", sa.Column("availability", sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column("profiles", "availability")
    op.drop_column("profiles", "location")
    op.drop_column("profiles", "certifications")
    op.drop_column("profiles", "education")
    op.drop_column("profiles", "work_experience")
