"""add tier to clusters

Revision ID: c9d8e7f6a5b4
Revises: b3e21a4f5c6d
Create Date: 2026-09-24 13:20:00.000000

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c9d8e7f6a5b4"
down_revision: str | None = "b3e21a4f5c6d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "clusters",
        sa.Column("tier", sa.String(length=20), nullable=False, server_default="standard"),
    )
    op.create_index("ix_clusters_tier", "clusters", ["tier"])


def downgrade() -> None:
    op.drop_index("ix_clusters_tier", table_name="clusters")
    op.drop_column("clusters", "tier")
