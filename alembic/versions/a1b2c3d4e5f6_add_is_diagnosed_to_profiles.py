"""add is_diagnosed to profiles

Revision ID: a1b2c3d4e5f6
Revises: c412290f9981
Create Date: 2026-07-01 14:00:00.000000

Supports the 2-phase CV analysis pipeline:
- Phase 1 marks cv_documents.status = 'completed' → profile visible immediately
- Phase 2 sets profiles.is_diagnosed = true → full skill/affinity data ready

This lets the frontend show basic profile data in <10s while the heavier
skill normalisation and cluster affinity computation finish in the background.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "6d3f97e10a1d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "profiles",
        sa.Column(
            "is_diagnosed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("profiles", "is_diagnosed")
