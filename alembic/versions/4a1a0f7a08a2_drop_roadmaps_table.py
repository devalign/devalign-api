"""Drop roadmaps table

Revision ID: 4a1a0f7a08a2
Revises: 8a7b6c5d4e3f
Create Date: 2026-09-19 00:47:42.403097

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4a1a0f7a08a2'
down_revision: Union[str, None] = '8a7b6c5d4e3f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS roadmaps CASCADE")


def downgrade() -> None:
    op.create_table(
        "roadmaps",
        sa.Column("roadmap_id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "diagnostic_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("diagnostics.diagnostic_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("roadmap_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(50), nullable=False, server_default="generating"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
