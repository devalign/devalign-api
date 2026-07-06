"""add_ai_enriched_to_job_offers

Revision ID: 1cd3761eb205
Revises: 55774ba07a3f
Create Date: 2026-07-05 19:18:22.459767

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1cd3761eb205'
down_revision: Union[str, None] = '55774ba07a3f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "job_offers",
        sa.Column("ai_enriched", sa.Boolean(), server_default="false", nullable=False),
    )
    op.create_index(
        op.f("ix_job_offers_ai_enriched"), "job_offers", ["ai_enriched"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_job_offers_ai_enriched"), table_name="job_offers")
    op.drop_column("job_offers", "ai_enriched")

