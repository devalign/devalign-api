"""add is_normalized to job_offers

Revision ID: 002_add_is_normalized
Revises: 001_create_all_tables
Create Date: 2026-05-17 03:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002_add_is_normalized'
down_revision: Union[str, None] = '001_create_all_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the column with a default value of False
    op.add_column('job_offers', sa.Column('is_normalized', sa.Boolean(), server_default='false', nullable=False))
    # Create an index on it since we'll be querying WHERE is_normalized = False
    op.create_index(op.f('ix_job_offers_is_normalized'), 'job_offers', ['is_normalized'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_job_offers_is_normalized'), table_name='job_offers')
    op.drop_column('job_offers', 'is_normalized')
