"""add_status_to_cv_documents

Revision ID: 6d3f97e10a1d
Revises: 0e407f4cfc27
Create Date: 2026-06-29 16:32:28.805784

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6d3f97e10a1d'
down_revision: Union[str, None] = '0e407f4cfc27'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add status column to cv_documents
    op.add_column(
        "cv_documents",
        sa.Column("status", sa.String(length=50), nullable=False, server_default="processing"),
    )
    # Enable supabase_realtime for cv_documents if the publication exists
    op.execute("""
        DO $$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'supabase_realtime') THEN
            ALTER PUBLICATION supabase_realtime ADD TABLE cv_documents;
          END IF;
        END $$;
    """)


def downgrade() -> None:
    # Remove from supabase_realtime publication
    op.execute("""
        DO $$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'supabase_realtime') THEN
            ALTER PUBLICATION supabase_realtime DROP TABLE cv_documents;
          END IF;
        END $$;
    """)
    # Drop status column
    op.drop_column("cv_documents", "status")
