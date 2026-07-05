"""remove_esco_uri_and_add_skill_standards

Revision ID: 55774ba07a3f
Revises: fcdbbaaaa6c7
Create Date: 2026-07-05 10:16:06.429015

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '55774ba07a3f'
down_revision: Union[str, None] = 'fcdbbaaaa6c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Purge old skills with esco_uri populated (to clear non-IT and old ESCO rows)
    op.execute("DELETE FROM skills WHERE esco_uri IS NOT NULL")

    # 2. Drop the index and column from skills table
    op.drop_index("ix_skills_esco_uri", table_name="skills")
    op.drop_column("skills", "esco_uri")

    # 3. Create the new skill_standards table
    op.create_table(
        "skill_standards",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("skill_id", sa.UUID(), nullable=False),
        sa.Column("standard_name", sa.String(length=50), nullable=False),
        sa.Column("standard_uri", sa.String(length=512), nullable=False),
        sa.Column("standard_code", sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.skill_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_skill_standards_skill_id"), "skill_standards", ["skill_id"], unique=False)
    op.create_index(op.f("ix_skill_standards_standard_uri"), "skill_standards", ["standard_uri"], unique=True)


def downgrade() -> None:
    # Drop index and table
    op.drop_index(op.f("ix_skill_standards_standard_uri"), table_name="skill_standards")
    op.drop_index(op.f("ix_skill_standards_skill_id"), table_name="skill_standards")
    op.drop_table("skill_standards")

    # Re-create column and index
    op.add_column("skills", sa.Column("esco_uri", sa.String(length=255), nullable=True))
    op.create_index("ix_skills_esco_uri", "skills", ["esco_uri"], unique=True)
