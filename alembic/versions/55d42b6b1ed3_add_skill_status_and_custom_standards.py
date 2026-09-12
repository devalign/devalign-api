"""add_skill_status_and_custom_standards

Revision ID: 55d42b6b1ed3
Revises: 1cd3761eb205
Create Date: 2026-09-11 12:56:03.334939

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '55d42b6b1ed3'
down_revision: Union[str, None] = '1cd3761eb205'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add status column to skills with default 'canonical'
    op.add_column(
        "skills",
        sa.Column("status", sa.String(length=20), server_default="canonical", nullable=False),
    )

    # 2. Add check constraint to enforce lifecycle states
    op.create_check_constraint(
        "chk_skills_status",
        "skills",
        "status IN ('canonical', 'pending_review', 'deprecated')",
    )

    # 3. Create index for fast filtering in autocomplete and graph
    op.create_index(op.f("ix_skills_status"), "skills", ["status"], unique=False)

    # 4. Ensure all existing skills are canonical
    op.execute("UPDATE skills SET status = 'canonical'")

    # 5. Backfill non-standard skills into skill_standards with standard_name='Custom'
    op.execute("""
        INSERT INTO skill_standards (id, skill_id, standard_name, standard_uri, standard_code)
        SELECT
            gen_random_uuid(),
            s.skill_id,
            'Custom',
            'devalign:skill:custom:' || s.skill_id::text,
            'CUSTOM'
        FROM skills s
        LEFT JOIN skill_standards ss ON ss.skill_id = s.skill_id
        WHERE ss.skill_id IS NULL;
    """)


def downgrade() -> None:
    # 1. Remove backfilled Custom standards
    op.execute("DELETE FROM skill_standards WHERE standard_name = 'Custom'")

    # 2. Drop index
    op.drop_index(op.f("ix_skills_status"), table_name="skills")

    # 3. Drop check constraint
    op.drop_constraint("chk_skills_status", "skills", type_="check")

    # 4. Drop status column
    op.drop_column("skills", "status")

