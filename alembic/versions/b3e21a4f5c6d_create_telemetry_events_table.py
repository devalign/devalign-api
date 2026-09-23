"""create telemetry_events table

Revision ID: b3e21a4f5c6d
Revises: 4a1a0f7a08a2
Create Date: 2026-09-23 17:16:00.000000

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "b3e21a4f5c6d"
down_revision: str | None = "4a1a0f7a08a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "telemetry_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="success"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_telemetry_events_user_id", "telemetry_events", ["user_id"])
    op.create_index("ix_telemetry_events_event_type", "telemetry_events", ["event_type"])
    op.create_index("ix_telemetry_events_created_at", "telemetry_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_telemetry_events_created_at", table_name="telemetry_events")
    op.drop_index("ix_telemetry_events_event_type", table_name="telemetry_events")
    op.drop_index("ix_telemetry_events_user_id", table_name="telemetry_events")
    op.drop_table("telemetry_events")
