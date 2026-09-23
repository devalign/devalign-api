"""Telemetry ORM models for pilot metrics and thesis documentation."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.database import Base


class TelemetryEventModel(Base):
    """ORM model for the telemetry_events table.

    Captures system performance, latencies, model inference times,
    and extraction quality metrics for pilot user evaluation and thesis documentation.
    """

    __tablename__ = "telemetry_events"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), nullable=True, index=True, default=None
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="success")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    def __init__(
        self,
        *,
        event_id: UUID | None = None,
        user_id: UUID | None = None,
        event_type: str,
        duration_ms: int = 0,
        status: str = "success",
        error_message: str | None = None,
        metadata_: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            id=event_id or uuid4(),
            user_id=user_id,
            event_type=event_type,
            duration_ms=duration_ms,
            status=status,
            error_message=error_message,
            metadata_=metadata_,
            created_at=created_at or datetime.now(UTC),
            **kwargs,
        )
