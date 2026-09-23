"""Telemetry tracker and event recording utilities for pilot evaluation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from src.shared.database import AsyncSessionLocal
from src.shared.telemetry.models import TelemetryEventModel

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


async def record_telemetry_event(
    event_type: str,
    *,
    user_id: UUID | None = None,
    duration_ms: int = 0,
    status: str = "success",
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
    session: AsyncSession | None = None,
) -> TelemetryEventModel | None:
    """Safely record a telemetry event to the database.

    If no session is provided, creates and commits its own session to ensure
    telemetry persistence even if caller transactions roll back.
    Fails safely without raising exceptions to prevent impacting business flows.
    """
    try:
        event = TelemetryEventModel(
            user_id=user_id,
            event_type=event_type,
            duration_ms=duration_ms,
            status=status,
            error_message=error_message,
            metadata_=metadata,
        )

        if session is not None:
            session.add(event)
            await session.commit()
            return event

        async with AsyncSessionLocal() as local_session:
            local_session.add(event)
            await local_session.commit()
            return event
    except Exception as exc:
        logger.warning(
            "Failed to persist telemetry event",
            event_type=event_type,
            user_id=str(user_id) if user_id else None,
            error=str(exc),
        )
        return None


class TelemetryTracker:
    """Async context manager to benchmark execution time and record telemetry.

    Example usage:
        async with TelemetryTracker("llm_extraction_phase1", user_id=uid) as tracker:
            result = await run_extraction()
            tracker.add_metadata({
                "tokens_used": result.tokens,
                "skills_count": len(result.skills),
            })
    """

    def __init__(
        self,
        event_type: str,
        *,
        user_id: UUID | None = None,
        session: AsyncSession | None = None,
        initial_metadata: dict[str, Any] | None = None,
        suppress_exceptions: bool = False,
    ) -> None:
        self.event_type = event_type
        self.user_id = user_id
        self.session = session
        self.metadata: dict[str, Any] = dict(initial_metadata or {})
        self.suppress_exceptions = suppress_exceptions
        self._start_time: float = 0.0
        self.duration_ms: int = 0
        self.event: TelemetryEventModel | None = None

    def add_metadata(self, data: dict[str, Any]) -> None:
        """Merge additional key-value metrics into event metadata."""
        self.metadata.update(data)

    async def __aenter__(self) -> TelemetryTracker:
        self._start_time = time.perf_counter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        elapsed = time.perf_counter() - self._start_time
        self.duration_ms = max(0, int(elapsed * 1000))

        if exc_val is not None:
            status = "error"
            error_message = str(exc_val)
        else:
            status = "success"
            error_message = None

        self.event = await record_telemetry_event(
            event_type=self.event_type,
            user_id=self.user_id,
            duration_ms=self.duration_ms,
            status=status,
            error_message=error_message,
            metadata=self.metadata,
            session=self.session,
        )

        # Return True if suppressing, False to propagate exception
        return self.suppress_exceptions if exc_val is not None else False
