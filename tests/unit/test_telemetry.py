"""Unit tests for telemetry tracking and pilot metrics collection."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.shared.telemetry.models import TelemetryEventModel
from src.shared.telemetry.tracker import TelemetryTracker, record_telemetry_event


def test_telemetry_event_model_defaults() -> None:
    """TelemetryEventModel should initialize with correct defaults."""
    uid = uuid4()
    event = TelemetryEventModel(
        user_id=uid,
        event_type="cv_parsing",
        duration_ms=120,
        metadata_={"file_size_bytes": 1024, "char_count": 500},
    )

    assert event.user_id == uid
    assert event.event_type == "cv_parsing"
    assert event.duration_ms == 120
    assert event.status == "success"
    assert event.error_message is None
    assert event.metadata_["file_size_bytes"] == 1024
    assert event.metadata_["char_count"] == 500
    assert event.created_at is not None


@pytest.mark.asyncio
async def test_record_telemetry_event_with_session() -> None:
    """record_telemetry_event should add and commit to the provided session."""
    session = AsyncMock()
    session.add = MagicMock()
    uid = uuid4()

    event = await record_telemetry_event(
        event_type="diagnosis_phase2",
        user_id=uid,
        duration_ms=450,
        status="success",
        metadata={"total_skills": 10, "standard_skills": 8, "custom_skills": 2},
        session=session,
    )

    assert event is not None
    assert event.event_type == "diagnosis_phase2"
    assert event.user_id == uid
    assert event.duration_ms == 450
    assert event.metadata_["standard_skills"] == 8
    session.add.assert_called_once_with(event)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_telemetry_event_fail_safe() -> None:
    """record_telemetry_event must not raise exceptions when the database errors."""
    broken_session = AsyncMock()
    broken_session.add = MagicMock()
    broken_session.commit.side_effect = RuntimeError("Database connection lost")

    # Should catch error, log warning, and return None without raising
    result = await record_telemetry_event(
        event_type="test_error",
        session=broken_session,
    )
    assert result is None


@pytest.mark.asyncio
async def test_telemetry_tracker_success() -> None:
    """TelemetryTracker measures elapsed time and records success status with metadata."""
    session = AsyncMock()
    session.add = MagicMock()
    uid = uuid4()

    tracker = TelemetryTracker(
        event_type="llm_extraction_phase1",
        user_id=uid,
        session=session,
        initial_metadata={"cv_id": "test-cv-123"},
    )

    async with tracker:
        await asyncio.sleep(0.01)  # Simulate small work
        tracker.add_metadata({"raw_skills_count": 15, "standardization_ratio": 0.8})

    assert tracker.duration_ms >= 0
    assert tracker.event is not None
    assert tracker.event.event_type == "llm_extraction_phase1"
    assert tracker.event.status == "success"
    assert tracker.event.metadata_["cv_id"] == "test-cv-123"
    assert tracker.event.metadata_["raw_skills_count"] == 15
    assert tracker.event.metadata_["standardization_ratio"] == 0.8
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_telemetry_tracker_captures_exception() -> None:
    """TelemetryTracker captures exceptions, records status='error', and propagates."""
    session = AsyncMock()
    session.add = MagicMock()
    uid = uuid4()

    tracker = TelemetryTracker(
        event_type="cv_parsing",
        user_id=uid,
        session=session,
        suppress_exceptions=False,
    )

    with pytest.raises(ValueError, match="Corrupted PDF"):
        async with tracker:
            raise ValueError("Corrupted PDF")

    assert tracker.event is not None
    assert tracker.event.status == "error"
    assert tracker.event.error_message == "Corrupted PDF"
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_telemetry_tracker_suppress_exception() -> None:
    """TelemetryTracker can optionally suppress exceptions when configured."""
    session = AsyncMock()
    session.add = MagicMock()

    tracker = TelemetryTracker(
        event_type="optional_step",
        session=session,
        suppress_exceptions=True,
    )

    # Should not raise
    async with tracker:
        raise KeyError("Missing optional key")

    assert tracker.event is not None
    assert tracker.event.status == "error"
    assert "Missing optional key" in (tracker.event.error_message or "")
