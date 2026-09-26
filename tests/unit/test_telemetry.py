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

    broken_local = AsyncMock()
    broken_local.add = MagicMock()
    broken_local.commit.side_effect = RuntimeError("Local connection lost")
    session_ctx = AsyncMock()
    session_ctx.__aenter__.return_value = broken_local
    session_ctx.__aexit__.return_value = None

    import src.shared.telemetry.tracker as tracker_mod

    orig_session_factory = tracker_mod.AsyncSessionLocal
    tracker_mod.AsyncSessionLocal = MagicMock(return_value=session_ctx)  # type: ignore[misc]

    try:
        # Should catch error, log warning, and return None without raising
        result = await record_telemetry_event(
            event_type="test_error",
            session=broken_session,
        )
        assert result is None
    finally:
        tracker_mod.AsyncSessionLocal = orig_session_factory


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


@pytest.mark.asyncio
async def test_telemetry_tracker_hybrid_extraction_metadata() -> None:
    """TelemetryTracker records hybrid extraction metrics properly."""
    session = AsyncMock()
    session.add = MagicMock()
    uid = uuid4()

    tracker = TelemetryTracker(
        event_type="llm_extraction_phase1",
        user_id=uid,
        session=session,
        initial_metadata={"cv_id": "cv-456", "classification_confidence": 0.98},
    )

    async with tracker:
        tracker.add_metadata(
            {
                "raw_skills_count": 12,
                "llm_skills_count": 12,
                "direct_catalog_skills_count": 4,
                "hybrid_total_extracted": 16,
                "hybrid_boost_ratio": 0.25,
                "total_skills_phase1": 16,
                "standard_skills_phase1": 14,
                "custom_skills_phase1": 2,
                "standardization_ratio_phase1": 0.875,
            }
        )

    assert tracker.event is not None
    assert tracker.event.event_type == "llm_extraction_phase1"
    meta = tracker.event.metadata_
    assert meta["llm_skills_count"] == 12
    assert meta["direct_catalog_skills_count"] == 4
    assert meta["hybrid_boost_ratio"] == 0.25
    assert meta["standardization_ratio_phase1"] == 0.875


@pytest.mark.asyncio
async def test_telemetry_tracker_skill_normalization() -> None:
    """TelemetryTracker records skill_normalization event and standardization ratio."""
    session = AsyncMock()
    session.add = MagicMock()
    uid = uuid4()

    tracker = TelemetryTracker(
        event_type="skill_normalization",
        user_id=uid,
        session=session,
        initial_metadata={"cv_id": "cv-789", "validated_skills_count": 10},
    )

    async with tracker:
        tracker.add_metadata(
            {
                "total_skills": 10,
                "standard_skills": 8,
                "custom_skills": 2,
                "standardization_ratio": 0.8,
            }
        )

    assert tracker.event is not None
    assert tracker.event.event_type == "skill_normalization"
    assert tracker.event.metadata_["standardization_ratio"] == 0.8
    assert tracker.event.metadata_["standard_skills"] == 8


@pytest.mark.asyncio
async def test_telemetry_tracker_graph_and_affinity() -> None:
    """TelemetryTracker records graph inference, cluster affinity, and gap severities."""
    session = AsyncMock()
    session.add = MagicMock()
    uid = uuid4()

    tracker = TelemetryTracker(
        event_type="graph_and_affinity",
        user_id=uid,
        session=session,
        initial_metadata={"cv_id": "cv-101"},
    )

    async with tracker:
        tracker.add_metadata(
            {
                "total_skills": 12,
                "inferred_skills": 3,
                "inference_ratio": 0.25,
                "primary_cluster": "Backend Python / FastAPI",
                "affinity_score": 0.8421,
                "total_gaps_count": 4,
                "critical_gaps_count": 1,
                "high_gaps_count": 2,
                "medium_gaps_count": 1,
                "seniority": "mid",
            }
        )

    assert tracker.event is not None
    assert tracker.event.event_type == "graph_and_affinity"
    meta = tracker.event.metadata_
    assert meta["inferred_skills"] == 3
    assert meta["inference_ratio"] == 0.25
    assert meta["affinity_score"] == 0.8421
    assert meta["critical_gaps_count"] == 1
    assert meta["high_gaps_count"] == 2
    assert meta["medium_gaps_count"] == 1


@pytest.mark.asyncio
async def test_export_telemetry_to_csv_empty(tmp_path: pytest.TempPathFactory) -> None:
    """export_telemetry_to_csv writes header when no events exist."""
    from scripts.export_telemetry import CSV_FIELDNAMES, export_telemetry_to_csv

    target_csv = tmp_path / "empty_telemetry.csv"  # type: ignore[operator]
    # Patch AsyncSessionLocal to return no events
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    session_mock = AsyncMock()
    session_mock.execute.return_value = result_mock
    session_ctx = AsyncMock()
    session_ctx.__aenter__.return_value = session_mock
    session_ctx.__aexit__.return_value = None

    import scripts.export_telemetry as export_mod

    orig_session = export_mod.AsyncSessionLocal
    export_mod.AsyncSessionLocal = MagicMock(return_value=session_ctx)  # type: ignore[misc]

    try:
        count = await export_telemetry_to_csv(target_csv)
        assert count == 0
        assert target_csv.exists()
        header = target_csv.read_text(encoding="utf-8").strip().split(",")
        assert header == CSV_FIELDNAMES
    finally:
        export_mod.AsyncSessionLocal = orig_session


@pytest.mark.asyncio
async def test_export_telemetry_to_csv_with_data(tmp_path: pytest.TempPathFactory) -> None:
    """export_telemetry_to_csv exports records and parses all fine-grained columns."""
    from scripts.export_telemetry import CSV_FIELDNAMES, export_telemetry_to_csv

    target_csv = tmp_path / "populated_telemetry.csv"  # type: ignore[operator]
    uid = uuid4()
    mock_event = TelemetryEventModel(
        user_id=uid,
        event_type="graph_and_affinity",
        duration_ms=180,
        status="success",
        metadata_={
            "total_skills": 12,
            "inferred_skills": 3,
            "inference_ratio": 0.25,
            "primary_cluster": "Backend Python",
            "affinity_score": 0.85,
            "total_gaps_count": 3,
            "critical_gaps_count": 1,
            "high_gaps_count": 1,
            "medium_gaps_count": 1,
            "seniority": "mid",
        },
    )

    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [mock_event]
    session_mock = AsyncMock()
    session_mock.execute.return_value = result_mock
    session_ctx = AsyncMock()
    session_ctx.__aenter__.return_value = session_mock
    session_ctx.__aexit__.return_value = None

    import scripts.export_telemetry as export_mod

    orig_session = export_mod.AsyncSessionLocal
    export_mod.AsyncSessionLocal = MagicMock(return_value=session_ctx)  # type: ignore[misc]

    try:
        count = await export_telemetry_to_csv(target_csv)
        assert count == 1
        assert target_csv.exists()
        content = target_csv.read_text(encoding="utf-8").strip().splitlines()
        header = content[0].split(",")
        assert header == CSV_FIELDNAMES
        assert "graph_and_affinity" in content[1]
        assert "Backend Python" in content[1]
    finally:
        export_mod.AsyncSessionLocal = orig_session


@pytest.mark.asyncio
async def test_record_telemetry_event_isolated_session_fallback_on_error() -> None:
    """record_telemetry_event falls back to isolated session if provided session fails to commit."""
    broken_session = AsyncMock()
    broken_session.add = MagicMock()
    broken_session.commit.side_effect = RuntimeError("Transaction aborted")
    broken_session.rollback = AsyncMock()

    local_session = AsyncMock()
    local_session.add = MagicMock()
    local_session.commit = AsyncMock()
    session_ctx = AsyncMock()
    session_ctx.__aenter__.return_value = local_session
    session_ctx.__aexit__.return_value = None

    import src.shared.telemetry.tracker as tracker_mod

    orig_session_factory = tracker_mod.AsyncSessionLocal
    tracker_mod.AsyncSessionLocal = MagicMock(return_value=session_ctx)  # type: ignore[misc]

    try:
        event = await record_telemetry_event(
            event_type="test_fallback",
            session=broken_session,
        )
        assert event is not None
        assert event.event_type == "test_fallback"
        broken_session.rollback.assert_awaited_once()
        local_session.add.assert_called_once_with(event)
        local_session.commit.assert_awaited_once()
    finally:
        tracker_mod.AsyncSessionLocal = orig_session_factory


def test_cv_heuristic_non_cv_rejection() -> None:
    """_looks_like_a_cv returns False for obvious homework assignments or invoices."""
    from src.ml_engine.application.use_cases import _looks_like_a_cv

    non_cv_text = (
        "Universidad Nacional de Ingeniería. Ejercicios prácticos de laboratorio de base de datos. "
        "Pregunta 1: Crear índices para la tabla productos. Pregunta 2: Optimizar consultas SQL. "
        "Guía de práctica calificada ciclo 2026-I."
    )
    assert not _looks_like_a_cv(non_cv_text)

    short_text = "Hello world this is a short test"
    assert not _looks_like_a_cv(short_text)

    valid_cv_text = (
        "Jack Arana Ramos - Senior Software Engineer\n"
        "Professional Summary: Experienced backend developer specialized in Python, FastAPI, Postgres.\n"
        "Work Experience:\n"
        "- Software Engineer at Tech Corp (2022 - Present): Designed scalable microservices.\n"
        "Education:\n"
        "- B.S. in Computer Science (2018 - 2022)\n"
        "Technical Skills: Python, FastAPI, Docker, PostgreSQL, Redis, CI/CD."
    )
    assert _looks_like_a_cv(valid_cv_text)


@pytest.mark.asyncio
async def test_combined_llm_extraction_timeout() -> None:
    """_combined_llm_extraction raises MLPipelineError when LLM exceeds timeout."""
    from src.ml_engine.application.use_cases import ProfileUserFromCVUseCase
    from src.shared.exceptions import MLPipelineError

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(side_effect=TimeoutError("Request timed out"))

    use_case = ProfileUserFromCVUseCase(
        cv_parser=MagicMock(),
        cluster_repository=MagicMock(),
        profile_repository=MagicMock(),
        llm_service=mock_llm,
        skill_repository=MagicMock(),
    )

    with pytest.raises(MLPipelineError, match="LLM extraction timed out"):
        await use_case._combined_llm_extraction("Short CV text with enough characters to prompt.")
