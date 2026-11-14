"""Unit tests for the streamlined CV extraction pipeline."""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from src.delivery.interface.router import delete_account, run_profile_analysis_task
from src.ml_engine.application.use_cases import ProfileUserFromCVUseCase
from src.shared.exceptions import MLPipelineError


@pytest.fixture
def mock_dependencies():
    cv_parser = MagicMock()
    cluster_repo = MagicMock()
    profile_repo = MagicMock()
    llm_service = MagicMock()
    skill_repo = MagicMock()
    skill_catalog = MagicMock()

    use_case = ProfileUserFromCVUseCase(
        cv_parser=cv_parser,
        cluster_repository=cluster_repo,
        profile_repository=profile_repo,
        llm_service=llm_service,
        skill_repository=skill_repo,
        skill_catalog=skill_catalog,
    )
    return use_case, llm_service


@pytest.mark.asyncio
async def test_extract_llm_insights_profile_metadata_success(mock_dependencies):
    """extract_llm_insights synthesizes role, years, and summary without generating skills."""
    use_case, llm_service = mock_dependencies

    llm_payload = {
        "current_job_role": "Senior Fullstack Engineer",
        "professional_summary": "Experienced engineer specializing in React and Python microservices.",
        "years_experience": 6,
    }
    llm_service.generate = AsyncMock(return_value=json.dumps(llm_payload))

    result = await use_case.extract_llm_insights(
        user_id=uuid4(),
        cv_id=uuid4(),
        cv_text="Extensive CV text describing professional achievements...",
    )

    assert result["current_job_role"] == "Senior Fullstack Engineer"
    assert result["years_experience"] == 6
    assert (
        result["professional_summary"]
        == "Experienced engineer specializing in React and Python microservices."
    )
    assert "skills" not in result

    # Verify LLM was called with max_tokens=300 for fast synthesis
    llm_service.generate.assert_awaited_once()
    _, kwargs = llm_service.generate.call_args
    assert kwargs.get("max_tokens") == 300


@pytest.mark.asyncio
async def test_extract_llm_insights_rejects_non_cv(mock_dependencies):
    """extract_llm_insights raises MLPipelineError when LLM classifies document as non-CV."""
    use_case, llm_service = mock_dependencies

    llm_payload = {"error": "not_a_cv", "document_type": "invoice"}
    llm_service.generate = AsyncMock(return_value=json.dumps(llm_payload))

    with pytest.raises(MLPipelineError, match="does not appear to be a CV/resume"):
        await use_case.extract_llm_insights(
            user_id=uuid4(),
            cv_id=uuid4(),
            cv_text="Invoice #12345 for services rendered.",
        )


@pytest.mark.asyncio
async def test_extract_llm_insights_timeout_15s(mock_dependencies):
    """extract_llm_insights raises MLPipelineError with timeout message after 15 seconds."""
    use_case, llm_service = mock_dependencies

    llm_service.generate = AsyncMock(side_effect=TimeoutError("Timed out"))

    with pytest.raises(MLPipelineError, match="LLM extraction timed out after 15s"):
        await use_case.extract_llm_insights(
            user_id=uuid4(),
            cv_id=uuid4(),
            cv_text="Some CV text...",
        )


@pytest.mark.asyncio
async def test_run_profile_analysis_task_success(monkeypatch):
    """run_profile_analysis_task coordinates deterministic scan and LLM profile synthesis."""
    mock_session = AsyncMock()
    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr("src.shared.database.AsyncSessionLocal", lambda: mock_session_ctx)

    mock_cv_repo = MagicMock()
    mock_cv_repo.update_status = AsyncMock(return_value=1)
    mock_cv_repo.update_extracted_data = AsyncMock(return_value=1)
    monkeypatch.setattr(
        "src.delivery.interface.router.SQLAlchemyCVRepository", lambda session: mock_cv_repo
    )

    mock_use_case = MagicMock()
    mock_use_case.extract_deterministic_data = AsyncMock(
        return_value={
            "cv_text": "Sample CV Text",
            "extracted_data": {
                "skills": [{"name": "FastAPI", "source": "exact"}],
            },
        }
    )
    mock_use_case.extract_llm_insights = AsyncMock(
        return_value={
            "current_job_role": "Backend Engineer",
            "years_experience": 4,
            "professional_summary": "Python backend engineer.",
        }
    )
    monkeypatch.setattr(
        "src.ml_engine.application.use_cases.ProfileUserFromCVUseCase",
        lambda **kwargs: mock_use_case,
    )

    user_id = uuid4()
    cv_id = uuid4()
    await run_profile_analysis_task(
        user_id=user_id,
        cv_id=cv_id,
        content=b"%PDF-sample",
        content_type="application/pdf",
    )

    mock_cv_repo.update_status.assert_any_call(cv_id, "processing")
    mock_cv_repo.update_status.assert_any_call(cv_id, "skills_detected_partial")
    mock_cv_repo.update_status.assert_any_call(cv_id, "skills_detected")
    mock_cv_repo.update_extracted_data.assert_any_call(
        cv_id,
        {
            "skills": [{"name": "FastAPI", "source": "exact"}],
            "current_job_role": "Backend Engineer",
            "years_experience": 4,
            "professional_summary": "Python backend engineer.",
        },
    )


@pytest.mark.asyncio
async def test_delete_account_endpoint(monkeypatch):
    """delete_account offloads supabase admin deletion to thread without blocking event loop."""
    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock()
    monkeypatch.setattr(
        "src.delivery.interface.router.ResetAccountUseCase", lambda **kwargs: mock_use_case
    )

    mock_admin = MagicMock()
    mock_admin.auth.admin.delete_user = MagicMock()
    monkeypatch.setattr(
        "src.delivery.interface.router.get_supabase_admin_client", lambda: mock_admin
    )

    mock_session = AsyncMock()
    user_id = str(uuid4())
    await delete_account(current_user_id=user_id, session=mock_session)

    mock_use_case.execute.assert_awaited_once_with(UUID(user_id))
    mock_admin.auth.admin.delete_user.assert_called_once_with(user_id)
