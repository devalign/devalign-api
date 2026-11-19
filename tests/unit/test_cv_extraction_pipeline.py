"""Unit tests for the streamlined CV extraction pipeline."""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from src.delivery.interface.router import delete_account, run_profile_analysis_task
from src.ml_engine.application.skill_catalog_service import SkillCatalogService
from src.ml_engine.application.use_cases import ProfileUserFromCVUseCase
from src.ml_engine.domain.entities import Skill, SkillNature
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


@pytest.mark.asyncio
async def test_scan_text_filters_dirty_aliases_and_generic_terms():
    """Catalog scan blocks generic prose nouns, category qualifiers, and colliding aliases."""
    catalog = [
        Skill(name="3M (Software)", nature=SkillNature.TECH, aliases=["3M", "Software"]),
        Skill(name="Bitbucket", nature=SkillNature.TECH, aliases=["github", "bitbucket"]),
        Skill(name="GitHub", nature=SkillNature.TECH, aliases=["github"]),
        Skill(name="Java APIs", nature=SkillNature.TECH, aliases=["apis"]),
        Skill(
            name="Application Programming Interface (API)",
            nature=SkillNature.TECH,
            aliases=["API", "Application Programming Interface"],
        ),
        Skill(name="Developer", nature=SkillNature.TECH, aliases=[]),
        Skill(name="Automation", nature=SkillNature.TECH, aliases=[]),
        Skill(name="Cloud", nature=SkillNature.TECH, aliases=[]),
        Skill(
            name="Chip Multithreading (Server Architecture)",
            nature=SkillNature.TECH,
            aliases=["Chip Multithreading", "Server Architecture"],
        ),
        Skill(name=".NET Development", nature=SkillNature.TECH, aliases=["net"]),
        Skill(name="SOLID", nature=SkillNature.CONCEPT, aliases=["solid"]),
        Skill(name="RESTful API", nature=SkillNature.TECH, aliases=["rest", "rest api"]),
        Skill(name="Python (Programming Language)", nature=SkillNature.TECH, aliases=["python"]),
    ]

    service = SkillCatalogService(skill_repository=MagicMock(), llm_service=MagicMock())
    cv_text = (
        "Senior Software Developer at CloudScale. Built test automation and RESTful APIs "
        "with Python, GitHub, and Server Architecture. Solid experience for the rest of the net."
    )

    results = await service.scan_text_for_catalog_skills(cv_text, existing_skills_cache=catalog)
    found_names = {r["name"] for r in results}

    assert "Python (Programming Language)" in found_names
    assert "GitHub" in found_names
    assert "RESTful API" in found_names

    # Ensure false positives are blocked
    assert "3M (Software)" not in found_names
    assert "Bitbucket" not in found_names
    assert "Java APIs" not in found_names
    assert "Application Programming Interface (API)" not in found_names
    assert "Developer" not in found_names
    assert "Automation" not in found_names
    assert "Cloud" not in found_names
    assert "Chip Multithreading (Server Architecture)" not in found_names
    assert ".NET Development" not in found_names
    assert "SOLID" not in found_names


@pytest.mark.asyncio
async def test_scan_text_resolves_standard_aliases_and_case_sensitive_short_terms():
    """Catalog scan matches AWS/Azure/Maven/OAuth2/EMR and respects uppercase short terms."""
    catalog = [
        Skill(name="Amazon Web Services", nature=SkillNature.TECH, aliases=[]),
        Skill(name="Microsoft Azure", nature=SkillNature.TECH, aliases=[]),
        Skill(name="Apache Maven", nature=SkillNature.TECH, aliases=[]),
        Skill(name="OAuth", nature=SkillNature.TECH, aliases=[]),
        Skill(name="Amazon S3", nature=SkillNature.TECH, aliases=["awss3"]),
        Skill(name="AWS Elastic MapReduce (EMR)", nature=SkillNature.TECH, aliases=["EMR"]),
        Skill(name="HyperText Markup Language (HTML)", nature=SkillNature.TECH, aliases=["HTML"]),
        Skill(name="SOLID", nature=SkillNature.CONCEPT, aliases=["solid"]),
    ]

    service = SkillCatalogService(skill_repository=MagicMock(), llm_service=MagicMock())
    cv_text = "Deployed Java services with Maven, OAuth2, AWS S3, AWS EMR, Azure, HTML, and SOLID principles."

    results = await service.scan_text_for_catalog_skills(cv_text, existing_skills_cache=catalog)
    found_names = {r["name"] for r in results}

    assert {
        "Amazon Web Services",
        "Microsoft Azure",
        "Apache Maven",
        "OAuth",
        "Amazon S3",
        "AWS Elastic MapReduce (EMR)",
        "HyperText Markup Language (HTML)",
        "SOLID",
    }.issubset(found_names)
