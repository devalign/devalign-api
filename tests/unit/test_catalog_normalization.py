"""Unit tests for Phase 1 pre-normalization and Phase 2 custom skill retention."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.ml_engine.application.skill_catalog_service import SkillCatalogService
from src.ml_engine.domain.entities import Skill, SkillNature


@pytest.fixture
def mock_catalog_skills():
    return [
        Skill(
            id=uuid4(),
            name="MongoDB",
            nature=SkillNature.TECH,
            normalized_name="mongodb",
            core_domains=["Backend", "Data"],
            domain_tags=["database", "nosql"],
            aliases=["mongodb", "mongo db", "nosql (mongodb)"],
        ),
        Skill(
            id=uuid4(),
            name="FastAPI",
            nature=SkillNature.TECH,
            normalized_name="fastapi",
            core_domains=["Backend"],
            domain_tags=["web", "api", "python"],
            aliases=["fastapi", "fast-api"],
        ),
        Skill(
            id=uuid4(),
            name="NoSQL",
            nature=SkillNature.CONCEPT,
            normalized_name="nosql",
            core_domains=["Data", "Backend"],
            domain_tags=["database"],
            aliases=["nosql"],
        ),
    ]


@pytest.mark.asyncio
async def test_normalize_extracted_skills_phase_1(mock_catalog_skills):
    mock_repo = AsyncMock()
    mock_repo.get_all_skills.return_value = mock_catalog_skills
    mock_llm = AsyncMock()

    service = SkillCatalogService(mock_repo, mock_llm)

    raw_extracted = [
        {
            "name": "NoSQL (MongoDB)",
            "category": "technical",
            "years_of_experience": 3,
            "personal_projects": True,
        },
        {
            "name": "FastAPI",
            "category": "technical",
            "years_of_experience": 2,
            "has_certification": True,
        },
        {
            "name": "MCPs",
            "category": "technical",
            "years_of_experience": 1,
        },
    ]

    normalized = await service.normalize_extracted_skills(
        raw_extracted, existing_skills_cache=mock_catalog_skills
    )

    norm_map = {item["name"]: item for item in normalized}

    # 1. MongoDB should be normalized to canonical name and flagged in_catalog
    assert "MongoDB" in norm_map
    assert norm_map["MongoDB"]["in_catalog"] is True
    assert norm_map["MongoDB"]["is_custom"] is False
    assert norm_map["MongoDB"]["original_raw_name"] == "NoSQL (MongoDB)"
    assert norm_map["MongoDB"]["years_of_experience"] == 3

    # 2. FastAPI should be matched in catalog
    assert "FastAPI" in norm_map
    assert norm_map["FastAPI"]["in_catalog"] is True
    assert norm_map["FastAPI"]["is_custom"] is False

    # 3. MCPs should NOT be dropped, but marked as custom
    assert "MCPs" in norm_map
    assert norm_map["MCPs"]["in_catalog"] is False
    assert norm_map["MCPs"]["is_custom"] is True


@pytest.mark.asyncio
async def test_resolve_skills_preserves_custom_skills_without_llm(mock_catalog_skills):
    mock_repo = AsyncMock()
    mock_repo.get_all_skills.return_value = mock_catalog_skills
    mock_llm = AsyncMock()

    service = SkillCatalogService(mock_repo, mock_llm)

    raw_strings = ["MongoDB", "MCPs", "CustomInternalTool"]

    resolved = await service.resolve_skills(
        raw_strings, use_llm_fallback=False, existing_skills_cache=mock_catalog_skills
    )

    names = {s.name: s for s in resolved}

    # MongoDB is canonical
    assert "MongoDB" in names
    assert names["MongoDB"].is_custom is False

    # MCPs is preserved as custom skill
    assert "MCPs" in names
    assert names["MCPs"].is_custom is True
    assert names["MCPs"].id is not None

    # CustomInternalTool is also preserved as custom skill
    assert "CustomInternalTool" in names
    assert names["CustomInternalTool"].is_custom is True


@pytest.mark.asyncio
async def test_normalize_extracted_skills_fuzzy_matching_and_suggestion(mock_catalog_skills):
    mock_repo = AsyncMock()
    mock_repo.get_all_skills.return_value = mock_catalog_skills
    mock_llm = AsyncMock()

    service = SkillCatalogService(mock_repo, mock_llm)

    raw_extracted = [
        # Typos that should auto-match (ratio >= 0.88)
        {"name": "Mongo DBB", "category": "technical"},
        # Related tool name that should suggest canonical (0.78 <= ratio < 0.88)
        {"name": "FastAPIJS", "category": "technical"},
    ]

    normalized = await service.normalize_extracted_skills(
        raw_extracted, existing_skills_cache=mock_catalog_skills
    )

    norm_map = {item["name"]: item for item in normalized}

    # "Mongo DBB" auto-maps to canonical "MongoDB"
    assert "MongoDB" in norm_map
    assert norm_map["MongoDB"]["in_catalog"] is True
    assert norm_map["MongoDB"]["is_custom"] is False

    # "FastAPIJS" stays as custom but suggests "FastAPI"
    assert "FastAPIJS" in norm_map
    assert norm_map["FastAPIJS"]["is_custom"] is True
    assert norm_map["FastAPIJS"]["suggested_canonical"] == "FastAPI"


@pytest.mark.asyncio
async def test_scan_text_for_catalog_skills(mock_catalog_skills):
    mock_repo = AsyncMock()
    mock_repo.get_all_skills.return_value = mock_catalog_skills
    mock_llm = AsyncMock()

    service = SkillCatalogService(mock_repo, mock_llm)

    cv_raw_text = """
    Brittany Chiang
    Experienced engineer.
    Built high-performance applications using MongoDB for document storage and NoSQL design.
    """

    results = await service.scan_text_for_catalog_skills(
        cv_raw_text, existing_skills_cache=mock_catalog_skills
    )

    detected_names = {item["name"] for item in results}

    assert "MongoDB" in detected_names
    assert "NoSQL" in detected_names
    assert "FastAPI" not in detected_names
