"""Unit tests for skill catalog hardening, alias indexing, and duplicate prevention."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.ml_engine.application.skill_catalog_service import (
    build_skill_lookup_indexes,
    match_single_skill,
)
from src.ml_engine.application.use_cases import NormalizeSkillsUseCase
from src.ml_engine.domain.entities import Skill, SkillNature


def test_build_skill_lookup_indexes_and_matching():
    """Verify that Lightcast canonical skills with parenthetical names and aliases match raw skills."""
    python_canonical = Skill(
        id=uuid4(),
        name="Python (Programming Language)",
        nature=SkillNature.TECH,
        normalized_name="python(programminglanguage)",
        aliases=["python", "py", "python3"],
    )
    docker_canonical = Skill(
        id=uuid4(),
        name="Docker (Software)",
        nature=SkillNature.TECH,
        normalized_name="docker(software)",
        aliases=["docker"],
    )
    sql_canonical = Skill(
        id=uuid4(),
        name="SQL (Programming Language)",
        nature=SkillNature.TECH,
        normalized_name="sql(programminglanguage)",
        aliases=["sql", "structured query language"],
    )

    catalog = [python_canonical, docker_canonical, sql_canonical]
    alias_to_skill, norm_to_skill = build_skill_lookup_indexes(catalog)

    # 1. Simple clean names
    assert match_single_skill("Python", alias_to_skill, norm_to_skill) == python_canonical
    assert match_single_skill("python", alias_to_skill, norm_to_skill) == python_canonical
    assert match_single_skill("Docker", alias_to_skill, norm_to_skill) == docker_canonical
    assert match_single_skill("SQL", alias_to_skill, norm_to_skill) == sql_canonical

    # 2. ESCO-style parenthesized raw skill
    assert (
        match_single_skill("python (programación informática)", alias_to_skill, norm_to_skill)
        == python_canonical
    )

    # 3. Conversational prefixes
    assert (
        match_single_skill("lenguaje Python", alias_to_skill, norm_to_skill)
        == python_canonical
    )
    assert (
        match_single_skill("herramienta Docker", alias_to_skill, norm_to_skill)
        == docker_canonical
    )

    # 4. Hallucinations and invalid inputs are rejected
    assert (
        match_single_skill("Python (no mencionado en la vacante)", alias_to_skill, norm_to_skill)
        is None
    )
    assert (
        match_single_skill("AWS (not mentioned)", alias_to_skill, norm_to_skill)
        is None
    )
    assert match_single_skill("a" * 100, alias_to_skill, norm_to_skill) is None
    assert match_single_skill("", alias_to_skill, norm_to_skill) is None


@pytest.mark.asyncio
async def test_normalize_skills_use_case_prevents_duplicates():
    """Verify NormalizeSkillsUseCase maps raw skills to existing Lightcast skills without creating duplicates."""
    python_canonical = Skill(
        id=uuid4(),
        name="Python (Programming Language)",
        nature=SkillNature.TECH,
        normalized_name="python(programminglanguage)",
        aliases=["python", "py"],
    )
    sql_canonical = Skill(
        id=uuid4(),
        name="SQL (Programming Language)",
        nature=SkillNature.TECH,
        normalized_name="sql(programminglanguage)",
        aliases=["sql"],
    )

    job_offer_id = uuid4()
    mock_job_offers = MagicMock()
    mock_job_offers.get_unnormalized_offers = AsyncMock(
        return_value=[
            {
                "id": job_offer_id,
                # Both 'Python' and ESCO-style 'python (programación informática)' are in the offer
                "raw_hard_skills": ["Python", "python (programación informática)", "SQL"],
            }
        ]
    )
    mock_job_offers.save_offer_skills = AsyncMock()
    mock_job_offers.mark_as_normalized = AsyncMock()

    mock_skills = MagicMock()
    mock_skills.get_all_skills = AsyncMock(return_value=[python_canonical, sql_canonical])
    mock_skills.save_skills = AsyncMock()

    mock_embeddings = MagicMock()
    mock_embeddings.embed_batch = AsyncMock()

    use_case = NormalizeSkillsUseCase(
        job_offer_repo=mock_job_offers,
        skill_repo=mock_skills,
        embedding_service=mock_embeddings,
        llm_service=None,
    )

    result = await use_case.execute()

    # Verify zero new skills were created
    assert result["new_skills"] == 0
    mock_skills.save_skills.assert_not_called()

    # Verify that save_offer_skills was called with exactly 2 links (Python deduplicated + SQL)
    mock_job_offers.save_offer_skills.assert_called_once()
    saved_links = mock_job_offers.save_offer_skills.call_args[0][0]
    assert len(saved_links) == 2

    skill_ids = [link["skill_id"] for link in saved_links]
    assert python_canonical.id in skill_ids
    assert sql_canonical.id in skill_ids

    # Verify offer marked as normalized
    mock_job_offers.mark_as_normalized.assert_called_once_with([job_offer_id])
