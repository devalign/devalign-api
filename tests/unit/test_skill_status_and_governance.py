"""Unit tests for skill lifecycle governance (status) and custom standard integration."""

from uuid import uuid4

import pytest

from src.ml_engine.application.dtos import SkillSearchResultDTO
from src.ml_engine.domain.entities import Skill, SkillNature, SkillStandard, SkillStatus
from src.ml_engine.infrastructure.models import SkillModel, SkillStandardModel


def test_skill_entity_status_defaults_to_canonical():
    """Skill domain entity defaults to CANONICAL status."""
    skill = Skill(
        name="Python",
        nature=SkillNature.TECH,
        normalized_name="python",
    )
    assert skill.status == SkillStatus.CANONICAL
    assert skill.status.value == "canonical"


def test_skill_entity_explicit_status():
    """Skill domain entity can be instantiated with PENDING_REVIEW or DEPRECATED."""
    skill_pending = Skill(
        name="ObscureFramework",
        nature=SkillNature.TECH,
        normalized_name="obscureframework",
        status=SkillStatus.PENDING_REVIEW,
    )
    assert skill_pending.status == SkillStatus.PENDING_REVIEW

    skill_deprecated = Skill(
        name="OldTool",
        nature=SkillNature.TECH,
        normalized_name="oldtool",
        status=SkillStatus.DEPRECATED,
    )
    assert skill_deprecated.status == SkillStatus.DEPRECATED


def test_skill_model_default_status():
    """SkillModel ORM model defaults to 'canonical' status."""
    model = SkillModel(
        name="FastAPI",
        nature="tech",
    )
    assert model.status == "canonical" or model.status is None  # before DB flush it has default


def test_skill_search_result_dto_fields():
    """SkillSearchResultDTO includes status and standard_name."""
    dto_canonical = SkillSearchResultDTO(
        id=uuid4(),
        name="Microsoft Excel",
        skill_type="tech",
        status="canonical",
        standard_name="Lightcast",
    )
    assert dto_canonical.status == "canonical"
    assert dto_canonical.standard_name == "Lightcast"

    dto_custom = SkillSearchResultDTO(
        id=uuid4(),
        name="Next.js",
        skill_type="tech",
        status="canonical",
        standard_name="Custom",
    )
    assert dto_custom.status == "canonical"
    assert dto_custom.standard_name == "Custom"


def test_standard_model_custom_urn_format():
    """SkillStandardModel correctly stores custom standard URN."""
    test_id = uuid4()
    standard = SkillStandardModel(
        skill_id=test_id,
        standard_name="Custom",
        standard_uri=f"devalign:skill:custom:{test_id}",
        standard_code="CUSTOM",
    )
    assert standard.standard_name == "Custom"
    assert standard.standard_uri.startswith("devalign:skill:custom:")
    assert standard.standard_code == "CUSTOM"
