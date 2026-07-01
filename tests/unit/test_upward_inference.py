"""Unit tests for the upward skill inference algorithm in ProfileUserFromCVUseCase.

Covers:
- Multi-level BELONGS_TO traversal (PostgreSQL → SQL → RDBMS)
- REQUIRES relation traversal (Angular → TypeScript → JavaScript)
- Cycle prevention (A ↔ B graph must terminate without infinite loop)
- inferred_from provenance metadata is stamped on inferred skills
- ALTERNATIVE_TO edges are NOT traversed (horizontal, not upward)
- Empty input returns empty list
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.ml_engine.application.use_cases import ProfileUserFromCVUseCase
from src.ml_engine.domain.entities import Skill, SkillNature, SkillRelation, SkillRelationType


def _make_mock_repo(skills: list[Skill]) -> MagicMock:
    """Build a mock SkillRepository that returns the given skills as a graph dict."""
    skill_graph = {s.id: s for s in skills if s.id}
    mock = MagicMock()
    mock.get_skill_graph = AsyncMock(return_value=skill_graph)
    return mock


def _make_use_case(skills: list[Skill]) -> ProfileUserFromCVUseCase:
    """Build a ProfileUserFromCVUseCase with all dependencies mocked except the skill repo."""
    return ProfileUserFromCVUseCase(
        cv_parser=MagicMock(),
        cluster_repository=MagicMock(),
        profile_repository=MagicMock(),
        llm_service=MagicMock(),
        skill_repository=_make_mock_repo(skills),
    )


# ── Helpers to build skill fixtures ─────────────────────────────────────────


def _belongs_to(target_id, target_name: str) -> SkillRelation:
    return SkillRelation(
        target_skill_id=target_id,
        target_skill_name=target_name,
        relation_type=SkillRelationType.BELONGS_TO,
    )


def _requires(target_id, target_name: str) -> SkillRelation:
    return SkillRelation(
        target_skill_id=target_id,
        target_skill_name=target_name,
        relation_type=SkillRelationType.REQUIRES,
    )


def _alternative_to(target_id, target_name: str) -> SkillRelation:
    return SkillRelation(
        target_skill_id=target_id,
        target_skill_name=target_name,
        relation_type=SkillRelationType.ALTERNATIVE_TO,
    )


def _tech(name: str, normalized: str, relations: list[SkillRelation] | None = None) -> Skill:
    return Skill(
        id=uuid4(),
        name=name,
        nature=SkillNature.TECH,
        normalized_name=normalized,
        relations=relations or [],
    )


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_belongs_to_chain_infers_parents_recursively() -> None:
    """PostgreSQL → SQL → RDBMS: all ancestors must be inferred."""
    rdbms = _tech("RDBMS", "rdbms")
    sql = _tech("SQL", "sql", [_belongs_to(rdbms.id, rdbms.name)])
    postgres = _tech("PostgreSQL", "postgresql", [_belongs_to(sql.id, sql.name)])

    use_case = _make_use_case([rdbms, sql, postgres])
    result = await use_case._expand_with_upward_inference([postgres])

    names = {s.name for s in result}
    assert names == {"PostgreSQL", "SQL", "RDBMS"}
    assert len(result) == 3


@pytest.mark.asyncio
async def test_requires_relation_infers_parent() -> None:
    """Angular requires TypeScript which requires JavaScript — all should be inferred."""
    js = _tech("JavaScript", "javascript")
    ts = _tech("TypeScript", "typescript", [_requires(js.id, js.name)])
    angular = _tech(
        "Angular",
        "angular",
        [
            _requires(ts.id, ts.name),
            _requires(js.id, js.name),
        ],
    )

    use_case = _make_use_case([js, ts, angular])
    result = await use_case._expand_with_upward_inference([angular])

    names = {s.name for s in result}
    assert "Angular" in names
    assert "TypeScript" in names
    assert "JavaScript" in names


@pytest.mark.asyncio
async def test_inferred_skill_has_inferred_from_provenance() -> None:
    """Inferred parent skills must carry the child's name in inferred_from."""
    sql = _tech("SQL", "sql")
    postgres = _tech("PostgreSQL", "postgresql", [_belongs_to(sql.id, sql.name)])

    use_case = _make_use_case([sql, postgres])
    result = await use_case._expand_with_upward_inference([postgres])

    sql_inferred = next((s for s in result if s.name == "SQL"), None)
    assert sql_inferred is not None, "SQL was not inferred"
    assert "PostgreSQL" in sql_inferred.inferred_from, (
        f"Expected 'PostgreSQL' in inferred_from, got: {sql_inferred.inferred_from}"
    )


@pytest.mark.asyncio
async def test_explicitly_detected_skills_have_no_inferred_from() -> None:
    """Skills explicitly extracted from the CV must NOT be stamped with inferred_from."""
    sql = _tech("SQL", "sql")
    postgres = _tech("PostgreSQL", "postgresql", [_belongs_to(sql.id, sql.name)])

    use_case = _make_use_case([sql, postgres])
    result = await use_case._expand_with_upward_inference([postgres])

    pg_skill = next(s for s in result if s.name == "PostgreSQL")
    assert pg_skill.inferred_from == [], "Explicitly detected skill should have empty inferred_from"


@pytest.mark.asyncio
async def test_alternative_to_is_not_traversed() -> None:
    """ALTERNATIVE_TO edges must NOT trigger upward inference."""
    vue = _tech("Vue.js", "vuejs")
    react = _tech("React", "react", [_alternative_to(vue.id, vue.name)])

    use_case = _make_use_case([react, vue])
    result = await use_case._expand_with_upward_inference([react])

    names = {s.name for s in result}
    assert "React" in names
    assert "Vue.js" not in names, "ALTERNATIVE_TO should not trigger inference"


@pytest.mark.asyncio
async def test_cycle_between_two_skills_terminates() -> None:
    """A ↔ B (both belong to each other) must terminate without infinite loop."""
    id_a, id_b = uuid4(), uuid4()
    skill_a = Skill(
        id=id_a,
        name="A",
        nature=SkillNature.TECH,
        normalized_name="a",
        relations=[_belongs_to(id_b, "B")],
    )
    skill_b = Skill(
        id=id_b,
        name="B",
        nature=SkillNature.TECH,
        normalized_name="b",
        relations=[_belongs_to(id_a, "A")],
    )

    use_case = _make_use_case([skill_a, skill_b])
    result = await use_case._expand_with_upward_inference([skill_a])

    names = {s.name for s in result}
    assert names == {"A", "B"}
    assert len(result) == 2


@pytest.mark.asyncio
async def test_empty_input_returns_empty_list() -> None:
    """Calling inference with no skills must return an empty list immediately."""
    use_case = _make_use_case([])
    result = await use_case._expand_with_upward_inference([])
    assert result == []


@pytest.mark.asyncio
async def test_multiple_children_share_parent() -> None:
    """MySQL and PostgreSQL both point to SQL — SQL should appear once."""
    sql = _tech("SQL", "sql")
    postgres = _tech("PostgreSQL", "postgresql", [_belongs_to(sql.id, sql.name)])
    mysql = _tech("MySQL", "mysql", [_belongs_to(sql.id, sql.name)])

    use_case = _make_use_case([sql, postgres, mysql])
    result = await use_case._expand_with_upward_inference([postgres, mysql])

    names = {s.name for s in result}
    assert "SQL" in names
    # SQL must appear exactly once (deduplication by ID)
    sql_count = sum(1 for s in result if s.name == "SQL")
    assert sql_count == 1, f"SQL appeared {sql_count} times — should be exactly 1"


@pytest.mark.asyncio
async def test_skill_not_in_graph_is_skipped_gracefully() -> None:
    """Skills without a DB-assigned id (catalog misses) don't crash the traversal."""
    sql = _tech("SQL", "sql")
    # Simulate a skill without id that came from LLM but isn't in catalog
    unknown = Skill(
        id=None,
        name="UnknownLib",
        nature=SkillNature.TECH,
        normalized_name="unknownlib",
    )

    use_case = _make_use_case([sql])
    result = await use_case._expand_with_upward_inference([unknown])

    # Unknown skill has no id so it cannot be traversed — just returned as-is
    assert len(result) == 0  # Skipped because id is None
