"""Unit tests for skills update logic and compute_domain_affinities."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.ml_engine.application.dtos import SkillDTO, SkillsUpdateDTO
from src.ml_engine.application.use_cases import GetMyProfileUseCase, compute_domain_affinities
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    SeniorityLevel,
    Skill,
    SkillGap,
    SkillNature,
    TechCluster,
    UserProfile,
)
from src.ml_engine.infrastructure.models import DiagnosticModel, ProfileModel, SkillModel
from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository
from src.ml_engine.interface.router import update_my_skills


def test_compute_domain_affinities_calculates_correct_scores():
    # Arrange: User skills with core_domains and weights
    skill_1 = Skill(
        id=uuid4(),
        name="React",
        nature=SkillNature.TECH,
        normalized_name="react",
        core_domains=["Frontend"],
        weight=2.0,
        frequency=0.9,
    )
    skill_2 = Skill(
        id=uuid4(),
        name="Node.js",
        nature=SkillNature.TECH,
        normalized_name="nodejs",
        core_domains=["Backend"],
        weight=3.0,
        frequency=0.8,
    )
    skill_3 = Skill(
        id=uuid4(),
        name="GraphQL",
        nature=SkillNature.TECH,
        normalized_name="graphql",
        core_domains=["Frontend", "Backend"],
        weight=1.5,
        frequency=0.7,
    )

    # Active clusters with core_domains to calculate average market demand
    cluster_skill_1 = Skill(
        id=uuid4(),
        name="React",
        nature=SkillNature.TECH,
        normalized_name="react",
        core_domains=["Frontend"],
        weight=2.0,
        frequency=0.9,
    )
    cluster_skill_2 = Skill(
        id=uuid4(),
        name="FastAPI",
        nature=SkillNature.TECH,
        normalized_name="fastapi",
        core_domains=["Backend"],
        weight=2.5,
        frequency=0.8,
    )

    cluster_1 = TechCluster(
        id=uuid4(),
        name="Frontend React",
        description="React frontend development",
        centroid_skills=[cluster_skill_1],
        job_offer_count=50,
        cluster_index=0,
    )
    cluster_2 = TechCluster(
        id=uuid4(),
        name="Backend Python",
        description="Python backend development",
        centroid_skills=[cluster_skill_2],
        job_offer_count=30,
        cluster_index=1,
    )

    # Act
    domain_affinities = compute_domain_affinities(
        [skill_1, skill_2, skill_3], [cluster_1, cluster_2]
    )

    # Assert
    # Scores:
    # Frontend score = (React) 2.0*0.9 + (GraphQL) 1.5*0.7 = 1.8 + 1.05 = 2.85
    # Backend score = (Node) 3.0*0.8 + (GraphQL) 1.5*0.7 = 2.4 + 1.05 = 3.45
    # Total score = 2.85 + 3.45 = 6.30
    # Frontend affinity = 2.85 / 6.30 ≈ 0.45238
    # Backend affinity = 3.45 / 6.30 ≈ 0.54761

    domain_map = {d.domain: d.affinity_score for d in domain_affinities}
    assert "Frontend" in domain_map
    assert "Backend" in domain_map
    assert pytest.approx(domain_map["Frontend"], abs=1e-4) == 2.85 / 6.30
    assert pytest.approx(domain_map["Backend"], abs=1e-4) == 3.45 / 6.30


@pytest.mark.asyncio
async def test_get_my_profile_refreshes_stale_primary_affinity_and_gaps():
    user_id = uuid4()
    skill_py = Skill(
        id=uuid4(),
        name="Python",
        nature=SkillNature.TECH,
        normalized_name="python",
        core_domains=["Backend"],
        weight=2.0,
        frequency=0.9,
        ict_score=8.0,
    )
    skill_fastapi = Skill(
        id=uuid4(),
        name="FastAPI",
        nature=SkillNature.TECH,
        normalized_name="fastapi",
        core_domains=["Backend"],
        weight=2.5,
        frequency=0.85,
        ict_score=10.0,
    )

    cluster = TechCluster(
        id=uuid4(),
        name="Backend Python",
        description="Python backend",
        centroid_skills=[skill_py, skill_fastapi],
        job_offer_count=50,
        cluster_index=0,
    )

    stale_primary = ClusterAffinity(
        cluster_id=cluster.id,
        cluster_name="Backend Python",
        affinity_score=0.15,
        is_primary=True,
        detected_skills=[skill_py],
        skill_gaps=[SkillGap(skill=skill_fastapi, market_importance="critical")],
    )

    profile = UserProfile(
        user_id=user_id,
        cv_id=uuid4(),
        embedding=[],
        seniority=SeniorityLevel.MID,
        detected_skills=[skill_py, skill_fastapi],
        skill_gaps=[SkillGap(skill=skill_fastapi, market_importance="critical")],
        primary_affinity=stale_primary,
        secondary_affinities=[],
        is_diagnosed=True,
    )

    profile_repo = AsyncMock()
    profile_repo.get_by_user_id.return_value = profile
    cluster_repo = AsyncMock()
    cluster_repo.get_all_active.return_value = [cluster]

    dto = await GetMyProfileUseCase(profile_repo, cluster_repo).execute(user_id)

    assert dto is not None
    assert dto.alignment_score > 0.15
    assert [g.name for g in dto.skill_gaps] == []
    primary_dto = next(a for a in dto.all_affinities if a.is_primary)
    assert primary_dto.affinity_score == dto.alignment_score
    assert {s.name: s.ict_score for s in primary_dto.detected_skills}["FastAPI"] == 10.0
    assert primary_dto.skill_gaps == []


@pytest.mark.asyncio
async def test_update_my_skills_persists_diagnostics_and_ict():
    user_id = uuid4()
    py_id = uuid4()
    docker_id = uuid4()

    skill_py = Skill(
        id=py_id,
        name="Python",
        nature=SkillNature.TECH,
        normalized_name="python",
        core_domains=["Backend"],
        weight=2.0,
        frequency=0.9,
        ict_score=7.5,
    )
    skill_docker = Skill(
        id=docker_id,
        name="Docker",
        nature=SkillNature.TECH,
        normalized_name="docker",
        core_domains=["DevOps"],
        weight=2.5,
        frequency=0.85,
    )

    cluster = TechCluster(
        id=uuid4(),
        name="Backend Python",
        description="Python backend",
        centroid_skills=[skill_py, skill_docker],
        job_offer_count=40,
        cluster_index=0,
    )

    existing_profile = UserProfile(
        user_id=user_id,
        cv_id=uuid4(),
        embedding=[],
        seniority=SeniorityLevel.MID,
        detected_skills=[skill_py],
        skill_gaps=[SkillGap(skill=skill_docker, market_importance="critical")],
        primary_affinity=ClusterAffinity(
            cluster_id=cluster.id,
            cluster_name="Backend Python",
            affinity_score=0.2,
            is_primary=True,
            detected_skills=[skill_py],
            skill_gaps=[SkillGap(skill=skill_docker, market_importance="critical")],
        ),
        secondary_affinities=[],
        is_diagnosed=True,
    )

    db_py = MagicMock(
        skill_id=py_id,
        name="Python",
        nature="tech",
        weight=2.0,
        domain_tags=["backend"],
        core_domains=["Backend"],
        status="canonical",
    )
    db_docker = MagicMock(
        skill_id=docker_id,
        name="Docker",
        nature="tech",
        weight=2.5,
        domain_tags=["devops"],
        core_domains=["DevOps"],
        status="canonical",
    )

    session = AsyncMock()
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = [db_py, db_docker]
    execute_res = MagicMock()
    execute_res.scalars.return_value = scalars_mock
    session.execute.return_value = execute_res

    mock_repo = MagicMock()
    mock_repo.get_by_user_id = AsyncMock(return_value=existing_profile)

    async def _save_profile(prof, *, persist_diagnostics=True, persist_primary_only=False):
        mock_repo.get_by_user_id.return_value = prof
        return prof

    mock_repo.save_profile = AsyncMock(side_effect=_save_profile)

    mock_cluster_repo = MagicMock()
    mock_cluster_repo.get_all_active = AsyncMock(return_value=[cluster])

    payload = SkillsUpdateDTO(
        skills=[
            SkillDTO(name="Python", skill_type="tech", ict_score=7.5),
            SkillDTO(
                name="Docker",
                skill_type="tech",
                years_of_experience=5,
                personal_projects=True,
                has_certification=True,
                ict_score=10.0,
            ),
        ]
    )

    with (
        patch("src.ml_engine.interface.router.SQLUserProfileRepository", return_value=mock_repo),
        patch(
            "src.ml_engine.interface.router.SQLClusterRepository", return_value=mock_cluster_repo
        ),
    ):
        dto = await update_my_skills(
            current_user_id=str(user_id),
            session=session,
            data=payload,
        )

    assert mock_repo.save_profile.await_count == 1
    _, kwargs = mock_repo.save_profile.await_args
    assert kwargs["persist_diagnostics"] is True
    assert kwargs["persist_primary_only"] is True
    assert dto.skill_gaps == []
    assert {s.name: s.ict_score for s in dto.detected_skills}["Docker"] == 10.0


@pytest.mark.asyncio
async def test_sql_user_profile_repository_save_profile_persists_diagnostics_and_deduplicates():
    user_id = uuid4()
    profile_id = uuid4()
    cluster_id = uuid4()
    py_id = uuid4()
    docker_id = uuid4()

    profile_model = ProfileModel(profile_id=profile_id, user_id=user_id)
    canonical_py = SkillModel(
        skill_id=py_id,
        name="Python",
        status="canonical",
        nature="tech",
        weight=2.0,
        domain_tags=["backend"],
        core_domains=["Backend"],
    )
    pending_py = SkillModel(
        skill_id=uuid4(),
        name="python",
        status="pending_review",
        nature="tech",
        weight=1.0,
        domain_tags=[],
        core_domains=[],
    )
    canonical_docker = SkillModel(
        skill_id=docker_id,
        name="Docker",
        status="canonical",
        nature="tech",
        weight=2.5,
        domain_tags=["devops"],
        core_domains=["DevOps"],
    )

    skill_py = Skill(
        id=py_id,
        name="Python",
        nature=SkillNature.TECH,
        normalized_name="python",
        ict_score=10.0,
    )
    skill_py_dup = Skill(
        id=py_id,
        name="python",
        nature=SkillNature.TECH,
        normalized_name="python",
        ict_score=5.0,
    )
    skill_docker = Skill(
        id=docker_id,
        name="Docker",
        nature=SkillNature.TECH,
        normalized_name="docker",
    )

    primary = ClusterAffinity(
        cluster_id=cluster_id,
        cluster_name="Backend Python",
        affinity_score=0.75,
        is_primary=True,
        detected_skills=[skill_py],
        skill_gaps=[SkillGap(skill=skill_docker, market_importance="high")],
    )
    profile = UserProfile(
        user_id=user_id,
        cv_id=uuid4(),
        embedding=[],
        seniority=SeniorityLevel.MID,
        detected_skills=[skill_py, skill_py_dup],
        skill_gaps=[SkillGap(skill=skill_docker, market_importance="high")],
        primary_affinity=primary,
        secondary_affinities=[],
        is_diagnosed=True,
    )

    res_profile = MagicMock()
    res_profile.scalar_one_or_none.return_value = profile_model

    res_skills = MagicMock()
    res_skills.scalars.return_value.all.return_value = [pending_py, canonical_py]

    res_diag_skills = MagicMock()
    res_diag_skills.scalars.return_value.all.return_value = [canonical_py, canonical_docker]

    session = MagicMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            res_profile,  # select ProfileModel
            MagicMock(),  # delete ProfileSkillModel
            res_skills,  # select SkillModel for profile_skills
            MagicMock(),  # delete DiagnosticModel
            res_diag_skills,  # select SkillModel for diagnostic_skills
        ]
    )

    repo = SQLUserProfileRepository(session)
    await repo.save_profile(profile, persist_diagnostics=True, persist_primary_only=True)

    added_types = [type(call.args[0]) for call in session.add.call_args_list]
    assert added_types.count(DiagnosticModel) == 1
    assert session.flush.await_count == 2


@pytest.mark.asyncio
async def test_sql_user_profile_repository_save_single_diagnostic_replaces_existing():
    user_id = uuid4()
    profile_id = uuid4()
    cluster_id = uuid4()
    py_id = uuid4()

    profile_model = ProfileModel(profile_id=profile_id, user_id=user_id)
    canonical_py = SkillModel(
        skill_id=py_id,
        name="Python",
        status="canonical",
        nature="tech",
        weight=2.0,
        domain_tags=["backend"],
        core_domains=["Backend"],
    )
    skill_py = Skill(
        id=py_id,
        name="Python",
        nature=SkillNature.TECH,
        normalized_name="python",
    )
    affinity = ClusterAffinity(
        cluster_id=cluster_id,
        cluster_name="Backend Python",
        affinity_score=0.8,
        is_primary=False,
        detected_skills=[skill_py],
        skill_gaps=[],
    )

    res_profile = MagicMock()
    res_profile.scalar_one_or_none.return_value = profile_model
    res_skills = MagicMock()
    res_skills.scalars.return_value.all.return_value = [canonical_py]

    session = MagicMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock(side_effect=[res_profile, MagicMock(), res_skills])

    repo = SQLUserProfileRepository(session)
    await repo.save_single_diagnostic(user_id, affinity)

    assert session.execute.await_count == 3
    added_types = [type(call.args[0]) for call in session.add.call_args_list]
    assert added_types.count(DiagnosticModel) == 1
