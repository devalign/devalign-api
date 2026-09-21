"""Unit tests for GetKnowledgeGraphUseCase."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.ml_engine.application.use_cases import GetKnowledgeGraphUseCase
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    SeniorityLevel,
    Skill,
    SkillGap,
    SkillNature,
    TechCluster,
    UserProfile,
)


@pytest.mark.asyncio
async def test_get_knowledge_graph_user_scopes_market_skills():
    # Arrange
    user_id = uuid4()
    skill_py = Skill(
        id=uuid4(),
        name="Python",
        nature=SkillNature.TECH,
        normalized_name="python",
        domain_tags=["backend", "python"],
    )
    skill_fastapi = Skill(
        id=uuid4(),
        name="FastAPI",
        nature=SkillNature.TECH,
        normalized_name="fastapi",
        domain_tags=["backend", "web"],
    )
    skill_docker = Skill(
        id=uuid4(),
        name="Docker",
        nature=SkillNature.TECH,
        normalized_name="docker",
        domain_tags=["devops", "cloud"],
    )
    skill_redis = Skill(
        id=uuid4(),
        name="Redis",
        nature=SkillNature.TECH,
        normalized_name="redis",
        domain_tags=["backend", "database"],
    )

    gap_fastapi = SkillGap(skill=skill_fastapi, market_importance="high")

    cluster = TechCluster(
        id=uuid4(),
        name="Backend Python",
        description="Python backend",
        centroid_skills=[skill_py, skill_fastapi, skill_redis],
        job_offer_count=100,
        cluster_index=0,
    )

    affinity = ClusterAffinity(
        cluster_id=cluster.id,
        cluster_name="Backend Python",
        affinity_score=0.85,
        is_primary=True,
        detected_skills=[skill_py],
        skill_gaps=[gap_fastapi],
    )

    profile = UserProfile(
        user_id=user_id,
        cv_id=uuid4(),
        embedding=[],
        seniority=SeniorityLevel.MID,
        detected_skills=[skill_py, skill_docker],
        skill_gaps=[gap_fastapi],
        primary_affinity=affinity,
        secondary_affinities=[],
    )

    mock_skill_repo = AsyncMock()
    mock_profile_repo = AsyncMock()
    mock_profile_repo.get_by_user_id.return_value = profile

    mock_cluster_repo = AsyncMock()
    mock_cluster_repo.get_all_active.return_value = [cluster]

    use_case = GetKnowledgeGraphUseCase(
        skill_repository=mock_skill_repo,
        profile_repository=mock_profile_repo,
        cluster_repository=mock_cluster_repo,
    )

    # Act
    result = await use_case.execute(user_id=user_id, cluster_name="Backend Python")

    # Assert
    assert result is not None
    nodes = result.nodes
    statuses = {n.id: n.status for n in nodes}

    # Python is acquired
    assert statuses.get("python") == "acquired"
    # FastAPI is a gap
    assert statuses.get("fastapi") == "gap"
    # Docker is in profile but not in cluster affinity -> neutral
    assert statuses.get("docker") == "neutral"
    # Redis is from cluster centroid -> market
    assert statuses.get("redis") == "market"

    # Verify links exist
    assert len(result.links) > 0


@pytest.mark.asyncio
async def test_get_knowledge_graph_unauthenticated_is_bounded():
    centroid_skills = [
        Skill(
            id=uuid4(),
            name=f"Skill {i}",
            nature=SkillNature.TECH,
            normalized_name=f"skill{i}",
            domain_tags=["domain1"],
        )
        for i in range(100)
    ]

    cluster = TechCluster(
        id=uuid4(),
        name="Fullstack",
        description="Fullstack tech",
        centroid_skills=centroid_skills,
        job_offer_count=50,
        cluster_index=0,
    )

    mock_skill_repo = AsyncMock()
    mock_profile_repo = AsyncMock()
    mock_cluster_repo = AsyncMock()
    mock_cluster_repo.get_all_active.return_value = [cluster]

    use_case = GetKnowledgeGraphUseCase(
        skill_repository=mock_skill_repo,
        profile_repository=mock_profile_repo,
        cluster_repository=mock_cluster_repo,
    )

    # Act
    result = await use_case.execute(user_id=None, cluster_name="Fullstack")

    # Assert
    assert len(result.nodes) <= 60
