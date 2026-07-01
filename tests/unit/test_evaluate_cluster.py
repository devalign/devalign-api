"""Unit tests for EvaluateClusterDiagnosticUseCase."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.ml_engine.application.use_cases import EvaluateClusterDiagnosticUseCase
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    SeniorityLevel,
    Skill,
    SkillNature,
    TechCluster,
    UserProfile,
)


@pytest.mark.asyncio
async def test_evaluate_cluster_diagnostic_use_case_success():
    # Arrange
    user_id = uuid4()
    cluster_name = "Backend Cloud-Native Java"
    cluster_id = uuid4()

    # Mock user profile
    mock_profile = UserProfile(
        user_id=user_id,
        cv_id=uuid4(),
        embedding=[0.0] * 1024,
        detected_skills=[
            Skill(
                id=uuid4(),
                name="Java",
                nature=SkillNature.TECH,
                normalized_name="java",
                weight=3.0,
                frequency=1.0,
                ict_score=10.0,
            )
        ],
        seniority=SeniorityLevel.MID,
        primary_affinity=ClusterAffinity(
            cluster_id=uuid4(),
            cluster_name="Frontend React",
            affinity_score=0.5,
            is_primary=True,
        ),
        secondary_affinities=[]
    )

    # Mock cluster
    mock_cluster = TechCluster(
        id=cluster_id,
        name=cluster_name,
        description="Java Backend",
        centroid_skills=[
            Skill(
                id=uuid4(),
                name="Java",
                nature=SkillNature.TECH,
                normalized_name="java",
                weight=3.0,
                frequency=1.0,
            )
        ],
        job_offer_count=10,
        cluster_index=0,
    )

    # Mock repositories
    profile_repo = MagicMock()
    profile_repo.get_by_user_id = AsyncMock(return_value=mock_profile)
    profile_repo.save = AsyncMock(return_value=mock_profile)

    cluster_repo = MagicMock()
    cluster_repo.get_all_active = AsyncMock(return_value=[mock_cluster])

    use_case = EvaluateClusterDiagnosticUseCase(profile_repo, cluster_repo)

    # Act
    dto = await use_case.execute(user_id, cluster_name)

    # Assert
    assert dto is not None
    assert profile_repo.get_by_user_id.call_count == 2
    assert cluster_repo.get_all_active.call_count == 2

    # Verify save was called with a profile containing the new secondary affinity
    saved_profile = profile_repo.save.call_args[0][0]
    assert len(saved_profile.secondary_affinities) == 1
    assert saved_profile.secondary_affinities[0].cluster_name == cluster_name
    assert saved_profile.secondary_affinities[0].affinity_score == 1.0  # Since user has Java and cluster has Java
