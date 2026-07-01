"""Unit tests for skills update logic and compute_domain_affinities."""

from uuid import uuid4

import pytest

from src.ml_engine.application.use_cases import compute_domain_affinities
from src.ml_engine.domain.entities import Skill, SkillNature, TechCluster


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
    assert pytest.approx(domain_map["Frontend"]) == 2.85 / 6.30
    assert pytest.approx(domain_map["Backend"]) == 3.45 / 6.30
