"""Unit tests for domain affinities calculation in compute_affinities_and_domains."""

from uuid import uuid4
import pytest

from src.ml_engine.application.use_cases import compute_affinities_and_domains
from src.ml_engine.domain.entities import Skill, SkillNature, TechCluster


def test_compute_affinities_and_domains_uses_core_domains():
    # Arrange: Create user skills with domain_tags and core_domains
    user_skill_1 = Skill(
        id=uuid4(),
        name="React",
        nature=SkillNature.TECH,
        normalized_name="react",
        domain_tags=["react", "web"],
        core_domains=["Frontend"],
        weight=2.0,
        frequency=0.9,
    )
    user_skill_2 = Skill(
        id=uuid4(),
        name="Python",
        nature=SkillNature.TECH,
        normalized_name="python",
        domain_tags=["python", "scripting"],
        core_domains=["Backend", "Data"],
        weight=3.0,
        frequency=0.8,
    )
    
    # Centroid skills for a cluster
    centroid_skill = Skill(
        id=uuid4(),
        name="FastAPI",
        nature=SkillNature.TECH,
        normalized_name="fastapi",
        domain_tags=["fastapi", "web"],
        core_domains=["Backend"],
        weight=2.5,
        frequency=0.85,
    )
    
    cluster = TechCluster(
        id=uuid4(),
        name="Backend Python",
        description="Python backend development",
        centroid_skills=[centroid_skill],
        job_offer_count=50,
        cluster_index=0,
    )
    
    # Act: Run calculation
    detected_skills = [user_skill_1, user_skill_2]
    active_clusters = [cluster]
    
    primary, secondaries, affinities, domain_affinities = compute_affinities_and_domains(
        detected_skills, active_clusters
    )
    
    # Assert: Verify domain affinities are calculated based on core_domains
    # User has: Frontend (weight 2.0, freq 0.9 -> score 1.8), Backend (weight 3.0, freq 0.8 -> score 2.4), Data (weight 3.0, freq 0.8 -> score 2.4)
    # Total score = 1.8 (Frontend) + 2.4 (Backend) + 2.4 (Data) = 6.6
    # Frontend affinity = 1.8 / 6.6 ≈ 0.2727
    # Backend affinity = 2.4 / 6.6 ≈ 0.3636
    # Data affinity = 2.4 / 6.6 ≈ 0.3636
    
    domain_map = {d.domain: d.affinity_score for d in domain_affinities}
    
    assert "Frontend" in domain_map
    assert "Backend" in domain_map
    assert "Data" in domain_map
    assert "react" not in domain_map  # Specific tags should not be in core domains
    assert "web" not in domain_map
    
    assert round(domain_map["Frontend"], 4) == 0.2727
    assert round(domain_map["Backend"], 4) == 0.3636
    assert round(domain_map["Data"], 4) == 0.3636
