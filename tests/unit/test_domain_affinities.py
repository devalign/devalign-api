"""Unit tests for domain affinities calculation in compute_affinities_and_domains."""

from uuid import uuid4

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

    _primary, _secondaries, _affinities, domain_affinities = compute_affinities_and_domains(
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


def test_compute_affinities_and_domains_filters_non_canonical_domains():
    user_skill = Skill(
        id=uuid4(),
        name="Scrum",
        nature=SkillNature.TECH,
        normalized_name="scrum",
        domain_tags=["engineering", "management", "software_engineering", "security"],
        core_domains=[],
        weight=2.0,
        frequency=1.0,
    )

    cluster = TechCluster(
        id=uuid4(),
        name="General Tech",
        description="General tech",
        centroid_skills=[],
        job_offer_count=10,
        cluster_index=0,
    )

    _primary, _secondaries, _affinities, domain_affinities = compute_affinities_and_domains(
        [user_skill], [cluster]
    )

    domains = [d.domain for d in domain_affinities]
    assert "Engineering" not in domains
    assert "Management" not in domains
    assert "Software_engineering" not in domains
    assert "Security" not in domains


def test_compute_affinities_and_domains_normalizes_market_demand_bound():
    centroid_skill = Skill(
        id=uuid4(),
        name="Flutter",
        nature=SkillNature.TECH,
        normalized_name="flutter",
        domain_tags=["mobile"],
        core_domains=["Mobile"],
        weight=2.5,
        frequency=2.37,  # Unnormalized raw importance score
    )

    cluster = TechCluster(
        id=uuid4(),
        name="Mobile Flutter",
        description="Mobile cluster",
        centroid_skills=[centroid_skill],
        job_offer_count=20,
        cluster_index=0,
    )

    _primary, _secondaries, _affinities, domain_affinities = compute_affinities_and_domains(
        [], [cluster]
    )

    mobile_affinity = next((d for d in domain_affinities if d.domain == "Mobile"), None)
    assert mobile_affinity is not None
    assert 0.0 <= mobile_affinity.market_demand <= 1.0
    assert mobile_affinity.market_demand <= 0.98


def test_normalize_demand_percentage_handles_baseline_and_continuous_scales():
    from src.ml_engine.application.use_cases import _normalize_demand_percentage

    # None defaults to 70%
    assert _normalize_demand_percentage(None) == 70

    # True fractions (0.0 < f < 1.0)
    assert _normalize_demand_percentage(0.78) == 78
    assert _normalize_demand_percentage(0.50) == 50

    # Discrete/continuous importance scores (1.0 to 3.0 scale)
    # Baseline 1.0 importance must NOT yield 100%
    assert _normalize_demand_percentage(1.0) == 33
    assert _normalize_demand_percentage(1.5) == 50
    assert _normalize_demand_percentage(2.0) == 67
    assert _normalize_demand_percentage(2.4) == 80
    assert _normalize_demand_percentage(3.0) == 98

