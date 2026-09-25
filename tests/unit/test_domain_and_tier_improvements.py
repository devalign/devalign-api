"""Tests for domain routing, tier filtering, and blacklist improvements."""

from uuid import uuid4

from src.ml_engine.application.skill_catalog_service import PROFILE_SKILL_BLACKLIST
from src.ml_engine.application.use_cases import (
    _build_combined_cv_extraction_prompt,
    _filter_affinities_by_domain,
    _get_cluster_primary_domain,
)
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    Skill,
    SkillNature,
    TechCluster,
)


def _make_skill(
    name: str, core_domains: list[str], weight: float = 1.0, frequency: float = 1.0
) -> Skill:
    return Skill(
        id=uuid4(),
        name=name,
        nature=SkillNature.TECH,
        normalized_name=name.lower().replace(" ", "").replace(".", ""),
        weight=weight,
        frequency=frequency,
        core_domains=core_domains,
        domain_tags=core_domains,
    )


def _make_cluster(name: str, skills: list[Skill], tier: str = "standard") -> TechCluster:
    return TechCluster(
        id=uuid4(),
        name=name,
        description=f"Description for {name}",
        centroid_skills=skills,
        job_offer_count=50,
        cluster_index=0,
        tier=tier,
    )


def test_blacklist_contains_critical_abstractions():
    """Verify abstract meta-skills are properly blacklisted."""
    assert "software configuration management" in PROFILE_SKILL_BLACKLIST
    assert "devops" in PROFILE_SKILL_BLACKLIST
    assert "software engineering" in PROFILE_SKILL_BLACKLIST
    assert "full stack development" in PROFILE_SKILL_BLACKLIST


def test_cv_extraction_prompt_prohibits_abstractions():
    """Verify prompt explicitly instructs not to extract abstract categories."""
    prompt = _build_combined_cv_extraction_prompt("Sample CV text")
    assert "Software Configuration Management" in prompt
    assert "DO NOT extract" in prompt
    assert "IMPLICIT SKILL DETECTION" in prompt


def test_get_cluster_primary_domain():
    """Verify primary domain resolution from weighted centroid skills."""
    fe_cluster = _make_cluster(
        "Frontend React",
        [
            _make_skill("React.js", ["Frontend"], weight=3.0, frequency=0.9),
            _make_skill("Next.js", ["Frontend"], weight=2.8, frequency=0.8),
            _make_skill("Docker", ["DevOps"], weight=1.0, frequency=0.2),
        ],
    )
    assert _get_cluster_primary_domain(fe_cluster) == "Frontend"

    devops_cluster = _make_cluster(
        "DevOps Cloud",
        [
            _make_skill("Kubernetes", ["DevOps"], weight=3.0, frequency=0.9),
            _make_skill("Terraform", ["Cloud", "DevOps"], weight=2.5, frequency=0.8),
            _make_skill("Python", ["Backend"], weight=1.0, frequency=0.2),
        ],
    )
    assert _get_cluster_primary_domain(devops_cluster) in ("DevOps", "Cloud")


def test_domain_filter_blocks_devops_under_threshold():
    """A frontend-dominant user (61% FE, 29% DevOps) must NOT see DevOps cluster."""
    fe_cluster = _make_cluster(
        "Desarrollador Frontend Web",
        [_make_skill("React.js", ["Frontend"], weight=3.0)],
    )
    devops_cluster = _make_cluster(
        "Ingeniero DevOps",
        [_make_skill("Kubernetes", ["DevOps"], weight=3.0)],
    )

    affinities = [
        ClusterAffinity(
            cluster_id=fe_cluster.id, cluster_name=fe_cluster.name, affinity_score=0.87
        ),
        ClusterAffinity(
            cluster_id=devops_cluster.id, cluster_name=devops_cluster.name, affinity_score=0.81
        ),
    ]

    domain_scores = {
        "Frontend": 61.0,
        "Backend": 37.0,
        "DevOps": 29.0,  # 29 / 127 = 22.8% (< 35% threshold)
    }

    filtered = _filter_affinities_by_domain(
        affinities,
        [fe_cluster, devops_cluster],
        domain_scores,
    )

    cluster_names = [a.cluster_name for a in filtered]
    assert "Desarrollador Frontend Web" in cluster_names
    assert "Ingeniero DevOps" not in cluster_names


def test_domain_filter_allows_devops_above_threshold():
    """When a user has strong cross-domain skills (>= 35%), the non-adjacent domain is permitted."""
    fe_cluster = _make_cluster(
        "Desarrollador Frontend Web",
        [_make_skill("React.js", ["Frontend"], weight=3.0)],
    )
    devops_cluster = _make_cluster(
        "Ingeniero DevOps",
        [_make_skill("Kubernetes", ["DevOps"], weight=3.0)],
    )

    affinities = [
        ClusterAffinity(
            cluster_id=fe_cluster.id, cluster_name=fe_cluster.name, affinity_score=0.87
        ),
        ClusterAffinity(
            cluster_id=devops_cluster.id, cluster_name=devops_cluster.name, affinity_score=0.81
        ),
    ]

    domain_scores = {
        "Frontend": 50.0,
        "DevOps": 40.0,  # 40 / 90 = 44.4% (>= 35% threshold)
    }

    filtered = _filter_affinities_by_domain(
        affinities,
        [fe_cluster, devops_cluster],
        domain_scores,
    )

    cluster_names = [a.cluster_name for a in filtered]
    assert "Desarrollador Frontend Web" in cluster_names
    assert "Ingeniero DevOps" in cluster_names


def test_domain_filter_allows_adjacent_domains():
    """Frontend users naturally allow Mobile (adjacent) even if Mobile radar is low."""
    fe_cluster = _make_cluster(
        "Desarrollador Frontend Web",
        [_make_skill("React.js", ["Frontend"], weight=3.0)],
    )
    mobile_cluster = _make_cluster(
        "Desarrollador Mobile Flutter",
        [_make_skill("Flutter", ["Mobile"], weight=3.0)],
    )

    affinities = [
        ClusterAffinity(
            cluster_id=fe_cluster.id, cluster_name=fe_cluster.name, affinity_score=0.85
        ),
        ClusterAffinity(
            cluster_id=mobile_cluster.id, cluster_name=mobile_cluster.name, affinity_score=0.70
        ),
    ]

    domain_scores = {
        "Frontend": 80.0,
        "Mobile": 10.0,  # 10 / 90 = 11% (< 35%, but Mobile is adjacent to Frontend)
    }

    filtered = _filter_affinities_by_domain(
        affinities,
        [fe_cluster, mobile_cluster],
        domain_scores,
    )

    cluster_names = [a.cluster_name for a in filtered]
    assert "Desarrollador Frontend Web" in cluster_names
    assert "Desarrollador Mobile Flutter" in cluster_names
