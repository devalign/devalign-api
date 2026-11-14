"""Domain service for tech cluster and domain affinity calculations."""

import math
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from uuid import UUID

from src.ml_engine.application.dtos import DomainAffinityDTO
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    Skill,
    SkillGap,
    SkillNature,
    TechCluster,
)

logger = structlog.get_logger("affinity_calculator")

CANONICAL_DOMAINS: list[str] = [
    "Backend",
    "Frontend",
    "Data",
    "DevOps",
    "QA",
    "Mobile",
    "Cloud",
]

DOMAIN_ADJACENCY: dict[str, set[str]] = {
    "Frontend": {"Frontend", "Mobile"},
    "Backend": {"Backend", "Frontend", "Data"},
    "Mobile": {"Mobile", "Frontend"},
    "DevOps": {"DevOps", "Cloud", "Backend"},
    "Cloud": {"Cloud", "DevOps", "Backend"},
    "Data": {"Data", "Backend", "Cloud"},
    "QA": {"QA", "DevOps", "Backend"},
}

DOMAIN_RADAR_THRESHOLD = 0.35

CONCEPT_PATTERNS: tuple[str, ...] = (
    "back end",
    "front end",
    "full stack",
    "agile",
    "configuration management",
    "software engineering",
    "software development",
    "cloud computing",
    "web development",
    "object-oriented",
    "concept",
    "pattern",
    "methodology",
    "architecture",
    "principle",
    "paradigm",
)

MOCK_TRENDS = {
    "react": "growing",
    "python": "growing",
    "docker": "growing",
    "kubernetes": "growing",
    "aws": "growing",
    "cloud computing": "growing",
    "typescript": "growing",
    "machine learning": "growing",
    "inteligencia artificial": "growing",
    "devops": "growing",
    "microservicios": "growing",
    "fastapi": "growing",
    "next.js": "growing",
    "tailwind css": "growing",
    "scrum": "stable",
    "sql": "stable",
    "git": "stable",
    "graphql": "stable",
    "ci/cd": "stable",
    "comunicación": "stable",
    "liderazgo": "stable",
    "angular": "declining",
    "jquery": "shrinking",
    "php": "declining",
    "cobol": "shrinking",
}


def is_concept_skill(skill: Any) -> bool:
    """Check if a skill is an abstract/umbrella conceptual skill rather than an actionable tool."""
    if not skill:
        return False
    if hasattr(skill, "nature") and skill.nature == SkillNature.CONCEPT:
        return True
    s_name = (getattr(skill, "name", "") or "").lower().strip()
    return any(pat in s_name for pat in CONCEPT_PATTERNS)


def determine_trend(name: str) -> str:
    """Determine trend for a given skill name."""
    norm_name = name.lower().strip()
    return MOCK_TRENDS.get(norm_name, "stable")


def _normalize_demand_percentage(frequency: float | None) -> int:
    """Normalize skill frequency or importance metric to a market demand percentage (20% - 98%)."""
    if frequency is None:
        return 70
    if 0.0 < frequency < 1.0:
        return round(frequency * 100)
    scaled = (frequency / 3.0) * 100
    return min(98, max(20, round(scaled)))


def normalize_domain_key(domain_str: str) -> list[str]:
    """Normalize domain strings to canonical title-case names."""
    clean = domain_str.strip().lower()
    if clean in ("backend", "back-end", "back_end"):
        return ["Backend"]
    if clean in ("frontend", "front-end", "front_end", "ui", "ux", "web"):
        return ["Frontend"]
    if clean in ("devops", "ci/cd", "ci_cd", "cicd", "infra", "infrastructure", "sre"):
        return ["DevOps"]
    if clean in ("cloud", "aws", "gcp", "azure"):
        return ["Cloud"]
    if clean in ("cloud_devops", "cloud/devops"):
        return ["Cloud", "DevOps"]
    if clean in (
        "data",
        "data engineering",
        "data_engineering",
        "data science",
        "data_science",
        "analytics",
        "big data",
        "big_data",
        "database",
        "sql",
    ):
        return ["Data"]
    if clean in (
        "qa",
        "testing",
        "quality assurance",
        "quality_assurance",
        "test",
        "automation_testing",
    ):
        return ["QA"]
    if clean in ("mobile", "ios", "android", "flutter", "react native", "react_native"):
        return ["Mobile"]
    return []


def _get_cluster_primary_domain(cluster: TechCluster) -> str | None:
    """Determine the primary domain of a cluster from its top-weighted skills."""
    domain_weights: dict[str, float] = {}
    for skill in cluster.centroid_skills:
        domains: list[str] = []
        if skill.core_domains:
            for d in skill.core_domains:
                domains.extend(normalize_domain_key(d))
        elif skill.domain_tags:
            for d in skill.domain_tags:
                domains.extend(normalize_domain_key(d))

        for domain in domains:
            domain_weights[domain] = domain_weights.get(domain, 0.0) + (
                skill.weight * (skill.frequency or 1.0)
            )

    if not domain_weights:
        return None
    return max(domain_weights, key=lambda k: domain_weights[k])


def _filter_affinities_by_domain(
    affinities: list[ClusterAffinity],
    active_clusters: list[TechCluster],
    domain_scores: dict[str, float],
) -> list[ClusterAffinity]:
    """Filter affinities using the user's domain radar as primary gate."""
    if not domain_scores or not affinities:
        return affinities

    total = sum(domain_scores.values()) or 1.0
    normalized_scores = {d: v / total for d, v in domain_scores.items()}

    dominant_domain = max(normalized_scores, key=lambda k: normalized_scores[k])
    adjacent_domains = DOMAIN_ADJACENCY.get(dominant_domain, {dominant_domain})

    cluster_domain_map: dict[UUID, str | None] = {
        c.id: _get_cluster_primary_domain(c) for c in active_clusters if c.id is not None
    }

    filtered = []
    for aff in affinities:
        cluster_domain = cluster_domain_map.get(aff.cluster_id) if aff.cluster_id else None
        if cluster_domain is None:
            filtered.append(aff)
            continue

        if cluster_domain in adjacent_domains:
            filtered.append(aff)
            continue

        user_score_for_domain = normalized_scores.get(cluster_domain, 0.0)
        if user_score_for_domain >= DOMAIN_RADAR_THRESHOLD:
            logger.info(
                "Non-adjacent domain allowed by radar threshold",
                cluster=aff.cluster_name,
                cluster_domain=cluster_domain,
                user_radar_score=round(user_score_for_domain, 2),
                threshold=DOMAIN_RADAR_THRESHOLD,
            )
            filtered.append(aff)

    return filtered if filtered else affinities[:1]


def _compute_domain_relevance_multiplier(
    cluster: TechCluster,
    user_domain_scores: dict[str, float],
    total_user_domain_score: float,
) -> float:
    """Compute a domain relevance multiplier for a cluster based on user's domain radar."""
    if total_user_domain_score <= 0:
        return 1.0

    cluster_primary_domain = _get_cluster_primary_domain(cluster)
    if cluster_primary_domain is None:
        return 0.70

    user_radar_for_domain = user_domain_scores.get(cluster_primary_domain, 0.0)
    user_radar_normalized = user_radar_for_domain / total_user_domain_score
    max_user_domain_score = max(user_domain_scores.values()) / total_user_domain_score

    if max_user_domain_score <= 0:
        return 1.0

    relative_strength = user_radar_normalized / max_user_domain_score
    floor_multiplier = 0.50
    multiplier = floor_multiplier + (1.0 - floor_multiplier) * relative_strength

    return round(multiplier, 4)


def compute_affinities_and_domains(
    detected_skills: list[Skill],
    active_clusters: list[TechCluster],
    skill_trends: dict[str, float] | None = None,
) -> tuple[
    ClusterAffinity | None, list[ClusterAffinity], list[ClusterAffinity], list[DomainAffinityDTO]
]:
    """Compute cluster affinities, secondary affinities, and domain breakdown."""
    user_tech_skills = [s for s in detected_skills if s.nature == SkillNature.TECH]
    user_tech_norms = {s.normalized_name: s for s in user_tech_skills}
    user_all_norms = {s.normalized_name: s for s in detected_skills}

    domain_scores: dict[str, float] = {}
    for s in detected_skills:
        if s.core_domains:
            for d in s.core_domains:
                for norm_d in normalize_domain_key(d):
                    if norm_d not in domain_scores:
                        domain_scores[norm_d] = 0.0
                    domain_scores[norm_d] += s.weight * (
                        s.frequency if s.frequency is not None else 1.0
                    )
        elif s.domain_tags:
            for d in s.domain_tags:
                for norm_d in normalize_domain_key(d):
                    if norm_d not in domain_scores:
                        domain_scores[norm_d] = 0.0
                    domain_scores[norm_d] += s.weight * (
                        s.frequency if s.frequency is not None else 1.0
                    )
    total_domain_score = sum(domain_scores.values()) if domain_scores else 0.0

    affinities: list[ClusterAffinity] = []
    for cluster in active_clusters:
        cluster_tech_skills = [s for s in cluster.centroid_skills if s.nature == SkillNature.TECH]

        if not cluster_tech_skills:
            continue

        cluster_weight_total = (
            sum(s.weight * (s.frequency or 1.0) for s in cluster_tech_skills) or 1.0
        )
        covered_weight = 0.0
        matched_skills = []
        partial_matches = []
        missing_skills = []

        for cluster_skill in cluster_tech_skills:
            norm_name = cluster_skill.normalized_name
            w = cluster_skill.weight
            f_s = cluster_skill.frequency or 1.0
            skill_target_weight = w * f_s

            if norm_name in user_tech_norms:
                covered_weight += skill_target_weight
                matched_skills.append(cluster_skill.name)
            else:
                cluster_domains = set(cluster_skill.domain_tags or [])
                partial_match_score = 0.0
                if cluster_domains:
                    alternative_skills = [
                        u for u in user_tech_skills if set(u.domain_tags or []) & cluster_domains
                    ]
                    if alternative_skills:
                        best_alt = max(alternative_skills, key=lambda u: u.ict_score or 10.0)
                        partial_match_score = 0.5
                        partial_matches.append((cluster_skill.name, best_alt.name))

                if partial_match_score > 0.0:
                    covered_weight += skill_target_weight * partial_match_score
                else:
                    missing_skills.append(cluster_skill.name)

        raw_coverage = covered_weight / cluster_weight_total
        matching_count = len(matched_skills) + (len(partial_matches) * 0.5)
        focus_ratio = min(1.0, matching_count / max(1, len(cluster_tech_skills)))

        score = (0.85 * raw_coverage) + (0.15 * (raw_coverage * focus_ratio))

        expected_skills = 15.0
        cluster_complexity_factor = min(
            1.0, math.log(len(cluster_tech_skills) + 1) / math.log(expected_skills + 1)
        )
        score *= cluster_complexity_factor

        domain_multiplier = _compute_domain_relevance_multiplier(
            cluster, domain_scores, total_domain_score
        )
        score = round(score * domain_multiplier, 4)

        insight_parts = []
        if matched_skills:
            insight_parts.append(
                f"Dominas {len(matched_skills)} tecnologías clave (como {', '.join(matched_skills[:2])})."
            )
        if partial_matches:
            examples = [f"tienes {u} en lugar de {c}" for c, u in partial_matches[:2]]
            insight_parts.append(f"Cubres áreas relacionadas ({'; '.join(examples)}).")
        if missing_skills:
            insight_parts.append(
                f"Para mejorar, considera aprender {', '.join(missing_skills[:2])}."
            )

        ai_insight = (
            " ".join(insight_parts) if insight_parts else "Afinidad calculada en base a tu perfil."
        )

        cluster_detected_skills = []
        cluster_skill_gaps = []
        for skill in cluster.centroid_skills:
            if skill.normalized_name in user_all_norms:
                cluster_detected_skills.append(skill)
            else:
                if is_concept_skill(skill):
                    continue

                trend_multiplier = 1.0
                if skill_trends and skill.normalized_name in skill_trends:
                    trend_multiplier = skill_trends[skill.normalized_name]

                priority = skill.weight * skill.frequency * trend_multiplier
                if priority >= 2.0:
                    importance = "critical"
                elif priority >= 1.0:
                    importance = "high"
                else:
                    importance = "medium"

                cluster_skill_gaps.append(
                    SkillGap(
                        skill=skill,
                        market_importance=importance,
                    )
                )

        cluster_detected_skills.sort(
            key=lambda s: (s.frequency if s.frequency is not None else 1.0, s.weight),
            reverse=True,
        )
        cluster_skill_gaps.sort(
            key=lambda g: g.skill.weight * g.skill.frequency,
            reverse=True,
        )

        affinities.append(
            ClusterAffinity(
                cluster_id=cluster.id,
                cluster_name=cluster.name,
                affinity_score=score,
                is_primary=False,
                market_insights=cluster.market_insights,
                compatible_roles=cluster.compatible_roles,
                ai_insight=ai_insight,
                detected_skills=cluster_detected_skills,
                skill_gaps=cluster_skill_gaps,
                job_offer_count=cluster.job_offer_count,
            )
        )

    affinities.sort(key=lambda a: (a.affinity_score, len(a.detected_skills)), reverse=True)
    if not affinities:
        return None, [], [], []

    filtered_affinities = _filter_affinities_by_domain(affinities, active_clusters, domain_scores)
    if not filtered_affinities:
        filtered_affinities = affinities

    primary = filtered_affinities[0]
    primary = ClusterAffinity(
        cluster_id=primary.cluster_id,
        cluster_name=primary.cluster_name,
        affinity_score=primary.affinity_score,
        is_primary=True,
        market_insights=primary.market_insights,
        compatible_roles=primary.compatible_roles,
        ai_insight=primary.ai_insight,
        detected_skills=primary.detected_skills,
        skill_gaps=primary.skill_gaps,
        job_offer_count=primary.job_offer_count,
    )

    secondaries = filtered_affinities[1:3]
    domain_affinities_dto = compute_domain_affinities(detected_skills, active_clusters)
    return primary, secondaries, filtered_affinities, domain_affinities_dto


def compute_domain_affinities(
    detected_skills: list[Skill],
    active_clusters: list[TechCluster],
) -> list[DomainAffinityDTO]:
    """Compute normalized domain affinities radar breakdown for the user's skills."""
    domain_scores: dict[str, float] = {}
    for s in detected_skills:
        domains: list[str] = []
        if s.core_domains:
            for d in s.core_domains:
                domains.extend(normalize_domain_key(d))
        elif s.domain_tags:
            for d in s.domain_tags:
                domains.extend(normalize_domain_key(d))

        for norm_d in domains:
            if norm_d not in domain_scores:
                domain_scores[norm_d] = 0.0
            domain_scores[norm_d] += s.weight * (s.frequency if s.frequency is not None else 1.0)

    domain_demands_accum: dict[str, list[float]] = {d: [] for d in CANONICAL_DOMAINS}
    for cluster in active_clusters:
        for skill in cluster.centroid_skills:
            domains = []
            if skill.core_domains:
                for d in skill.core_domains:
                    domains.extend(normalize_domain_key(d))
            elif skill.domain_tags:
                for d in skill.domain_tags:
                    domains.extend(normalize_domain_key(d))

            for norm_d in domains:
                if norm_d in domain_demands_accum and skill.frequency is not None:
                    domain_demands_accum[norm_d].append(skill.frequency)

    domain_market_demand: dict[str, float] = {}
    for d, freqs in domain_demands_accum.items():
        if freqs:
            normalized_freqs = [_normalize_demand_percentage(f) / 100.0 for f in freqs]
            domain_market_demand[d] = round(sum(normalized_freqs) / len(normalized_freqs), 4)
        else:
            domain_market_demand[d] = 0.50

    total_domain_score = sum(domain_scores.values()) if domain_scores else 1.0
    domain_affinities_dto = [
        DomainAffinityDTO(
            domain=d,
            affinity_score=round(domain_scores.get(d, 0.0) / total_domain_score, 4)
            if total_domain_score > 0
            else 0.0,
            market_demand=domain_market_demand.get(d, 0.50),
        )
        for d in CANONICAL_DOMAINS
    ]
    domain_affinities_dto.sort(
        key=lambda x: (x.affinity_score, CANONICAL_DOMAINS.index(x.domain)),
        reverse=False,
    )
    domain_affinities_dto.sort(key=lambda x: x.affinity_score, reverse=True)
    return domain_affinities_dto
