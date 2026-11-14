"""Domain services for ML Engine."""

from src.ml_engine.domain.services.affinity_calculator import (
    CANONICAL_DOMAINS,
    DOMAIN_ADJACENCY,
    DOMAIN_RADAR_THRESHOLD,
    compute_affinities_and_domains,
    compute_domain_affinities,
    determine_trend,
    is_concept_skill,
    normalize_domain_key,
)

__all__ = [
    "CANONICAL_DOMAINS",
    "DOMAIN_ADJACENCY",
    "DOMAIN_RADAR_THRESHOLD",
    "compute_affinities_and_domains",
    "compute_domain_affinities",
    "determine_trend",
    "is_concept_skill",
    "normalize_domain_key",
]
