"""Unit test ensuring modular use cases and domain services import and re-export correctly."""


def test_use_cases_backward_compatibility_reexports():
    """All historic public symbols can be imported from src.ml_engine.application.use_cases."""
    from src.ml_engine.application.use_cases import (
        CANONICAL_DOMAINS,
        DOMAIN_ADJACENCY,
        DOMAIN_RADAR_THRESHOLD,
        EvaluateClusterDiagnosticUseCase,
        GetClusterDiagnosticUseCase,
        GetKnowledgeGraphUseCase,
        GetMyProfileUseCase,
        ListClustersUseCase,
        NormalizeSkillsUseCase,
        ProfileUserFromCVUseCase,
        _build_combined_cv_extraction_prompt,
        _clean_and_unpack_skills,
        _cluster_affinity_to_dto,
        _estimate_seniority,
        _filter_affinities_by_domain,
        _get_cluster_primary_domain,
        _looks_like_a_cv,
        _nature_from_category,
        _normalize_demand_percentage,
        _parse_cv_extraction_output,
        compute_affinities_and_domains,
        compute_domain_affinities,
        determine_trend,
        is_concept_skill,
        normalize_domain_key,
    )

    assert len(CANONICAL_DOMAINS) == 7
    assert "Backend" in CANONICAL_DOMAINS
    assert "Frontend" in DOMAIN_ADJACENCY
    assert DOMAIN_RADAR_THRESHOLD == 0.35
    assert callable(compute_affinities_and_domains)
    assert callable(compute_domain_affinities)
    assert callable(determine_trend)
    assert callable(is_concept_skill)
    assert callable(normalize_domain_key)
    assert callable(_normalize_demand_percentage)
    assert callable(_cluster_affinity_to_dto)
    assert callable(_filter_affinities_by_domain)
    assert callable(_get_cluster_primary_domain)
    assert callable(_looks_like_a_cv)
    assert callable(_parse_cv_extraction_output)
    assert callable(_nature_from_category)
    assert callable(_estimate_seniority)
    assert callable(_clean_and_unpack_skills)
    assert callable(_build_combined_cv_extraction_prompt)
    assert ProfileUserFromCVUseCase is not None
    assert ListClustersUseCase is not None
    assert NormalizeSkillsUseCase is not None
    assert GetKnowledgeGraphUseCase is not None
    assert GetMyProfileUseCase is not None
    assert EvaluateClusterDiagnosticUseCase is not None
    assert GetClusterDiagnosticUseCase is not None


def test_domain_services_direct_imports():
    """Domain services can be imported directly from src.ml_engine.domain.services."""
    from src.ml_engine.domain.services import (
        CANONICAL_DOMAINS,
        DOMAIN_ADJACENCY,
        DOMAIN_RADAR_THRESHOLD,
        compute_affinities_and_domains,
        compute_domain_affinities,
        determine_trend,
        is_concept_skill,
        normalize_domain_key,
    )

    assert len(CANONICAL_DOMAINS) == 7
    assert DOMAIN_RADAR_THRESHOLD == 0.35
    assert "Frontend" in DOMAIN_ADJACENCY
    assert callable(compute_affinities_and_domains)
    assert callable(compute_domain_affinities)
    assert callable(is_concept_skill)
    assert normalize_domain_key("backend") == ["Backend"]
    assert determine_trend("python") == "growing"
