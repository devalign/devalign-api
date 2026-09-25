"""Unit tests for CV extraction unpacking and skill search mapping."""

from uuid import uuid4

from src.ml_engine.application.use_cases import (
    _clean_and_unpack_skills,
    _cluster_affinity_to_dto,
)
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    Skill,
    SkillGap,
    SkillNature,
)


def test_clean_and_unpack_skills_parentheses():
    """Verify that tools listed in parentheses are unpacked into independent skill objects."""
    raw_payload = {
        "skills": [
            {
                "name": "CI/CD (Bitbucket, Jenkins, GitHub Actions)",
                "category": "tools",
                "years_of_experience": 3,
            },
            {
                "name": "Azure (Functions, Key Vault, Service Bus)",
                "category": "tools",
                "years_of_experience": 2,
            },
            {
                "name": "TypeScript/JavaScript",
                "category": "technical",
                "years_of_experience": 4,
            },
            {
                "name": "Docker",
                "category": "tools",
                "years_of_experience": 3,
            },
        ]
    }

    result = _clean_and_unpack_skills(raw_payload)
    names = [s["name"] for s in result["skills"]]

    # Verify parenthetical sub-tools
    assert "Bitbucket" in names
    assert "Jenkins" in names
    assert "GitHub Actions" in names
    assert "CI/CD" in names

    # Verify Azure sub-tools
    assert "Azure Functions" in names
    assert "Azure Key Vault" in names
    assert "Azure Service Bus" in names
    assert "Azure" in names

    # Verify slash splitting
    assert "TypeScript" in names
    assert "JavaScript" in names

    # Verify standalone tool preserved
    assert "Docker" in names


def test_clean_and_unpack_skills_string_list():
    """Verify that skills provided as a streamlined string array are correctly parsed and unpacked."""
    raw_payload = {
        "years_experience": 4,
        "skills": [
            "Python",
            "FastAPI",
            "CI/CD (Bitbucket, Jenkins, GitHub Actions)",
            "AWS (Lambda, S3)",
            "TypeScript/JavaScript",
            "Docker",
        ],
    }

    result = _clean_and_unpack_skills(raw_payload)
    names = [s["name"] for s in result["skills"]]

    assert "Python" in names
    assert "FastAPI" in names
    assert "CI/CD" in names
    assert "Bitbucket" in names
    assert "Jenkins" in names
    assert "GitHub Actions" in names
    assert "AWS" in names
    assert "AWS Lambda" in names
    assert "AWS S3" in names
    assert "TypeScript" in names
    assert "JavaScript" in names
    assert "Docker" in names

    for item in result["skills"]:
        assert item["years_of_experience"] == 4
        assert item["category"] == "technical"
        assert item["self_taught"] is False


def test_cluster_affinity_to_dto_mapping():
    """Verify ClusterAffinity converts to ClusterAffinityDTO with detected skills and gaps."""
    cluster_id = uuid4()
    skill_tech = Skill(
        name="Java",
        nature=SkillNature.TECH,
        normalized_name="java",
        weight=2.0,
        frequency=0.9,
    )
    gap_tech = SkillGap(
        skill=Skill(
            name="Spring Boot",
            nature=SkillNature.TECH,
            normalized_name="springboot",
            weight=1.5,
            frequency=0.8,
        ),
        market_importance="high",
    )

    affinity = ClusterAffinity(
        cluster_id=cluster_id,
        cluster_name="Desarrollador Backend Java",
        affinity_score=0.85,
        is_primary=True,
        ai_insight="Alta afinidad en Java backend",
        detected_skills=[skill_tech],
        skill_gaps=[gap_tech],
    )

    user_skills_map = {
        "java": Skill(
            name="Java",
            nature=SkillNature.TECH,
            normalized_name="java",
            years_of_experience=4,
            self_taught=False,
            ict_score=8.0,
        )
    }

    dto = _cluster_affinity_to_dto(affinity, is_primary=True, user_skills_map=user_skills_map)

    assert dto.cluster_id == cluster_id
    assert dto.cluster_name == "Desarrollador Backend Java"
    assert dto.affinity_score == 0.85
    assert dto.is_primary is True
    assert len(dto.detected_skills) == 1
    assert dto.detected_skills[0].name == "Java"
    assert dto.detected_skills[0].years_of_experience == 4
    assert dto.detected_skills[0].ict_score == 8.0
    assert len(dto.skill_gaps) == 1
    assert dto.skill_gaps[0].name == "Spring Boot"
    assert dto.skill_gaps[0].market_importance == "high"


def test_clean_and_unpack_ocr_recovery():
    """Verify that OCR-truncated skill names (e.g. 'jQuer', 'wordpres', '11ty') are recovered."""
    raw_payload = {
        "skills": [
            "jQuer",
            "wordpres",
            "elevent",
            "11ty",
            "Figma",
            "Timber",
        ]
    }
    result = _clean_and_unpack_skills(raw_payload)
    names = [s["name"] for s in result["skills"]]

    assert "jQuery" in names
    assert "WordPress" in names
    assert "Eleventy" in names
    assert "Figma" in names
    assert "Timber" in names
