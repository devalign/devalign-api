"""Unit tests for ICT (Índice de Competencia Técnica) calculation on domain entity."""

from src.ml_engine.domain.entities import SeniorityLevel, Skill


def test_compute_ict_zero_experience():
    """No experience, no projects, no certifications -> 0.0."""
    skill = Skill(
        name="Python",
        self_taught=False,
        personal_projects=False,
        years_of_experience=0,
        has_certification=False,
    )
    assert skill.calculate_ict() == 0.0


def test_compute_ict_experience_scale():
    """Verify diminishing returns scale for experience (1 to 5+ years)."""
    assert Skill(name="P", years_of_experience=1).calculate_ict() == 2.5
    assert Skill(name="P", years_of_experience=2).calculate_ict() == 3.5
    assert Skill(name="P", years_of_experience=3).calculate_ict() == 4.2
    assert Skill(name="P", years_of_experience=4).calculate_ict() == 4.7
    assert Skill(name="P", years_of_experience=5).calculate_ict() == 5.0
    assert Skill(name="P", years_of_experience=10).calculate_ict() == 5.0  # Capped at 5.0


def test_compute_ict_projects_and_training():
    """Verify projects and training addition."""
    # 0 exp + personal projects (3.0) + courses (1.0) = 4.0
    assert (
        Skill(
            name="P", personal_projects=True, self_taught=True, years_of_experience=0
        ).calculate_ict()
        == 4.0
    )

    # 0 exp + personal projects (3.0) + certification (2.0) = 5.0
    assert (
        Skill(
            name="P", personal_projects=True, has_certification=True, years_of_experience=0
        ).calculate_ict()
        == 5.0
    )

    # 0 exp + personal projects (3.0) + courses (1.0) + certification (2.0) -> training capped at 2.0 -> 5.0
    assert (
        Skill(
            name="P",
            personal_projects=True,
            self_taught=True,
            has_certification=True,
            years_of_experience=0,
        ).calculate_ict()
        == 5.0
    )


def test_compute_ict_full_mastery_cap():
    """Verify max score is capped at 10.0."""
    # 5+ exp (5.0) + projects (3.0) + cert (2.0) = 10.0
    assert (
        Skill(
            name="P",
            self_taught=True,
            personal_projects=True,
            years_of_experience=8,
            has_certification=True,
        ).calculate_ict()
        == 10.0
    )


def test_compute_ict_seniority_fallback():
    """Verify seniority heuristic fallback when experience is 0."""
    assert Skill(name="P", years_of_experience=0).calculate_ict(SeniorityLevel.SENIOR) == 4.2
    assert Skill(name="P", years_of_experience=0).calculate_ict(SeniorityLevel.STAFF) == 4.7
    assert Skill(name="P", years_of_experience=0).calculate_ict(SeniorityLevel.MID) == 3.5
