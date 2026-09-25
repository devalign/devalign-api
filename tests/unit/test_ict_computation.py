"""Unit tests for ICT (Índice de Competencia Técnica) calculation."""

from src.delivery.interface.router import _compute_ict


def test_compute_ict_zero_experience():
    """No experience, no projects, no certifications -> 0.0."""
    score = _compute_ict(
        self_taught=False,
        personal_projects=False,
        years_of_experience=0,
        has_certification=False,
    )
    assert score == 0.0


def test_compute_ict_experience_scale():
    """Verify diminishing returns scale for experience (1 to 5+ years)."""
    assert _compute_ict(False, False, 1, False) == 2.5
    assert _compute_ict(False, False, 2, False) == 3.5
    assert _compute_ict(False, False, 3, False) == 4.2
    assert _compute_ict(False, False, 4, False) == 4.7
    assert _compute_ict(False, False, 5, False) == 5.0
    assert _compute_ict(False, False, 10, False) == 5.0  # Capped at 5.0


def test_compute_ict_projects_and_training():
    """Verify projects and training addition."""
    # 0 exp + personal projects (3.0) + courses (1.0) = 4.0
    assert _compute_ict(True, True, 0, False) == 4.0

    # 0 exp + personal projects (3.0) + certification (2.0) = 5.0
    assert _compute_ict(False, True, 0, True) == 5.0

    # 0 exp + personal projects (3.0) + courses (1.0) + certification (2.0) -> training capped at 2.0 -> 5.0
    assert _compute_ict(True, True, 0, True) == 5.0


def test_compute_ict_full_mastery_cap():
    """Verify max score is capped at 10.0."""
    # 5+ exp (5.0) + projects (3.0) + cert (2.0) = 10.0
    assert _compute_ict(True, True, 8, True) == 10.0
