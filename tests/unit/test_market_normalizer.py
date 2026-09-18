from datetime import UTC, datetime

from src.scraper.application.market_normalizer import (
    parse_experience_years,
    parse_relative_date,
    parse_salary,
)


def test_parse_salary_negotiable_and_empty():
    res = parse_salary("A convenir", "pe")
    assert res.is_negotiable is True
    assert res.min_salary_usd is None

    res2 = parse_salary("", "co")
    assert res2.is_negotiable is True


def test_parse_salary_usd_range():
    res = parse_salary("$2,000 - $3,500 USD / month", "latam")
    assert res.is_negotiable is False
    assert res.min_salary_usd == 2000.0
    assert res.max_salary_usd == 3500.0
    assert res.currency == "USD"


def test_parse_salary_pen():
    res = parse_salary("S/. 3,750", "pe")
    assert res.is_negotiable is False
    assert res.min_salary_usd == 1000.0
    assert res.currency == "PEN"


def test_parse_salary_cop():
    res = parse_salary("$ 4.000.000 (Mensual)", "co")
    assert res.is_negotiable is False
    assert res.min_salary_usd == 1000.0
    assert res.currency == "COP"


def test_parse_relative_date():
    ref = datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC)

    # ISO
    res_iso = parse_relative_date("2026-09-10", ref)
    assert res_iso.year == 2026 and res_iso.month == 9 and res_iso.day == 10

    # Ayer
    res_ayer = parse_relative_date("Ayer", ref)
    assert res_ayer.day == 16

    # Hace 3 días
    res_dias = parse_relative_date("Hace 3 días", ref)
    assert res_dias.day == 14


def test_parse_experience():
    assert parse_experience_years("Sin experiencia") == (0, 0)
    assert parse_experience_years("2 a 4 años de experiencia") == (2, 4)
    assert parse_experience_years("3+ años") == (3, None)
    assert parse_experience_years("No especificado") == (None, None)
