"""Market data normalization module.

Provides pure functions to normalize:
- Salaries from multi-country currencies (PE, CO, MX, CL, AR, USD, EUR) to standard monthly USD.
- Relative date strings ("Ayer", "Hace 2 días") to absolute timestamps.
- Experience requirements ("2 a 4 años", "3+ años") to numeric bounds.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

# Approximate FX rates to 1 USD
FX_RATES_TO_USD: dict[str, float] = {
    "pe": 3.75,  # PEN (S/.)
    "co": 4000.0,  # COP ($)
    "mx": 18.0,  # MXN ($)
    "cl": 950.0,  # CLP ($)
    "ar": 1100.0,  # ARS ($)
    "usd": 1.0,
    "eur": 0.92,
}

NEGOTIABLE_KEYWORDS = {
    "convenir",
    "no especificado",
    "acorde",
    "mercado",
    "segun perfil",
    "según perfil",
    "confidencial",
    "competitivo",
    "a tratar",
    "negociable",
    "no visible",
    "oculto",
}


class ParsedSalary(NamedTuple):
    min_salary_usd: float | None
    max_salary_usd: float | None
    currency: str | None
    is_negotiable: bool


class ParsedExperience(NamedTuple):
    min_years: int | None
    max_years: int | None


def parse_salary(salary_str: str | None, country: str | None = None) -> ParsedSalary:
    """Normalizes raw salary text into USD monthly bounds."""
    if not salary_str or not isinstance(salary_str, str):
        return ParsedSalary(
            min_salary_usd=None, max_salary_usd=None, currency=None, is_negotiable=True
        )

    text = salary_str.strip().lower()
    if not text:
        return ParsedSalary(
            min_salary_usd=None, max_salary_usd=None, currency=None, is_negotiable=True
        )

    # Check for negotiable keywords
    if any(kw in text for kw in NEGOTIABLE_KEYWORDS):
        return ParsedSalary(
            min_salary_usd=None, max_salary_usd=None, currency=None, is_negotiable=True
        )

    # Determine currency
    country_code = (country or "").strip().lower()
    detected_currency = "USD"
    fx_rate = 1.0

    if "usd" in text or "u$s" in text or "dólares" in text or "dolares" in text:
        detected_currency = "USD"
        fx_rate = 1.0
    elif "eur" in text or "€" in text or "euros" in text:
        detected_currency = "EUR"
        fx_rate = 0.92
    elif "s/." in text or "s/" in text or "soles" in text or "pen" in text:
        detected_currency = "PEN"
        fx_rate = FX_RATES_TO_USD["pe"]
    elif "cop" in text:
        detected_currency = "COP"
        fx_rate = FX_RATES_TO_USD["co"]
    elif "mxn" in text:
        detected_currency = "MXN"
        fx_rate = FX_RATES_TO_USD["mx"]
    elif "clp" in text:
        detected_currency = "CLP"
        fx_rate = FX_RATES_TO_USD["cl"]
    elif "ars" in text:
        detected_currency = "ARS"
        fx_rate = FX_RATES_TO_USD["ar"]
    elif country_code in FX_RATES_TO_USD:
        # Infer currency from country if not explicit
        detected_currency = {
            "pe": "PEN",
            "co": "COP",
            "mx": "MXN",
            "cl": "CLP",
            "ar": "ARS",
        }.get(country_code, "USD")
        fx_rate = FX_RATES_TO_USD.get(country_code, 1.0)

    # Extract all numeric values
    # Match numbers like 4.500.000 or 4,500,000 or 3500
    cleaned = re.sub(r"\s+", " ", text)
    matches = re.findall(r"\b\d+(?:[\.,]\d{3})*(?:[\.,]\d{1,2})?\b|\b\d+\b", cleaned)

    numbers: list[float] = []
    for m in matches:
        # Standardize thousands separator
        m_clean = m
        if "." in m_clean and "," in m_clean:
            # 1,500.00 or 1.500,00
            if m_clean.rfind(",") > m_clean.rfind("."):
                m_clean = m_clean.replace(".", "").replace(",", ".")
            else:
                m_clean = m_clean.replace(",", "")
        elif "." in m_clean:
            # 4.500.000 or 3.50
            if m_clean.count(".") > 1 or len(m_clean.split(".")[-1]) == 3:
                m_clean = m_clean.replace(".", "")
        elif "," in m_clean:
            # 4,500,000 or 3,50
            if m_clean.count(",") > 1 or len(m_clean.split(",")[-1]) == 3:
                m_clean = m_clean.replace(",", "")
            else:
                m_clean = m_clean.replace(",", ".")

        try:
            val = float(m_clean)
            if val > 0:
                numbers.append(val)
        except ValueError:
            continue

    if not numbers:
        return ParsedSalary(
            min_salary_usd=None, max_salary_usd=None, currency=None, is_negotiable=True
        )

    # Detect periodicity
    is_annual = "anual" in text or "año" in text or "yr" in text or "year" in text
    is_hourly = "hora" in text or "/hr" in text or "hr" in text or "hour" in text

    period_multiplier = 1.0
    if is_annual:
        period_multiplier = 1.0 / 12.0
    elif is_hourly:
        period_multiplier = 160.0  # 160 hrs / month full-time

    # Sort or assign range
    raw_min = min(numbers)
    raw_max = max(numbers)

    # If currency is USD but amount > 30000 and country is CO/CL, it was likely local currency with $ symbol
    if detected_currency == "USD" and raw_min > 50000 and country_code in ("co", "cl"):
        fx_rate = FX_RATES_TO_USD.get(country_code, 1.0)
        detected_currency = "COP" if country_code == "co" else "CLP"

    min_usd = (raw_min * period_multiplier) / fx_rate
    max_usd = (raw_max * period_multiplier) / fx_rate

    # Filter unreasonable bounds for monthly tech salaries in USD ($150 - $35,000 USD/mo)
    if min_usd < 150 or min_usd > 35000:
        min_usd = None
    if max_usd < 150 or max_usd > 35000:
        max_usd = None

    if min_usd is None and max_usd is None:
        return ParsedSalary(
            min_salary_usd=None, max_salary_usd=None, currency=detected_currency, is_negotiable=True
        )

    return ParsedSalary(
        min_salary_usd=round(min_usd, 2) if min_usd is not None else None,
        max_salary_usd=round(max_usd, 2) if max_usd is not None else None,
        currency=detected_currency,
        is_negotiable=False,
    )


def parse_relative_date(
    date_str: str | None, scraped_at: datetime | None = None
) -> datetime | None:
    """Parses relative and absolute date strings anchored to scraped_at."""
    if not date_str or not isinstance(date_str, str):
        return scraped_at

    ref = scraped_at or datetime.now(UTC)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=UTC)

    s = date_str.strip().lower()
    if not s:
        return ref

    # Direct ISO date YYYY-MM-DD
    iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if iso_match:
        try:
            year, month, day = map(int, iso_match.groups())
            return datetime(year, month, day, 12, 0, 0, tzinfo=UTC)
        except ValueError:
            pass

    # "Publicado hoy", "hoy", "reciente", "hace X horas", "hace X minutos"
    if "hoy" in s or "reciente" in s or "hora" in s or "minuto" in s:
        return ref

    # "Ayer", "publicado ayer"
    if "ayer" in s:
        return ref - timedelta(days=1)

    # "Hace X días", "hace X dias"
    days_match = re.search(r"hace\s+(\d+)\s+d[íi]as?", s)
    if days_match:
        days = int(days_match.group(1))
        return ref - timedelta(days=days)

    # "Hace X semanas", "hace X sem"
    weeks_match = re.search(r"hace\s+(\d+)\s+sem", s)
    if weeks_match:
        weeks = int(weeks_match.group(1))
        return ref - timedelta(weeks=weeks)

    # "Hace X meses", "hace más de 1 mes"
    months_match = re.search(r"hace\s+(\d+)\s+mes", s)
    if months_match:
        months = int(months_match.group(1))
        return ref - timedelta(days=months * 30)

    if "más de 1 mes" in s or "mas de 1 mes" in s or "más de 30 días" in s:
        return ref - timedelta(days=35)

    return ref


def parse_experience_years(exp_str: str | None) -> ParsedExperience:
    """Parses experience requirement text into numeric minimum and maximum bounds."""
    if not exp_str or not isinstance(exp_str, str):
        return ParsedExperience(min_years=None, max_years=None)

    text = exp_str.strip().lower()
    if not text or "no especific" in text or "a convenir" in text:
        return ParsedExperience(min_years=None, max_years=None)

    if (
        "sin experiencia" in text
        or "no requerid" in text
        or "primer empleo" in text
        or "0 año" in text
    ):
        return ParsedExperience(min_years=0, max_years=0)

    # Range: "2 a 4 años", "2 - 5 años", "de 1 a 3 años"
    range_match = re.search(r"(\d+)\s*(?:a|-|hasta)\s*(\d+)\s*a[ñn]o", text)
    if range_match:
        val1, val2 = int(range_match.group(1)), int(range_match.group(2))
        return ParsedExperience(min_years=min(val1, val2), max_years=max(val1, val2))

    # Single bound: "3+ años", "más de 3 años", "mínimo 2 años", "al menos 1 año", "3 años"
    single_match = re.search(r"(\d+)\s*(?:\+|a[ñn]o)", text)
    if single_match:
        val = int(single_match.group(1))
        return ParsedExperience(min_years=val, max_years=None)

    # Any standalone number
    any_num = re.search(r"\b(\d+)\b", text)
    if any_num:
        val = int(any_num.group(1))
        if val <= 25:
            return ParsedExperience(min_years=val, max_years=None)

    return ParsedExperience(min_years=None, max_years=None)
