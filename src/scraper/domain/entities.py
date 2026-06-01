"""Scraper module domain entities.

Defines the data contracts that bridge devalign-scraping (producer)
and devalign-api (consumer). Both repos reference the same field names,
ensuring the Supabase upsert from the scraper can be read directly by
the API's ORM layer.

Integration mode: Shared Database
    - devalign-scraping → writes to job_offers (Supabase upsert)
    - devalign-api      → reads job_offers for ML clustering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID


class JobPlatform(StrEnum):
    GETONBOARD = "getonboard"
    COMPUTRABAJO = "computrabajo"


@dataclass
class JobOffer:
    """A job offer scraped from a job platform.

    Field names match 1:1 with the job_offers table columns.
    This is the canonical contract between the scraper and the API.

    Notes:
        - salary: raw text as extracted ("S/. 3,500", "A convenir", "USD 2,500 - 4,000")
        - experience_years: raw text ("2 a 4 años de experiencia", "No especificado")
        - hard_skills / soft_skills: raw string lists (staging; normalized by ML Engine)
        - source_url: unique constraint in DB — used as upsert key
    """

    id: UUID | None = None
    platform: JobPlatform = JobPlatform.COMPUTRABAJO

    # Core fields — match job_offers columns exactly
    title: str = ""  # → job_title
    company: str = ""
    location: str = ""
    salary: str = ""  # varchar(100) — raw text from scraper
    modality: str = ""  # "Remoto" | "Híbrido" | "Presencial"
    experience_years: str = ""  # varchar(100) — raw text from scraper
    education_level: str = ""
    raw_description: str = ""  # → full_description
    source_url: str = ""
    date_posted: str = ""  # "Hace 2 días" | "2026-04-30"
    scraped_at: str = ""

    # Staging skill arrays — stored as JSONB in job_offers
    # ML Engine normalizes these into the skills + offer_skills tables
    hard_skills: list[str] = field(default_factory=list)  # → raw_hard_skills
    soft_skills: list[str] = field(default_factory=list)  # → raw_soft_skills

    @property
    def portal(self) -> str:
        """Alias for the DB column name."""
        return self.platform.value
