"""Scraper module domain ports (interfaces)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from src.scraper.domain.entities import JobOffer


class ScraperPort(ABC):
    """Port for web scraping operations.

    NOTE: The scraper is implemented in devalign-scraping (external repo).
    This port is kept here for potential future direct invocation.
    Current integration strategy: Shared Database (Supabase).
    """

    @abstractmethod
    async def scrape(self, max_offers: int = 5000) -> list[JobOffer]:
        """Scrape job offers and return normalized results."""
        ...


class JobOfferRepository(ABC):
    """Port for reading job offer data from the shared database.

    The API reads job offers that were produced by devalign-scraping.
    Write operations (upsert) are handled by the scraper's SupabaseExporter.
    """

    @abstractmethod
    async def save_batch(self, offers: list[JobOffer]) -> int:
        """Persist a batch of job offers. Returns count saved."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return total number of stored job offers."""
        ...

    @abstractmethod
    async def count_by_portal(self) -> dict[str, int]:
        """Return offer counts grouped by portal (e.g. {"computrabajo": 3200, "getonboard": 1800})."""
        ...

    @abstractmethod
    async def get_latest_scraped_at(self) -> datetime | None:
        """Return the most recent scraped_at timestamp across all offers.

        Used by the /scraper/status endpoint to report data freshness.
        """
        ...

    @abstractmethod
    async def get_all_for_clustering(self) -> list[JobOffer]:
        """Return all offers formatted for the K-Prototypes clustering pipeline."""
        ...
