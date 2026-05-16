"""Scraper module API router — stub for future integration."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/scraper", tags=["Scraper — Data Acquisition"])


class ScraperStatusResponse(BaseModel):
    status: str
    job_offer_count: int
    message: str


@router.get(
    "/status",
    response_model=ScraperStatusResponse,
    summary="Get scraper and dataset status",
)
async def get_scraper_status() -> ScraperStatusResponse:
    """
    Returns the current status of the scraping pipeline and dataset.

    NOTE: The scraper module is stubbed. Full integration with the
    external scraper repository is planned for Phase 4.
    """
    return ScraperStatusResponse(
        status="stub",
        job_offer_count=0,
        message=(
            "Scraper module is pending integration with the external scraper repository. "
            "Run clustering manually after importing job offer data."
        ),
    )
