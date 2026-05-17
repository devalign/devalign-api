"""SQLAlchemy implementation of MLJobOfferRepository."""

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.ml_engine.domain.ports import MLJobOfferRepository
from src.scraper.infrastructure.models import JobOfferModel, OfferSkillModel


class SQLMLJobOfferRepository(MLJobOfferRepository):
    """SQLAlchemy implementation of MLJobOfferRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_unnormalized_offers(self, limit: int = 100) -> list[dict]:
        """Retrieve job offers that have not yet been normalized."""
        stmt = (
            select(JobOfferModel)
            .where(JobOfferModel.is_normalized == False)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        
        return [
            {
                "id": m.job_offer_id,
                "raw_hard_skills": m.raw_hard_skills or [],
                "raw_soft_skills": m.raw_soft_skills or [],
            }
            for m in models
        ]

    async def save_offer_skills(self, offer_skills: list[dict]) -> None:
        """Bulk save offer_skills relations."""
        if not offer_skills:
            return

        models = [
            OfferSkillModel(
                job_offer_id=os["job_offer_id"],
                skill_id=os["skill_id"],
                skill_type=os["skill_type"],
            )
            for os in offer_skills
        ]
        self._session.add_all(models)
        await self._session.flush()

    async def mark_as_normalized(self, job_offer_ids: list[UUID]) -> None:
        """Mark job offers as normalized."""
        if not job_offer_ids:
            return
            
        stmt = (
            update(JobOfferModel)
            .where(JobOfferModel.job_offer_id.in_(job_offer_ids))
            .values(is_normalized=True)
        )
        await self._session.execute(stmt)
        await self._session.flush()
