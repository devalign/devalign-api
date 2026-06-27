"""SQLAlchemy implementation of ClusterRepository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.ml_engine.domain.entities import Skill, SkillNature, TechCluster
from src.ml_engine.domain.ports import ClusterRepository
from src.ml_engine.infrastructure.models import ClusterModel, ClusterSkillModel


class SQLClusterRepository(ClusterRepository):
    """SQLAlchemy implementation of ClusterRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_all_active(self) -> list[TechCluster]:
        """Retrieve all active tech clusters."""
        result = await self._session.execute(
            select(ClusterModel).options(
                selectinload(ClusterModel.cluster_skills).selectinload(ClusterSkillModel.skill)
            )
        )
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def get_by_id(self, cluster_id: UUID) -> TechCluster | None:
        """Retrieve a cluster by ID."""
        result = await self._session.execute(
            select(ClusterModel)
            .where(ClusterModel.cluster_id == cluster_id)
            .options(
                selectinload(ClusterModel.cluster_skills).selectinload(ClusterSkillModel.skill)
            )
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    def _to_entity(self, model: ClusterModel) -> TechCluster:
        centroid_skills = []
        for cs in model.cluster_skills:
            if cs.skill:
                centroid_skills.append(
                    Skill(
                        id=cs.skill.skill_id,
                        name=cs.skill.name,
                        nature=SkillNature(cs.skill.nature)
                        if cs.skill.nature
                        else SkillNature.TECH,
                        normalized_name=cs.skill.name.lower().replace(" ", "").replace(".", ""),
                        weight=float(cs.skill.weight),
                        frequency=float(cs.importance_score)
                        if cs.importance_score is not None
                        else 1.0,
                        domain_tags=cs.skill.domain_tags or [],
                        core_domains=cs.skill.core_domains or [],
                    )
                )

        return TechCluster(
            id=model.cluster_id,
            name=model.name,
            description=model.description or "",
            centroid_skills=centroid_skills,
            job_offer_count=model.job_offer_count,
            cluster_index=0,
            market_insights=model.market_insights,
            compatible_roles=model.compatible_roles,
        )
