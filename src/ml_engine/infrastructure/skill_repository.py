"""SQLAlchemy implementation of SkillRepository."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.ml_engine.domain.entities import Skill, SkillType
from src.ml_engine.domain.ports import SkillRepository
from src.ml_engine.infrastructure.models import SkillModel


class SQLSkillRepository(SkillRepository):
    """SQLAlchemy implementation of SkillRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_all_skills(self) -> list[Skill]:
        """Retrieve all canonical skills for similarity checking."""
        result = await self._session.execute(select(SkillModel))
        models = result.scalars().all()
        return [
            Skill(
                id=m.skill_id,
                name=m.name,
                skill_type=SkillType(m.category) if m.category else SkillType.HARD_SKILL,
                normalized_name=m.name.lower().replace(" ", "").replace(".", ""),
                weight=float(m.weight),
                embedding=m.embedding,
            )
            for m in models
        ]

    async def save_skills(self, skills: list[Skill]) -> list[Skill]:
        """Save new skills to the database."""
        if not skills:
            return []

        models = [
            SkillModel(
                name=s.name,
                category=s.skill_type.value,
                weight=s.weight,
                embedding=s.embedding,
            )
            for s in skills
        ]
        self._session.add_all(models)
        await self._session.flush()

        return [
            Skill(
                id=m.skill_id,
                name=m.name,
                skill_type=SkillType(m.category) if m.category else SkillType.HARD_SKILL,
                normalized_name=m.name.lower().replace(" ", "").replace(".", ""),
                weight=float(m.weight),
                embedding=m.embedding,
            )
            for m in models
        ]
