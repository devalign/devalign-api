"""SQLAlchemy implementation of SkillRepository."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.ml_engine.domain.entities import Skill, SkillNature, SkillRelation, SkillRelationType
from src.ml_engine.domain.ports import SkillRepository
from src.ml_engine.infrastructure.models import SkillAliasModel, SkillModel


class SQLSkillRepository(SkillRepository):
    """SQLAlchemy implementation of SkillRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_all_skills(self) -> list[Skill]:
        """Retrieve all canonical skills for similarity checking."""
        result = await self._session.execute(
            select(SkillModel).options(
                selectinload(SkillModel.aliases), selectinload(SkillModel.outgoing_relations)
            )
        )
        models = result.scalars().all()
        skills = []
        for m in models:
            aliases = [a.alias_name for a in m.aliases]
            relations = [
                SkillRelation(
                    target_skill_id=r.target_skill_id,
                    target_skill_name="",  # We'd need a join to get the name if required, or lazy load
                    relation_type=SkillRelationType(r.relation_type),
                )
                for r in m.outgoing_relations
            ]
            skills.append(
                Skill(
                    id=m.skill_id,
                    name=m.name,
                    nature=SkillNature(m.nature) if m.nature else SkillNature.TECH,
                    normalized_name=m.name.lower().replace(" ", "").replace(".", ""),
                    domain_tags=m.domain_tags if m.domain_tags else [],
                    aliases=aliases,
                    relations=relations,
                    weight=float(m.weight),
                    embedding=m.embedding,
                )
            )
        return skills

    async def save_skills(self, skills: list[Skill]) -> list[Skill]:
        """Save new skills to the database."""
        if not skills:
            return []

        models = []
        for s in skills:
            model = SkillModel(
                name=s.name,
                nature=s.nature.value,
                domain_tags=s.domain_tags,
                weight=s.weight,
                embedding=s.embedding,
            )
            if s.aliases:
                for alias in s.aliases:
                    model.aliases.append(SkillAliasModel(alias_name=alias))

            # Note: Relations are usually updated separately once both nodes exist
            models.append(model)

        self._session.add_all(models)
        await self._session.flush()

        # Simple return mapping
        saved_skills = []
        for m, s in zip(models, skills, strict=False):
            saved_skills.append(
                Skill(
                    id=m.skill_id,
                    name=m.name,
                    nature=SkillNature(m.nature) if m.nature else SkillNature.TECH,
                    normalized_name=m.name.lower().replace(" ", "").replace(".", ""),
                    domain_tags=m.domain_tags if m.domain_tags else [],
                    aliases=[a.alias_name for a in m.aliases],
                    relations=s.relations,  # Keep original relations in memory
                    weight=float(m.weight),
                    embedding=m.embedding,
                )
            )
        return saved_skills
