"""SQLAlchemy implementation of SkillRepository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from src.ml_engine.domain.entities import Skill, SkillNature, SkillRelation, SkillRelationType
from src.ml_engine.domain.ports import SkillRepository
from src.ml_engine.infrastructure.models import SkillAliasModel, SkillModel, SkillRelationModel


def _model_to_skill(m: SkillModel, name_map: dict[UUID, str] | None = None) -> Skill:
    """Convert a SkillModel ORM instance to a Skill domain entity.

    Args:
        m: The SQLAlchemy ORM model.
        name_map: Optional pre-built map of skill_id -> skill_name for resolving
                  relation target names without extra queries.

    Returns:
        A fully constructed Skill domain entity.
    """
    aliases = [a.alias_name for a in m.aliases]
    relations = [
        SkillRelation(
            target_skill_id=r.target_skill_id,
            target_skill_name=name_map.get(r.target_skill_id, "") if name_map else "",
            relation_type=SkillRelationType(r.relation_type),
        )
        for r in m.outgoing_relations
    ]
    return Skill(
        id=m.skill_id,
        name=m.name,
        nature=SkillNature(m.nature) if m.nature else SkillNature.TECH,
        normalized_name=m.name.lower().replace(" ", "").replace(".", ""),
        domain_tags=m.domain_tags if m.domain_tags else [],
        core_domains=m.core_domains if m.core_domains else [],
        aliases=aliases,
        relations=relations,
        weight=float(m.weight),
        embedding=m.embedding,
    )


class SQLSkillRepository(SkillRepository):
    """SQLAlchemy implementation of SkillRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_all_skills(self) -> list[Skill]:
        """Retrieve all canonical skills with their aliases and outgoing relations.

        Uses a single query with eagerly loaded aliases and relations to avoid
        N+1 query problems. Target skill names in relations are resolved via
        an in-memory name map built from the same query results.

        Returns:
            A list of Skill domain entities.
        """
        result = await self._session.execute(
            select(SkillModel).options(
                selectinload(SkillModel.aliases),
                selectinload(SkillModel.outgoing_relations),
            )
        )
        models = result.scalars().all()
        # Build a name map to resolve target_skill_name without extra queries
        name_map: dict[UUID, str] = {m.skill_id: m.name for m in models}
        return [_model_to_skill(m, name_map) for m in models]

    async def get_skill_graph(self) -> dict[UUID, Skill]:
        """Load the full skill graph into memory as a {skill_id: Skill} dict.

        This is the preferred entry point for the upward inference algorithm,
        as it resolves all relation names in a single DB round-trip and
        returns a structure that allows O(1) parent lookups during traversal.

        Returns:
            A dict mapping each skill's UUID to its Skill domain entity,
            with all outgoing relations (and their target names) resolved.
        """
        skills = await self.get_all_skills()
        return {s.id: s for s in skills if s.id}

    async def save_skills(self, skills: list[Skill]) -> list[Skill]:
        """Save new skills to the database.

        Relations are NOT persisted here — call add_relations separately once
        all skill IDs are known.

        Args:
            skills: List of Skill domain entities to persist.

        Returns:
            The same skills with their database-assigned UUIDs populated.
        """
        if not skills:
            return []

        models = []
        for s in skills:
            model = SkillModel(
                name=s.name,
                nature=s.nature.value,
                domain_tags=s.domain_tags,
                core_domains=s.core_domains,
                weight=s.weight,
                embedding=s.embedding,
            )
            if s.aliases:
                for alias in s.aliases:
                    model.aliases.append(SkillAliasModel(alias_name=alias))

            # Note: Relations are persisted separately via add_relations()
            models.append(model)

        self._session.add_all(models)
        await self._session.flush()

        # Map saved models back to domain entities with populated IDs
        saved_skills = []
        for m, s in zip(models, skills, strict=False):
            saved_skills.append(
                Skill(
                    id=m.skill_id,
                    name=m.name,
                    nature=SkillNature(m.nature) if m.nature else SkillNature.TECH,
                    normalized_name=m.name.lower().replace(" ", "").replace(".", ""),
                    domain_tags=m.domain_tags if m.domain_tags else [],
                    core_domains=m.core_domains if m.core_domains else [],
                    aliases=[a.alias_name for a in m.aliases],
                    relations=s.relations,  # Keep original domain relations in memory
                    weight=float(m.weight),
                    embedding=m.embedding,
                )
            )
        return saved_skills

    async def add_relations(
        self, relations: list[tuple[UUID, UUID, SkillRelationType]]
    ) -> None:
        """Persist skill-to-skill knowledge graph edges.

        Skips any edge where source_id == target_id or where the exact
        (source, target, type) triplet already exists, making this method
        safe to call multiple times (idempotent).

        Args:
            relations: A list of (source_skill_id, target_skill_id, relation_type) tuples.
        """
        if not relations:
            return

        # Fetch existing edges to avoid duplicates
        existing_result = await self._session.execute(select(SkillRelationModel))
        existing = {
            (r.source_skill_id, r.target_skill_id, r.relation_type)
            for r in existing_result.scalars().all()
        }

        new_models = []
        for source_id, target_id, rel_type in relations:
            if source_id == target_id:
                continue  # Skip self-loops
            key = (source_id, target_id, rel_type.value)
            if key not in existing:
                new_models.append(
                    SkillRelationModel(
                        source_skill_id=source_id,
                        target_skill_id=target_id,
                        relation_type=rel_type.value,
                    )
                )

        if new_models:
            self._session.add_all(new_models)
            await self._session.flush()
