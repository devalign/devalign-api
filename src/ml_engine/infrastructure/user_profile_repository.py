"""SQLAlchemy implementation of UserProfileRepository."""

from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.ml_engine.domain.entities import (
    ClusterAffinity,
    SeniorityLevel,
    Skill,
    SkillGap,
    SkillNature,
    UserProfile,
)
from src.ml_engine.domain.ports import UserProfileRepository
from src.ml_engine.infrastructure.models import (
    DiagnosticModel,
    DiagnosticSkillModel,
    ProfileModel,
    ProfileSkillModel,
    SkillModel,
)


class SQLUserProfileRepository(UserProfileRepository):
    """SQLAlchemy implementation of UserProfileRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, profile: UserProfile) -> UserProfile:
        # 1. Check if profile exists
        result = await self._session.execute(
            select(ProfileModel).where(ProfileModel.user_id == profile.user_id)
        )
        profile_model = result.scalar_one_or_none()

        if not profile_model:
            profile_model = ProfileModel(
                profile_id=uuid4(),
                user_id=profile.user_id,
            )
            self._session.add(profile_model)

        # Update fields
        profile_model.full_name = profile.full_name
        profile_model.current_job_role = profile.current_job_role
        profile_model.years_experience = profile.years_experience
        profile_model.preferred_modality = profile.preferred_modality
        profile_model.location = profile.location
        profile_model.availability = profile.availability
        profile_model.work_experience = profile.work_experience
        profile_model.education = profile.education
        profile_model.certifications = profile.certifications
        profile_model.cv_embedding = profile.embedding
        profile_model.cv_id = profile.cv_id

        await self._session.flush()

        # 2. Save Diagnostics
        # Delete existing diagnostics to prevent database bloat and timeouts
        await self._session.execute(
            delete(DiagnosticModel).where(DiagnosticModel.profile_id == profile_model.profile_id)
        )
        await self._session.flush()

        all_affinities = [profile.primary_affinity, *profile.secondary_affinities]
        all_affinities = [a for a in all_affinities if a.cluster_name != "Sin Diagnóstico"]

        for affinity in all_affinities:
            diagnostic_model = DiagnosticModel(
                diagnostic_id=uuid4(),
                profile_id=profile_model.profile_id,
                detected_cluster_id=affinity.cluster_id,
                affinity_score=affinity.affinity_score,
            )
            self._session.add(diagnostic_model)
            await self._session.flush()

            # 3. Save Diagnostic Skills (consolidated and gaps) for this diagnostic
            skill_names = [s.name for s in affinity.detected_skills] + [
                g.skill.name for g in affinity.skill_gaps
            ]

            db_skills = {}
            if skill_names:
                db_skills_result = await self._session.execute(
                    select(SkillModel).where(SkillModel.name.in_(skill_names))
                )
                db_skills = {m.name.lower(): m for m in db_skills_result.scalars().all()}

            # Insert missing skills
            for skill_name in skill_names:
                norm_name = skill_name.lower()
                if norm_name not in db_skills:
                    nature = SkillNature.TECH.value
                    weight = 1.0
                    domain_tags = []
                    core_domains = []
                    for s in affinity.detected_skills:
                        if s.name == skill_name:
                            nature = s.nature.value
                            weight = s.weight
                            domain_tags = s.domain_tags
                            core_domains = s.core_domains
                            break
                    for g in affinity.skill_gaps:
                        if g.skill.name == skill_name:
                            nature = g.skill.nature.value
                            weight = g.skill.weight
                            domain_tags = g.skill.domain_tags
                            core_domains = g.skill.core_domains
                            break

                    new_skill_model = SkillModel(
                        skill_id=uuid4(),
                        name=skill_name,
                        nature=nature,
                        weight=weight,
                        domain_tags=domain_tags,
                        core_domains=core_domains,
                    )
                    self._session.add(new_skill_model)
                    db_skills[norm_name] = new_skill_model

            await self._session.flush()

            # Insert DiagnosticSkills
            for skill in affinity.detected_skills:
                skill_model = db_skills[skill.name.lower()]
                diag_skill = DiagnosticSkillModel(
                    diagnostic_skill_id=uuid4(),
                    diagnostic_id=diagnostic_model.diagnostic_id,
                    skill_id=skill_model.skill_id,
                    skill_status="consolidated",
                    importance_score=skill.frequency,
                )
                self._session.add(diag_skill)

            for gap in affinity.skill_gaps:
                skill_model = db_skills[gap.skill.name.lower()]
                diag_skill = DiagnosticSkillModel(
                    diagnostic_skill_id=uuid4(),
                    diagnostic_id=diagnostic_model.diagnostic_id,
                    skill_id=skill_model.skill_id,
                    skill_status="gap",
                    importance_score=gap.skill.frequency,
                )
                self._session.add(diag_skill)

            await self._session.flush()

        # 4. Save Global Profile Skills (profile_skills)
        profile_skill_names = [s.name for s in profile.detected_skills]
        db_profile_skills = {}
        if profile_skill_names:
            db_profile_skills_result = await self._session.execute(
                select(SkillModel).where(SkillModel.name.in_(profile_skill_names))
            )
            db_profile_skills = {m.name.lower(): m for m in db_profile_skills_result.scalars().all()}

        for skill in profile.detected_skills:
            norm_name = skill.name.lower()
            if norm_name not in db_profile_skills:
                new_skill_model = SkillModel(
                    skill_id=uuid4(),
                    name=skill.name,
                    nature=skill.nature.value if skill.nature else SkillNature.TECH.value,
                    weight=skill.weight,
                    domain_tags=skill.domain_tags or [],
                    core_domains=skill.core_domains or [],
                )
                self._session.add(new_skill_model)
                db_profile_skills[norm_name] = new_skill_model

        await self._session.flush()

        # Delete existing profile_skills relations
        await self._session.execute(
            delete(ProfileSkillModel).where(ProfileSkillModel.profile_id == profile_model.profile_id)
        )
        await self._session.flush()

        # Insert new profile_skills relations
        for skill in profile.detected_skills:
            skill_model = db_profile_skills[skill.name.lower()]
            profile_skill_rel = ProfileSkillModel(
                profile_skill_id=uuid4(),
                profile_id=profile_model.profile_id,
                skill_id=skill_model.skill_id,
            )
            self._session.add(profile_skill_rel)

        await self._session.flush()

        return profile

    async def get_by_user_id(self, user_id: UUID) -> UserProfile | None:
        # Fetch profile with profile_skills relation
        result = await self._session.execute(
            select(ProfileModel)
            .where(ProfileModel.user_id == user_id)
            .options(
                selectinload(ProfileModel.profile_skills).selectinload(
                    ProfileSkillModel.skill
                )
            )
        )
        profile_model = result.scalar_one_or_none()
        if not profile_model:
            return None

        # Map global profile_skills
        global_detected_skills = []
        if profile_model.profile_skills:
            for ps in profile_model.profile_skills:
                if not ps.skill:
                    continue
                skill_entity = Skill(
                    id=ps.skill.skill_id,
                    name=ps.skill.name,
                    nature=SkillNature(ps.skill.nature) if ps.skill.nature else SkillNature.TECH,
                    normalized_name=ps.skill.name.lower().replace(" ", "").replace(".", ""),
                    weight=float(ps.skill.weight),
                    frequency=1.0,
                    domain_tags=ps.skill.domain_tags or [],
                    core_domains=ps.skill.core_domains or [],
                )
                global_detected_skills.append(skill_entity)

        # Fetch all diagnostics for the profile, sorted by created_at desc
        diag_result = await self._session.execute(
            select(DiagnosticModel)
            .where(DiagnosticModel.profile_id == profile_model.profile_id)
            .order_by(DiagnosticModel.created_at.desc())
            .options(
                selectinload(DiagnosticModel.detected_cluster),
                selectinload(DiagnosticModel.diagnostic_skills).selectinload(
                    DiagnosticSkillModel.skill
                ),
            )
        )
        all_diagnostics = diag_result.scalars().all()

        if not all_diagnostics:
            return UserProfile(
                user_id=profile_model.user_id,
                cv_id=profile_model.cv_id,
                embedding=[],
                detected_skills=global_detected_skills,
                seniority=SeniorityLevel.JUNIOR,
                primary_affinity=ClusterAffinity(
                    cluster_id=uuid4(),
                    cluster_name="Sin Diagnóstico",
                    affinity_score=0.0,
                    is_primary=True,
                ),
                secondary_affinities=[],
                skill_gaps=[],
                full_name=profile_model.full_name,
                current_job_role=profile_model.current_job_role,
                years_experience=profile_model.years_experience,
                preferred_modality=profile_model.preferred_modality,
                location=profile_model.location,
                availability=profile_model.availability,
                work_experience=profile_model.work_experience,
                education=profile_model.education,
                certifications=profile_model.certifications,
            )

        # Deduplicate: only keep the latest diagnostic for each cluster
        seen_clusters = set()
        unique_diagnostics = []
        for dm in all_diagnostics:
            if dm.detected_cluster_id not in seen_clusters:
                seen_clusters.add(dm.detected_cluster_id)
                unique_diagnostics.append(dm)

        # Map them all to ClusterAffinity objects
        affinities = []
        for dm in unique_diagnostics:
            detected_skills = []
            skill_gaps = []
            for ds in dm.diagnostic_skills:
                if not ds.skill:
                    continue
                skill_entity = Skill(
                    id=ds.skill.skill_id,
                    name=ds.skill.name,
                    nature=SkillNature(ds.skill.nature) if ds.skill.nature else SkillNature.TECH,
                    normalized_name=ds.skill.name.lower().replace(" ", "").replace(".", ""),
                    weight=float(ds.skill.weight),
                    frequency=float(ds.importance_score) if ds.importance_score is not None else 1.0,
                    domain_tags=ds.skill.domain_tags or [],
                    core_domains=ds.skill.core_domains or [],
                )
                if ds.skill_status == "consolidated":
                    detected_skills.append(skill_entity)
                elif ds.skill_status == "gap":
                    priority = skill_entity.weight * skill_entity.frequency
                    if priority >= 2.0:
                        importance = "critical"
                    elif priority >= 1.0:
                        importance = "high"
                    else:
                        importance = "medium"

                    skill_gaps.append(
                        SkillGap(
                            skill=skill_entity,
                            market_importance=importance,
                        )
                    )

            affinities.append(
                ClusterAffinity(
                    cluster_id=dm.detected_cluster_id,
                    cluster_name=dm.detected_cluster.name if dm.detected_cluster else "Unknown",
                    affinity_score=float(dm.affinity_score),
                    is_primary=False,
                    market_insights=dm.detected_cluster.market_insights if dm.detected_cluster else None,
                    compatible_roles=dm.detected_cluster.compatible_roles if dm.detected_cluster else None,
                    detected_skills=detected_skills,
                    skill_gaps=skill_gaps,
                )
            )

        # Sort all affinities by affinity_score descending
        affinities.sort(key=lambda a: a.affinity_score, reverse=True)

        # The primary affinity is the one with the highest affinity score
        primary_affinity = affinities[0]
        # Mark it as primary
        primary_affinity = ClusterAffinity(
            cluster_id=primary_affinity.cluster_id,
            cluster_name=primary_affinity.cluster_name,
            affinity_score=primary_affinity.affinity_score,
            is_primary=True,
            market_insights=primary_affinity.market_insights,
            compatible_roles=primary_affinity.compatible_roles,
            detected_skills=primary_affinity.detected_skills,
            skill_gaps=primary_affinity.skill_gaps,
        )

        secondary_affinities = affinities[1:]

        return UserProfile(
            user_id=profile_model.user_id,
            cv_id=profile_model.cv_id,
            embedding=list(profile_model.cv_embedding)
            if profile_model.cv_embedding is not None
            else [],
            detected_skills=global_detected_skills if global_detected_skills else primary_affinity.detected_skills,
            seniority=SeniorityLevel.MID,
            primary_affinity=primary_affinity,
            secondary_affinities=secondary_affinities,
            skill_gaps=primary_affinity.skill_gaps,
            full_name=profile_model.full_name,
            current_job_role=profile_model.current_job_role,
            years_experience=profile_model.years_experience,
            preferred_modality=profile_model.preferred_modality,
            location=profile_model.location,
            availability=profile_model.availability,
            work_experience=profile_model.work_experience,
            education=profile_model.education,
            certifications=profile_model.certifications,
        )

    async def delete_by_user_id(self, user_id: UUID) -> None:
        await self._session.execute(
            delete(ProfileModel).where(ProfileModel.user_id == user_id)
        )
        await self._session.flush()

