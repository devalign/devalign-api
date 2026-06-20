"""SQLAlchemy implementation of UserProfileRepository."""

from uuid import UUID, uuid4

from sqlalchemy import select
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

        # 2. Save Diagnostic
        diagnostic_model = DiagnosticModel(
            diagnostic_id=uuid4(),
            profile_id=profile_model.profile_id,
            detected_cluster_id=profile.primary_affinity.cluster_id,
            affinity_score=profile.primary_affinity.affinity_score,
        )
        self._session.add(diagnostic_model)
        await self._session.flush()

        # 3. Save Diagnostic Skills (consolidated and gaps)
        skill_names = [s.name for s in profile.detected_skills] + [
            g.skill.name for g in profile.skill_gaps
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
                # Find the skill in profile.detected_skills or profile.skill_gaps to get the correct properties
                nature = SkillNature.TECH.value
                weight = 1.0
                domain_tags = []
                for s in profile.detected_skills:
                    if s.name == skill_name:
                        nature = s.nature.value
                        weight = s.weight
                        domain_tags = s.domain_tags
                        break
                for g in profile.skill_gaps:
                    if g.skill.name == skill_name:
                        nature = g.skill.nature.value
                        weight = g.skill.weight
                        domain_tags = g.skill.domain_tags
                        break

                new_skill_model = SkillModel(
                    skill_id=uuid4(),
                    name=skill_name,
                    nature=nature,
                    weight=weight,
                    domain_tags=domain_tags,
                )
                self._session.add(new_skill_model)
                db_skills[norm_name] = new_skill_model

        await self._session.flush()

        # Insert DiagnosticSkills
        for skill in profile.detected_skills:
            skill_model = db_skills[skill.name.lower()]
            diag_skill = DiagnosticSkillModel(
                diagnostic_skill_id=uuid4(),
                diagnostic_id=diagnostic_model.diagnostic_id,
                skill_id=skill_model.skill_id,
                skill_status="consolidated",
                importance_score=skill.frequency,
            )
            self._session.add(diag_skill)

        for gap in profile.skill_gaps:
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
        return profile

    async def get_by_user_id(self, user_id: UUID) -> UserProfile | None:
        # Fetch profile
        result = await self._session.execute(
            select(ProfileModel).where(ProfileModel.user_id == user_id)
        )
        profile_model = result.scalar_one_or_none()
        if not profile_model:
            return None

        # Fetch latest diagnostic with its details
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
            .limit(1)
        )
        diagnostic_model = diag_result.scalar_one_or_none()

        if not diagnostic_model:
            return UserProfile(
                user_id=profile_model.user_id,
                cv_id=profile_model.cv_id,
                embedding=[],
                detected_skills=[],
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

        # Parse skills and gaps
        detected_skills = []
        skill_gaps = []
        for ds in diagnostic_model.diagnostic_skills:
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

        primary_affinity = ClusterAffinity(
            cluster_id=diagnostic_model.detected_cluster_id,
            cluster_name=diagnostic_model.detected_cluster.name
            if diagnostic_model.detected_cluster
            else "Unknown",
            affinity_score=float(diagnostic_model.affinity_score),
            is_primary=True,
        )

        return UserProfile(
            user_id=profile_model.user_id,
            cv_id=profile_model.cv_id,
            embedding=list(profile_model.cv_embedding)
            if profile_model.cv_embedding is not None
            else [],
            detected_skills=detected_skills,
            seniority=SeniorityLevel.MID,
            primary_affinity=primary_affinity,
            secondary_affinities=[],
            skill_gaps=skill_gaps,
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
