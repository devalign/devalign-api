"""ML Engine FastAPI router."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, UploadFile

from src.dependencies import SessionDep
from src.ml_engine.application.dtos import (
    ClusterDTO,
    ProfileUpdateDTO,
    SkillsUpdateDTO,
    UserProfileDTO,
)
from src.ml_engine.application.use_cases import (
    EvaluateClusterDiagnosticUseCase,
    GetMyProfileUseCase,
    ListClustersUseCase,
    ProfileUserFromCVUseCase,
)
from src.ml_engine.domain.entities import UserProfile
from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
from src.ml_engine.infrastructure.cv_parser import LocalCVParserService
from src.ml_engine.infrastructure.llm_client import get_llm_service
from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository
from src.shared.security import CurrentUserIdDep, CurrentUserPayloadDep, OptionalUserIdDep

router = APIRouter(prefix="/profile", tags=["ML Engine — Profiling"])


@router.post(
    "/analyze",
    response_model=UserProfileDTO,
    status_code=202,
    summary="Analyze CV and generate user profile",
)
async def analyze_cv(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
    file: UploadFile = File(..., description="CV to analyze (PDF or DOCX)"),
) -> UserProfileDTO:
    """
    Upload and analyze a CV to generate a developer profile.

    This endpoint:
    1. Parses the CV document
    2. Generates a semantic embedding
    3. Matches the embedding against known tech clusters
    4. Detects skill gaps vs the primary specialty
    5. Returns a structured profile with alignment scores

    Returns HTTP 202 as processing may take a few seconds.
    """
    from uuid import uuid4

    from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository

    content = await file.read()
    cv_id = uuid4()  # Temporary CV ID — persisted in full flow

    use_case = ProfileUserFromCVUseCase(
        cv_parser=LocalCVParserService(),
        cluster_repository=SQLClusterRepository(session),
        profile_repository=SQLUserProfileRepository(session),
        llm_service=get_llm_service(),
        skill_repository=SQLSkillRepository(session),
    )
    return await use_case.execute(
        user_id=UUID(current_user_id),
        cv_id=cv_id,
        cv_content=content,
        content_type=file.content_type or "application/pdf",
    )


@router.get(
    "/clusters",
    response_model=list[ClusterDTO],
    summary="List all discovered tech specialties",
)
async def list_clusters(session: SessionDep) -> list[ClusterDTO]:
    """
    Returns all tech clusters discovered by K-Prototypes clustering.

    Each cluster represents a real market specialty (e.g., "Backend Cloud-Native Java").
    """
    use_case = ListClustersUseCase(cluster_repository=SQLClusterRepository(session))
    return await use_case.execute()


@router.post(
    "/normalize-skills",
    summary="Normalize raw skills from job offers",
)
async def normalize_skills(session: SessionDep) -> dict[str, Any]:
    """
    Triggers the Skill Normalization pipeline:
    1. Fetches unnormalized job offers from the scraper.
    2. Uses word embeddings to deduplicate and link skills via offer_skills.
    3. Marks the processed offers as normalized.
    """
    from src.ml_engine.application.use_cases import NormalizeSkillsUseCase
    from src.ml_engine.infrastructure.embeddings import get_embedding_service
    from src.ml_engine.infrastructure.job_offer_repository import SQLMLJobOfferRepository
    from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository

    use_case = NormalizeSkillsUseCase(
        job_offer_repo=SQLMLJobOfferRepository(session),
        skill_repo=SQLSkillRepository(session),
        embedding_service=get_embedding_service(),
    )
    return await use_case.execute()


@router.get(
    "/me",
    response_model=UserProfileDTO,
    summary="Get logged-in user's computed profile",
)
async def get_my_profile(
    payload: CurrentUserPayloadDep,
    session: SessionDep,
) -> UserProfileDTO:
    """
    Get the computed profile of the authenticated developer.
    Returns HTTP 404 if no CV has been analyzed yet.
    """
    from datetime import datetime
    from uuid import UUID

    from fastapi import HTTPException
    from sqlalchemy import select

    from src.delivery.infrastructure.models import UserModel
    from src.ml_engine.application.use_cases import GetMyProfileUseCase
    from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository

    user_id = UUID(str(payload.get("sub")))

    # JIT Provisioning fallback
    user_result = await session.execute(
        select(UserModel).where(UserModel.user_id == user_id)
    )
    user_exists = user_result.scalar_one_or_none()

    if not user_exists:
        email = str(payload.get("email") or "")
        user_metadata = payload.get("user_metadata") or {}
        if not isinstance(user_metadata, dict):
            user_metadata = {}
        full_name = str(user_metadata.get("full_name") or user_metadata.get("name") or "") or None
        avatar_url = str(user_metadata.get("avatar_url") or user_metadata.get("picture") or "") or None

        new_user = UserModel(
            user_id=user_id,
            email=email,
            full_name=full_name,
            avatar_url=avatar_url,
            created_at=datetime.utcnow()
        )
        session.add(new_user)
        await session.flush()

    repo = SQLUserProfileRepository(session)
    cluster_repo = SQLClusterRepository(session)
    use_case = GetMyProfileUseCase(repo, cluster_repo)

    dto = await use_case.execute(user_id)
    if not dto:
        raise HTTPException(status_code=404, detail="No profile found. Please upload a CV first.")

    return dto


@router.patch(
    "/me",
    response_model=UserProfileDTO,
    summary="Update manual fields of developer profile",
)
async def update_my_profile(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
    data: ProfileUpdateDTO,
) -> UserProfileDTO:
    """
    Manually update profile details (location, modality, availability, experience lists).
    """
    from uuid import UUID

    from fastapi import HTTPException

    repo = SQLUserProfileRepository(session)
    profile = await repo.get_by_user_id(UUID(current_user_id))
    if not profile:
        raise HTTPException(status_code=404, detail="No profile found. Please upload a CV first.")

    from dataclasses import replace

    kwargs = {}
    for field in [
        "full_name",
        "current_job_role",
        "years_experience",
        "preferred_modality",
        "location",
        "availability",
        "work_experience",
        "education",
        "certifications",
    ]:
        val = getattr(data, field)
        if val is not None:
            kwargs[field] = val

    updated_profile = replace(profile, **kwargs)
    await repo.save(updated_profile)

    from src.ml_engine.application.use_cases import GetMyProfileUseCase
    from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository

    dto = await GetMyProfileUseCase(repo, SQLClusterRepository(session)).execute(
        UUID(current_user_id)
    )
    if not dto:
        raise HTTPException(status_code=404, detail="Profile not found after update")
    return dto


@router.put(
    "/skills",
    response_model=UserProfileDTO,
    summary="Update manual skills for developer profile",
)
async def update_my_skills(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
    data: SkillsUpdateDTO,
) -> UserProfileDTO:
    """
    Overwrite the skills and gaps associated with the latest diagnostic.
    """
    from uuid import UUID

    from fastapi import HTTPException
    from sqlalchemy import select

    from src.ml_engine.domain.entities import Skill, SkillNature
    from src.ml_engine.infrastructure.models import SkillModel

    repo = SQLUserProfileRepository(session)
    profile = await repo.get_by_user_id(UUID(current_user_id))
    if not profile:
        raise HTTPException(status_code=404, detail="No profile found. Please upload a CV first.")

    # 1. Enriquecer Habilidades con la DB
    skill_names = [s.name for s in data.skills]
    db_skills = {}
    if skill_names:
        db_skills_result = await session.execute(
            select(SkillModel).where(SkillModel.name.in_(skill_names))
        )
        db_skills = {m.name.lower(): m for m in db_skills_result.scalars().all()}

    detected_skills = []
    for s in data.skills:
        # Determine nature
        nature_val = SkillNature.TECH
        if s.skill_type:
            try:
                nature_val = SkillNature(s.skill_type)
            except ValueError:
                st_lower = s.skill_type.lower()
                if "soft" in st_lower:
                    nature_val = SkillNature.SOFT
                elif "concept" in st_lower:
                    nature_val = SkillNature.CONCEPT

        norm_name = s.name.lower()
        db_skill = db_skills.get(norm_name)
        if db_skill:
            skill_entity = Skill(
                id=db_skill.skill_id,
                name=db_skill.name,
                nature=SkillNature(db_skill.nature) if db_skill.nature else nature_val,
                normalized_name=db_skill.name.lower().replace(" ", "").replace(".", ""),
                weight=float(db_skill.weight),
                frequency=0.1,  # default placeholder frequency
                domain_tags=db_skill.domain_tags or [],
                core_domains=db_skill.core_domains or [],
            )
        else:
            skill_entity = Skill(
                name=s.name,
                nature=nature_val,
                normalized_name=s.name.lower().replace(" ", "").replace(".", ""),
            )
        detected_skills.append(skill_entity)

    # 2. Recalcular Diagnóstico Activo y Evaluados
    from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
    cluster_repo = SQLClusterRepository(session)

    existing_diagnostics = [profile.primary_affinity] + profile.secondary_affinities
    existing_diagnostics = [a for a in existing_diagnostics if a.cluster_name != "Sin Diagnóstico"]

    clusters_to_eval = []
    for diag in existing_diagnostics:
        cluster = await cluster_repo.get_by_id(diag.cluster_id)
        if cluster:
            clusters_to_eval.append(cluster)

    if not clusters_to_eval:
        active_clusters = await cluster_repo.get_all_active()
        clusters_to_eval = [c for c in active_clusters if c.centroid_skills]

    # Recalcular affinities
    from src.ml_engine.application.use_cases import compute_affinities_and_domains
    primary, secondaries, all_affinities, _ = compute_affinities_and_domains(
        detected_skills, clusters_to_eval
    )

    if not primary:
        from uuid import uuid4

        from src.ml_engine.domain.entities import ClusterAffinity
        primary = ClusterAffinity(
            cluster_id=uuid4(),
            cluster_name="Sin Diagnóstico",
            affinity_score=0.0,
            is_primary=True,
        )
        secondaries = []

    # 3. Persistir el perfil con los diagnósticos actualizados
    from dataclasses import replace
    updated_profile = replace(
        profile,
        detected_skills=detected_skills,
        primary_affinity=primary,
        secondary_affinities=secondaries,
        skill_gaps=primary.skill_gaps,
    )
    await repo.save(updated_profile)

    # 4. Devolver perfil actualizado
    dto = await GetMyProfileUseCase(repo, cluster_repo).execute(UUID(current_user_id))
    if not dto:
        raise HTTPException(status_code=404, detail="Profile not found after update")
    return dto


@router.post(
    "/evaluate-cluster/{cluster_name}",
    response_model=UserProfileDTO,
    summary="Evaluate user's CV/skills against a specific tech cluster",
)
async def evaluate_cluster_diagnostic(
    cluster_name: str,
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> UserProfileDTO:
    """
    Evaluate the user's current profile/skills against a specific tech cluster,
    saving the diagnostic results under secondary_affinities.
    """
    from uuid import UUID

    from fastapi import HTTPException

    repo = SQLUserProfileRepository(session)
    cluster_repo = SQLClusterRepository(session)
    use_case = EvaluateClusterDiagnosticUseCase(repo, cluster_repo)

    dto = await use_case.execute(UUID(current_user_id), cluster_name)
    if not dto:
        raise HTTPException(status_code=404, detail="Failed to evaluate cluster diagnostic.")
    return dto


@router.get(
    "/skills-graph",
    summary="Get the knowledge graph of skills and their relations",
)
async def get_skills_graph(
    session: SessionDep,
    current_user_id: OptionalUserIdDep = None,
    cluster: str | None = None,
) -> Any:
    """
    Returns the complete knowledge graph of skills, including explicit relationships
    and implicit domain connections. If authenticated, highlights user's acquired skills and gaps.
    """
    from uuid import UUID

    from src.ml_engine.application.use_cases import GetKnowledgeGraphUseCase
    from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository
    from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository

    use_case = GetKnowledgeGraphUseCase(
        skill_repository=SQLSkillRepository(session),
        profile_repository=SQLUserProfileRepository(session),
    )

    uid = UUID(current_user_id) if current_user_id else None
    return await use_case.execute(user_id=uid, cluster_name=cluster)


def _map_entity_to_dto(profile: UserProfile) -> UserProfileDTO:
    from src.ml_engine.application.dtos import ClusterAffinityDTO, SkillDTO

    return UserProfileDTO(
        user_id=profile.user_id,
        cv_id=profile.cv_id,
        seniority=profile.seniority.value,
        primary_specialty=profile.primary_specialty,
        alignment_score=profile.alignment_score,
        secondary_affinities=[
            ClusterAffinityDTO(
                cluster_id=a.cluster_id,
                cluster_name=a.cluster_name,
                affinity_score=a.affinity_score,
                is_primary=False,
                ai_insight=a.ai_insight,
            )
            for a in profile.secondary_affinities
        ],
        detected_skills=[
            SkillDTO(name=s.name, skill_type=s.nature.value, market_importance="consolidated")
            for s in profile.detected_skills
        ],
        skill_gaps=[
            SkillDTO(
                name=g.skill.name,
                skill_type=g.skill.nature.value,
                market_importance=g.market_importance,
            )
            for g in profile.skill_gaps
        ],
        full_name=profile.full_name,
        current_job_role=profile.current_job_role,
        years_experience=profile.years_experience,
        preferred_modality=profile.preferred_modality,
        location=profile.location,
        availability=profile.availability,
        work_experience=profile.work_experience,
        education=profile.education,
        certifications=profile.certifications,
        message="Profile processed successfully",
    )
