"""ML Engine FastAPI routers."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.dependencies import SessionDep
from src.ml_engine.application.dtos import (
    ClusterDTO,
    GraphResponseDTO,
    SkillsUpdateDTO,
    UserProfileDTO,
)
from src.ml_engine.application.use_cases import (
    EvaluateClusterDiagnosticUseCase,
    ListClustersUseCase,
)
from src.ml_engine.domain.entities import Skill, SkillNature
from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
from src.ml_engine.infrastructure.models import SkillModel
from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository
from src.shared.security import CurrentUserIdDep, OptionalUserIdDep

# Routers
me_router = APIRouter(prefix="/me", tags=["User Portal — Skills"])
market_router = APIRouter(prefix="/market", tags=["Market Intelligence"])
admin_router = APIRouter(prefix="/admin", tags=["Admin Operations"])


# === ME ROUTER ===

@me_router.put(
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
                frequency=0.1,
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
    cluster_repo = SQLClusterRepository(session)
    existing_diagnostics = [profile.primary_affinity, *profile.secondary_affinities]
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
    primary, secondaries, _all_affinities, _ = compute_affinities_and_domains(
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
    from src.ml_engine.application.use_cases import GetMyProfileUseCase
    dto = await GetMyProfileUseCase(repo, cluster_repo).execute(UUID(current_user_id))
    if not dto:
        raise HTTPException(status_code=404, detail="Profile not found after update")
    return dto


@me_router.post(
    "/affinities/{cluster_name}",
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
    repo = SQLUserProfileRepository(session)
    cluster_repo = SQLClusterRepository(session)
    use_case = EvaluateClusterDiagnosticUseCase(repo, cluster_repo)

    dto = await use_case.execute(UUID(current_user_id), cluster_name)
    if not dto:
        raise HTTPException(status_code=404, detail="Failed to evaluate cluster diagnostic.")
    return dto


# === MARKET ROUTER ===

@market_router.get(
    "/clusters",
    response_model=list[ClusterDTO],
    summary="List all discovered tech specialties",
)
async def list_clusters(session: SessionDep) -> list[ClusterDTO]:
    """
    Returns all tech clusters discovered by K-Prototypes clustering.
    """
    use_case = ListClustersUseCase(cluster_repository=SQLClusterRepository(session))
    return await use_case.execute()


@market_router.get(
    "/skills-graph",
    response_model=GraphResponseDTO,
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
    from src.ml_engine.application.use_cases import GetKnowledgeGraphUseCase
    from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository

    use_case = GetKnowledgeGraphUseCase(
        skill_repository=SQLSkillRepository(session),
        profile_repository=SQLUserProfileRepository(session),
    )

    uid = UUID(current_user_id) if current_user_id else None
    return await use_case.execute(user_id=uid, cluster_name=cluster)


# === ADMIN ROUTER ===

@admin_router.post(
    "/skills/normalize",
    summary="Normalize raw skills from job offers",
)
async def normalize_skills(session: SessionDep) -> dict[str, Any]:
    """
    Triggers the Skill Normalization pipeline.
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
