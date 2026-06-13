"""ML Engine FastAPI router."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, UploadFile

from src.dependencies import SessionDep
from src.genai.infrastructure.langchain_chain import get_llm_service
from src.ml_engine.application.dtos import (
    ClusterDTO,
    ProfileUpdateDTO,
    SkillsUpdateDTO,
    UserProfileDTO,
)
from src.ml_engine.application.use_cases import ListClustersUseCase, ProfileUserFromCVUseCase
from src.ml_engine.domain.entities import UserProfile
from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
from src.ml_engine.infrastructure.cv_parser import LocalCVParserService
from src.ml_engine.infrastructure.embeddings import get_embedding_service
from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository
from src.shared.security import CurrentUserIdDep

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
        embedding_service=get_embedding_service(),
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
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> UserProfileDTO:
    """
    Get the computed profile of the authenticated developer.
    Returns HTTP 404 if no CV has been analyzed yet.
    """
    from uuid import UUID

    from fastapi import HTTPException

    repo = SQLUserProfileRepository(session)
    profile = await repo.get_by_user_id(UUID(current_user_id))
    if not profile:
        raise HTTPException(status_code=404, detail="No profile found. Please upload a CV first.")

    return _map_entity_to_dto(profile)


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

    return _map_entity_to_dto(updated_profile)


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

    from src.ml_engine.domain.entities import Skill, SkillGap, SkillType

    repo = SQLUserProfileRepository(session)
    profile = await repo.get_by_user_id(UUID(current_user_id))
    if not profile:
        raise HTTPException(status_code=404, detail="No profile found. Please upload a CV first.")

    detected_skills = []
    skill_gaps = []
    for s in data.skills:
        skill_entity = Skill(
            name=s.name,
            skill_type=SkillType(s.skill_type) if s.skill_type else SkillType.HARD_SKILL,
            normalized_name=s.name.lower().replace(" ", "").replace(".", ""),
        )
        if s.market_importance == "consolidated":
            detected_skills.append(skill_entity)
        else:
            skill_gaps.append(
                SkillGap(
                    skill=skill_entity,
                    market_importance=s.market_importance or "high",
                )
            )

    from dataclasses import replace

    updated_profile = replace(profile, detected_skills=detected_skills, skill_gaps=skill_gaps)
    await repo.save(updated_profile)

    return _map_entity_to_dto(updated_profile)


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
            )
            for a in profile.secondary_affinities
        ],
        detected_skills=[
            SkillDTO(name=s.name, skill_type=s.skill_type.value, market_importance="consolidated")
            for s in profile.detected_skills
        ],
        skill_gaps=[
            SkillDTO(
                name=g.skill.name,
                skill_type=g.skill.skill_type.value,
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
