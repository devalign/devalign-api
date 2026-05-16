"""ML Engine FastAPI router."""

from uuid import UUID

from fastapi import APIRouter, File, UploadFile

from src.dependencies import SessionDep
from src.ml_engine.application.dtos import ClusterDTO, UserProfileDTO
from src.ml_engine.application.use_cases import ListClustersUseCase, ProfileUserFromCVUseCase
from src.ml_engine.infrastructure.cv_parser import LocalCVParserService
from src.ml_engine.infrastructure.embeddings import get_embedding_service
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

    content = await file.read()
    cv_id = uuid4()  # Temporary CV ID — persisted in full flow

    use_case = ProfileUserFromCVUseCase(
        cv_parser=LocalCVParserService(),
        embedding_service=get_embedding_service(),
        cluster_repository=...,  # TODO: inject real repo
        profile_repository=...,  # TODO: inject real repo
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
    use_case = ListClustersUseCase(cluster_repository=...)  # TODO: inject real repo
    return await use_case.execute()
