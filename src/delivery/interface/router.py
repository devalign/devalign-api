"""Delivery module API router."""

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, UploadFile

from src.delivery.application.dtos import CVListDTO, CVUploadResultDTO, UserProfileDTO
from src.delivery.application.use_cases import (
    GetCurrentUserUseCase,
    ListUserCVsUseCase,
    UploadCVUseCase,
)
from src.delivery.infrastructure.repository import SQLAlchemyCVRepository, SQLAlchemyUserRepository
from src.delivery.infrastructure.supabase_storage import SupabaseStorageService
from src.dependencies import SessionDep
from src.shared.security import CurrentUserIdDep, CurrentUserPayloadDep
from src.shared.supabase_client import get_supabase_admin_client

router = APIRouter(prefix="/users", tags=["Users & CV"])


def _get_upload_cv_use_case(session: SessionDep) -> UploadCVUseCase:
    """Build UploadCVUseCase with its dependencies."""
    return UploadCVUseCase(
        cv_repository=SQLAlchemyCVRepository(session),
        storage_service=SupabaseStorageService(get_supabase_admin_client()),
    )


def _get_list_cvs_use_case(session: SessionDep) -> ListUserCVsUseCase:
    return ListUserCVsUseCase(
        cv_repository=SQLAlchemyCVRepository(session),
        storage_service=SupabaseStorageService(get_supabase_admin_client()),
    )


@router.get("/me", response_model=UserProfileDTO, summary="Get current user profile")
async def get_me(
    payload: CurrentUserPayloadDep,
    session: SessionDep,
) -> UserProfileDTO:
    """
    Returns the profile of the currently authenticated user.

    Requires a valid Supabase JWT Bearer token.
    If the user does not exist locally, they will be provisioned JIT (Just-In-Time)
    using metadata extracted from the JWT payload.
    """
    user_id = UUID(str(payload.get("sub")))
    email = str(payload.get("email") or "")

    # Extract identity metadata safely
    user_metadata = payload.get("user_metadata")
    if not isinstance(user_metadata, dict):
        user_metadata = {}

    full_name = str(user_metadata.get("full_name") or user_metadata.get("name") or "") or None
    avatar_url = str(user_metadata.get("avatar_url") or user_metadata.get("picture") or "") or None

    use_case = GetCurrentUserUseCase(SQLAlchemyUserRepository(session))
    return await use_case.execute(
        user_id=user_id,
        email=email,
        full_name=full_name,
        avatar_url=avatar_url,
    )


@router.post(
    "/me/cv", response_model=CVUploadResultDTO, status_code=201, summary="Upload CV document"
)
async def upload_cv(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CV document (PDF or DOCX, max 5MB)"),
) -> CVUploadResultDTO:
    """
    Upload a CV document for processing.

    - Accepts PDF and DOCX formats
    - Maximum file size: 5MB
    - The CV will be stored securely and used for profile analysis
    """
    content = await file.read()
    use_case = _get_upload_cv_use_case(session)
    result = await use_case.execute(
        user_id=UUID(current_user_id),
        filename=file.filename or "cv",
        content=content,
        content_type=file.content_type or "application/octet-stream",
    )

    # Queue background task to run profile analysis
    background_tasks.add_task(
        run_profile_analysis_task,
        user_id=result.user_id,
        cv_id=result.cv_id,
        content=content,
        content_type=file.content_type or "application/octet-stream",
    )

    return result


async def run_profile_analysis_task(
    user_id: UUID,
    cv_id: UUID,
    content: bytes,
    content_type: str,
) -> None:
    import structlog

    from src.genai.infrastructure.langchain_chain import get_llm_service
    from src.ml_engine.application.use_cases import ProfileUserFromCVUseCase
    from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
    from src.ml_engine.infrastructure.cv_parser import LocalCVParserService
    from src.ml_engine.infrastructure.embeddings import get_embedding_service
    from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository
    from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository
    from src.shared.database import AsyncSessionLocal

    bg_logger = structlog.get_logger("background_tasks")
    bg_logger.info("Starting background CV analysis", user_id=str(user_id), cv_id=str(cv_id))

    try:
        async with AsyncSessionLocal() as session:
            use_case = ProfileUserFromCVUseCase(
                cv_parser=LocalCVParserService(),
                embedding_service=get_embedding_service(),
                cluster_repository=SQLClusterRepository(session),
                profile_repository=SQLUserProfileRepository(session),
                llm_service=get_llm_service(),
                skill_repository=SQLSkillRepository(session),
            )
            await use_case.execute(
                user_id=user_id,
                cv_id=cv_id,
                cv_content=content,
                content_type=content_type,
            )
            # Session context manager commits on success or rolls back on exception
            await session.commit()
            bg_logger.info("Background CV analysis completed successfully", user_id=str(user_id))
    except Exception as exc:
        bg_logger.exception("Background CV analysis failed", user_id=str(user_id), error=str(exc))


@router.get("/me/cvs", response_model=CVListDTO, summary="List uploaded CVs")
async def list_cvs(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> CVListDTO:
    """List all CVs uploaded by the current user."""
    use_case = _get_list_cvs_use_case(session)
    return await use_case.execute(UUID(current_user_id))
