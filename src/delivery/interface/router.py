"""Delivery module API router."""

from uuid import UUID

from fastapi import APIRouter, File, UploadFile

from src.delivery.application.dtos import CVListDTO, CVUploadResultDTO, UserProfileDTO
from src.delivery.application.use_cases import (
    GetCurrentUserUseCase,
    ListUserCVsUseCase,
    UploadCVUseCase,
)
from src.delivery.infrastructure.repository import SQLAlchemyCVRepository, SQLAlchemyUserRepository
from src.delivery.infrastructure.supabase_storage import SupabaseStorageService
from src.dependencies import SessionDep
from src.shared.security import CurrentUserIdDep
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
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> UserProfileDTO:
    """
    Returns the profile of the currently authenticated user.

    Requires a valid Supabase JWT Bearer token.
    """
    use_case = GetCurrentUserUseCase(SQLAlchemyUserRepository(session))
    return await use_case.execute(UUID(current_user_id))


@router.post(
    "/me/cv", response_model=CVUploadResultDTO, status_code=201, summary="Upload CV document"
)
async def upload_cv(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
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
    return await use_case.execute(
        user_id=UUID(current_user_id),
        filename=file.filename or "cv",
        content=content,
        content_type=file.content_type or "application/octet-stream",
    )


@router.get("/me/cvs", response_model=CVListDTO, summary="List uploaded CVs")
async def list_cvs(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> CVListDTO:
    """List all CVs uploaded by the current user."""
    use_case = _get_list_cvs_use_case(session)
    return await use_case.execute(UUID(current_user_id))
