"""Delivery module API router."""

import asyncio
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from src.delivery.application.dtos import CVListDTO, CVStatusDTO, CVUploadResultDTO
from src.delivery.application.use_cases import (
    DeleteCVUseCase,
    GetCurrentUserUseCase,
    ListUserCVsUseCase,
    ResetAccountUseCase,
    UploadCVUseCase,
)
from src.delivery.infrastructure.repository import SQLAlchemyCVRepository, SQLAlchemyUserRepository
from src.delivery.infrastructure.supabase_storage import SupabaseStorageService
from src.dependencies import SessionDep
from src.ml_engine.application.dtos import ProfileUpdateDTO

# ML Engine imports for profile unifications
from src.ml_engine.application.dtos import UserProfileDTO as MLUserProfileDTO
from src.ml_engine.application.use_cases import GetMyProfileUseCase
from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository
from src.shared.security import CurrentUserIdDep, CurrentUserPayloadDep
from src.shared.supabase_client import get_supabase_admin_client

router = APIRouter(prefix="/me", tags=["User Portal — Profile & CV"])


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


@router.get("", response_model=MLUserProfileDTO, summary="Get current user profile")
async def get_me(
    payload: CurrentUserPayloadDep,
    session: SessionDep,
) -> MLUserProfileDTO:
    """
    Returns the computed profile of the currently authenticated developer.
    If no CV is analyzed yet, returns a profile draft with basic details.
    """
    user_id = UUID(str(payload.get("sub")))
    email = str(payload.get("email") or "")

    user_metadata = payload.get("user_metadata")
    if not isinstance(user_metadata, dict):
        user_metadata = {}

    full_name = str(user_metadata.get("full_name") or user_metadata.get("name") or "") or None
    avatar_url = str(user_metadata.get("avatar_url") or user_metadata.get("picture") or "") or None

    # JIT Provisioning (ensure UserModel exists)
    user_repo = SQLAlchemyUserRepository(session)
    user_model = await user_repo.get_by_id(user_id)
    if user_model:
        user_full_name = user_model.full_name
    else:
        user_use_case = GetCurrentUserUseCase(user_repo)
        user_profile_dto = await user_use_case.execute(
            user_id=user_id,
            email=email,
            full_name=full_name,
            avatar_url=avatar_url,
        )
        user_full_name = user_profile_dto.full_name

    # Fetch profile from ML Engine
    repo = SQLUserProfileRepository(session)
    cluster_repo = SQLClusterRepository(session)
    use_case = GetMyProfileUseCase(repo, cluster_repo)
    dto = await use_case.execute(user_id)

    if not dto:
        # Fallback to basic user profile if no CV analyzed yet
        return MLUserProfileDTO(
            user_id=user_id,
            cv_id=None,
            seniority="mid",
            primary_specialty="Software Engineering",
            alignment_score=0.0,
            full_name=user_full_name,
            message="No profile found. Please upload a CV first.",
        )

    return dto


@router.patch(
    "", response_model=MLUserProfileDTO, summary="Update manual fields of developer profile"
)
async def update_my_profile(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
    data: ProfileUpdateDTO,
) -> MLUserProfileDTO:
    """
    Manually update profile details (location, modality, availability, experience lists).
    """
    from dataclasses import replace

    repo = SQLUserProfileRepository(session)
    profile = await repo.get_by_user_id(UUID(current_user_id))
    if not profile:
        raise HTTPException(status_code=404, detail="No profile found. Please upload a CV first.")

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

    cluster_repo = SQLClusterRepository(session)
    dto = await GetMyProfileUseCase(repo, cluster_repo).execute(UUID(current_user_id))
    if not dto:
        raise HTTPException(status_code=404, detail="Profile not found after update")
    return dto


@router.post("/cv", response_model=CVUploadResultDTO, status_code=201, summary="Upload CV document")
async def upload_cv(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CV document (PDF or DOCX, max 5MB)"),
) -> CVUploadResultDTO:
    """
    Upload a CV document for processing.
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

    from src.ml_engine.application.use_cases import ProfileUserFromCVUseCase
    from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
    from src.ml_engine.infrastructure.cv_parser import LocalCVParserService
    from src.ml_engine.infrastructure.llm_client import get_llm_service
    from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository
    from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository
    from src.shared.database import AsyncSessionLocal

    bg_logger = structlog.get_logger("background_tasks")
    bg_logger.info("Starting background CV analysis", user_id=str(user_id), cv_id=str(cv_id))

    max_retries = 3
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            async with AsyncSessionLocal() as session:
                cv_repo = SQLAlchemyCVRepository(session)
                await cv_repo.update_status(cv_id, "processing")
                await session.commit()

                use_case = ProfileUserFromCVUseCase(
                    cv_parser=LocalCVParserService(),
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
                await cv_repo.update_status(cv_id, "completed")
                await session.commit()
                bg_logger.info(
                    "Background CV analysis completed successfully",
                    user_id=str(user_id),
                )
                return
        except Exception as exc:
            last_exception = exc
            bg_logger.warning(
                "Background CV analysis attempt failed",
                attempt=attempt + 1,
                max_retries=max_retries,
                user_id=str(user_id),
                error=str(exc),
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

    bg_logger.exception(
        "Background CV analysis failed after all retries",
        user_id=str(user_id),
        error=str(last_exception),
    )
    try:
        async with AsyncSessionLocal() as fail_session:
            cv_repo = SQLAlchemyCVRepository(fail_session)
            await cv_repo.update_status(cv_id, "failed")
            await fail_session.commit()
    except Exception as db_exc:
        bg_logger.exception(
            "Failed to update CV status to failed", user_id=str(user_id), error=str(db_exc)
        )


@router.get("/cv/status", response_model=CVStatusDTO, summary="Get active CV processing status")
async def get_cv_status(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> CVStatusDTO:
    """Gets the status of the latest CV uploaded by the user."""
    cv_repo = SQLAlchemyCVRepository(session)
    cvs = await cv_repo.get_by_user_id(UUID(current_user_id))
    if not cvs:
        return CVStatusDTO(cv_id=None, status="none")

    cvs.sort(key=lambda x: x.uploaded_at, reverse=True)
    latest_cv = cvs[0]

    return CVStatusDTO(
        cv_id=latest_cv.id, status=latest_cv.status, uploaded_at=latest_cv.uploaded_at
    )


@router.get("/cvs", response_model=CVListDTO, summary="List uploaded CVs")
async def list_cvs(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> CVListDTO:
    """List all CVs uploaded by the current user."""
    use_case = _get_list_cvs_use_case(session)
    return await use_case.execute(UUID(current_user_id))


@router.post(
    "/cvs/{cv_id}/reanalyze",
    response_model=CVUploadResultDTO,
    summary="Re-analyze an existing CV",
)
async def reanalyze_cv(
    cv_id: UUID,
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
    background_tasks: BackgroundTasks,
) -> CVUploadResultDTO:
    """
    Triggers background re-analysis of a CV that was previously uploaded.
    """
    cv_repo = SQLAlchemyCVRepository(session)
    cv = await cv_repo.get_by_id(cv_id)
    if not cv or str(cv.user_id) != current_user_id:
        raise HTTPException(status_code=404, detail="CV not found")

    storage_service = SupabaseStorageService(get_supabase_admin_client())
    content = await storage_service.download_cv(cv.storage_path)

    background_tasks.add_task(
        run_profile_analysis_task,
        user_id=cv.user_id,
        cv_id=cv.id,
        content=content,
        content_type=cv.content_type,
    )

    try:
        url = await storage_service.get_signed_url(cv.storage_path)
    except Exception:
        url = None

    return CVUploadResultDTO(
        cv_id=cv.id,
        user_id=cv.user_id,
        storage_path=cv.storage_path,
        original_filename=cv.original_filename,
        size_bytes=cv.size_bytes,
        download_url=url,
        uploaded_at=cv.uploaded_at,
    )


def _get_delete_cv_use_case(session: SessionDep) -> DeleteCVUseCase:
    return DeleteCVUseCase(
        cv_repository=SQLAlchemyCVRepository(session),
        storage_service=SupabaseStorageService(get_supabase_admin_client()),
    )


@router.delete(
    "/cvs/{cv_id}",
    status_code=204,
    summary="Delete a CV from version history",
)
async def delete_cv(
    cv_id: UUID,
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> None:
    use_case = _get_delete_cv_use_case(session)
    await use_case.execute(
        user_id=UUID(current_user_id),
        cv_id=cv_id,
    )


@router.post(
    "/reset",
    status_code=204,
    summary="Reset user account data",
)
async def reset_account(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> None:
    use_case = ResetAccountUseCase(
        cv_repository=SQLAlchemyCVRepository(session),
        profile_repository=SQLUserProfileRepository(session),
        storage_service=SupabaseStorageService(get_supabase_admin_client()),
    )
    await use_case.execute(UUID(current_user_id))


@router.delete(
    "",
    status_code=204,
    summary="Permanently delete user account",
)
async def delete_account(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> None:
    use_case = ResetAccountUseCase(
        cv_repository=SQLAlchemyCVRepository(session),
        profile_repository=SQLUserProfileRepository(session),
        storage_service=SupabaseStorageService(get_supabase_admin_client()),
    )
    await use_case.execute(UUID(current_user_id))

    admin_client = get_supabase_admin_client()
    admin_client.auth.admin.delete_user(current_user_id)
