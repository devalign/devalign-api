"""Delivery module API router."""

import asyncio
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from src.delivery.application.dtos import (
    CVListDTO,
    CVStatusDTO,
    CVUploadResultDTO,
    FinalizeRequestDTO,
    FinalizeResponseDTO,
)
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
from src.ml_engine.application.dtos import ProfileUpdateDTO, SkillDTO
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
        "professional_summary",
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

    await session.commit()

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
    """Run the CV analysis pipeline as a FastAPI background task.

    Extracts CV text, runs combined LLM extraction, stores extracted data
    on the CVDocument (not the profile), and marks the CV as
    ``"skills_detected"`` so the frontend can display skills for user
    validation before persisting the diagnosis.
    """
    import structlog

    from src.ml_engine.application.use_cases import ProfileUserFromCVUseCase
    from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
    from src.ml_engine.infrastructure.cv_parser import LocalCVParserService
    from src.ml_engine.infrastructure.llm_client import get_llm_service
    from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository
    from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository
    from src.shared.database import AsyncSessionLocal
    from src.shared.exceptions import RateLimitError

    bg_logger = structlog.get_logger("background_tasks")
    bg_logger.info(
        "Starting CV analysis",
        user_id=str(user_id),
        cv_id=str(cv_id),
    )

    try:
        async with AsyncSessionLocal() as session:
            cv_repo = SQLAlchemyCVRepository(session)

            # Mark as processing — retry if row not yet visible (race condition)
            bg_logger.info("Setting status to processing", cv_id=str(cv_id))
            rows = await cv_repo.update_status(cv_id, "processing")
            for attempt in range(3):
                if rows > 0:
                    break
                bg_logger.info(
                    "CV not found yet, retrying...",
                    cv_id=str(cv_id),
                    attempt=attempt + 1,
                )
                await asyncio.sleep(1)
                await session.commit()
                rows = await cv_repo.update_status(cv_id, "processing")
            if rows == 0:
                raise RuntimeError(
                    f"CV {cv_id} not found in database after 3 retries"
                )
            await session.commit()

            use_case = ProfileUserFromCVUseCase(
                cv_parser=LocalCVParserService(),
                cluster_repository=SQLClusterRepository(session),
                profile_repository=SQLUserProfileRepository(session),
                llm_service=get_llm_service(),
                skill_repository=SQLSkillRepository(session),
            )

            bg_logger.info("Starting LLM extraction", cv_id=str(cv_id))
            result = await use_case.extract_skills(
                user_id=user_id,
                cv_id=cv_id,
                cv_content=content,
                content_type=content_type,
            )
            bg_logger.info("LLM extraction completed", cv_id=str(cv_id))

            # Store extracted_data directly (without touching status)
            bg_logger.info("Storing extracted data", cv_id=str(cv_id))
            rows = await cv_repo.update_extracted_data(
                cv_id, result["extracted_data"]
            )
            if rows == 0:
                raise RuntimeError(
                    f"Failed to update extracted_data for CV {cv_id}"
                )

            # Mark as skills_detected so the frontend polling picks it up
            bg_logger.info(
                "Setting status to skills_detected", cv_id=str(cv_id)
            )
            rows = await cv_repo.update_status(cv_id, "skills_detected")
            if rows == 0:
                raise RuntimeError(
                    f"Failed to update status to skills_detected for CV {cv_id}"
                )
            await session.commit()

            bg_logger.info(
                "Combined extraction complete — waiting for user validation",
                user_id=str(user_id),
                cv_id=str(cv_id),
            )
    except Exception as exc:
        is_rate_limit = isinstance(exc, RateLimitError)
        error_msg = str(exc) if is_rate_limit else None
        bg_logger.exception(
            "CV analysis background task failed",
            user_id=str(user_id),
            cv_id=str(cv_id),
            error=str(exc),
            is_rate_limit=is_rate_limit,
        )

        # Retry setting status to "failed" up to 3 times
        last_db_exc: Exception | None = None
        for attempt in range(3):
            try:
                async with AsyncSessionLocal() as fail_session:
                    cv_repo = SQLAlchemyCVRepository(fail_session)
                    rows = await cv_repo.update_status(
                        cv_id, "failed", error_message=error_msg
                    )
                    if rows > 0:
                        await fail_session.commit()
                        bg_logger.info(
                            "CV status set to failed",
                            cv_id=str(cv_id),
                            attempt=attempt + 1,
                        )
                        last_db_exc = None
                        break
                    bg_logger.warning(
                        "CV not found when setting failed status",
                        cv_id=str(cv_id),
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(1)
            except Exception as db_exc:
                last_db_exc = db_exc
                bg_logger.exception(
                    "Retry setting failed status errored",
                    cv_id=str(cv_id),
                    attempt=attempt + 1,
                )
                await asyncio.sleep(1)

        if last_db_exc is not None:
            bg_logger.exception(
                "All retries exhausted — could not set CV status to failed",
                user_id=str(user_id),
                cv_id=str(cv_id),
                error=str(last_db_exc),
            )
            if attempt < 3 - 1:
                await asyncio.sleep(2**attempt)

    bg_logger.exception(
        "Background CV analysis failed after all retries",
        user_id=str(user_id),
        error=str(last_db_exc),
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

    error_message = getattr(latest_cv, "error_message", None) or None
    return CVStatusDTO(
        cv_id=latest_cv.id,
        status=latest_cv.status,
        uploaded_at=latest_cv.uploaded_at,
        error_message=error_message,
    )


@router.get("/cvs", response_model=CVListDTO, summary="List uploaded CVs")
async def list_cvs(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> CVListDTO:
    """List all CVs uploaded by the current user."""
    use_case = _get_list_cvs_use_case(session)
    return await use_case.execute(UUID(current_user_id))


def _compute_ict(
    self_taught: bool,
    personal_projects: bool,
    years_of_experience: int,
    has_certification: bool,
) -> float:
    exp_points = 3 * years_of_experience
    cert_points = 4 if has_certification else 0
    projects_points = 2 if personal_projects else 0
    self_taught_points = 1 if self_taught else 0
    return float(min(10.0, self_taught_points + projects_points + exp_points + cert_points))


@router.get(
    "/cvs/{cv_id}/status",
    response_model=CVStatusDTO,
    summary="Get a specific CV's processing status",
)
async def get_cv_status_by_id(
    cv_id: UUID,
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> CVStatusDTO:
    """Gets the processing status for a specific CV."""
    cv_repo = SQLAlchemyCVRepository(session)
    cv = await cv_repo.get_by_id(cv_id)
    if not cv or str(cv.user_id) != current_user_id:
        # CV might still be in-flight from a previous request's transaction.
        # Return a non-error response so the frontend can keep polling.
        return CVStatusDTO(cv_id=None, status="not_found")

    error_message = getattr(cv, "error_message", None) or None
    extracted_skills = None
    if cv.status == "skills_detected" and cv.extracted_data is not None:
        raw_skills = cv.extracted_data.get("skills", None)
        if raw_skills is None:
            raw_skills = cv.extracted_data.get("technical_skills", None)
        if raw_skills is None:
            raw_skills = cv.extracted_data.get("soft_skills", None)
        if isinstance(raw_skills, list):
            extracted_skills = []
            for s in raw_skills:
                if not isinstance(s, dict) or "name" not in s:
                    continue
                years_of_experience = s.get("years_of_experience", 0) or 0
                self_taught = bool(s.get("self_taught", False))
                personal_projects = bool(s.get("personal_projects", False))
                has_certification = bool(s.get("has_certification", False))
                extracted_skills.append(
                    SkillDTO(
                        name=s["name"],
                        skill_type=s.get("category", "technical"),
                        years_of_experience=years_of_experience,
                        self_taught=self_taught,
                        personal_projects=personal_projects,
                        has_certification=has_certification,
                        ict_score=_compute_ict(
                            self_taught=self_taught,
                            personal_projects=personal_projects,
                            years_of_experience=years_of_experience,
                            has_certification=has_certification,
                        ),
                    )
                )

    return CVStatusDTO(
        cv_id=cv.id,
        status=cv.status,
        uploaded_at=cv.uploaded_at,
        error_message=error_message,
        extracted_skills=extracted_skills,
    )


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
    Uses the new extract-only flow — the user validates skills before diagnosis.
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


@router.post(
    "/cv/{cv_id}/finalize",
    response_model=FinalizeResponseDTO,
    status_code=202,
    summary="Finalize CV analysis with validated skills",
)
async def finalize_cv_analysis(
    cv_id: UUID,
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
    background_tasks: BackgroundTasks,
    data: FinalizeRequestDTO | None = None,
) -> FinalizeResponseDTO:
    """
    Finalize the CV analysis pipeline (Phase 2).

    Called after the user has reviewed and validated their detected skills.
    Runs the full diagnosis (catalog normalization, affinity computation,
    gap detection) and persists the complete profile with ``is_diagnosed=True``.
    """
    from src.ml_engine.application.use_cases import ProfileUserFromCVUseCase
    from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
    from src.ml_engine.infrastructure.cv_parser import LocalCVParserService
    from src.ml_engine.infrastructure.llm_client import get_llm_service
    from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository
    from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository

    # Verify the CV exists and is in the right state
    cv_repo = SQLAlchemyCVRepository(session)
    cv = await cv_repo.get_by_id(cv_id)
    if not cv or str(cv.user_id) != current_user_id:
        raise HTTPException(status_code=404, detail="CV not found")

    if cv.status not in ("skills_detected", "processing", "completed"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot finalize CV in status '{cv.status}'. Expected 'skills_detected'.",
        )

    validated_skills = data.skills if data else None

    async def run_finalize_in_background(
        uid: UUID,
        cvid: UUID,
        skills: list[Any] | None,
    ) -> None:
        """Run Phase 2 diagnosis in a background task.

        Reads extracted_data from CVDocument, builds the UserProfile,
        saves it, then runs full diagnosis (normalize, affinity, gaps).
        """
        from uuid import uuid4

        import structlog

        from src.ml_engine.application.use_cases import _nature_from_category
        from src.ml_engine.domain.entities import (
            ClusterAffinity,
            SeniorityLevel,
            Skill,
            UserProfile,
        )
        from src.shared.database import AsyncSessionLocal

        bg_logger = structlog.get_logger("background_tasks")

        async with AsyncSessionLocal() as bg_session:
            cv_repo_bg = SQLAlchemyCVRepository(bg_session)
            cv = await cv_repo_bg.get_by_id(cvid)

            if not cv or not cv.extracted_data:
                bg_logger.error("No extracted data found for CV", cv_id=str(cvid))
                await cv_repo_bg.update_status(cvid, "failed", error_message="No extracted data found")
                await bg_session.commit()
                return

            extracted_data = cv.extracted_data
            years_exp = extracted_data.get("years_experience")

            if isinstance(years_exp, (int, float)):
                if years_exp >= 6:
                    seniority = SeniorityLevel.SENIOR
                elif years_exp >= 3:
                    seniority = SeniorityLevel.MID
                else:
                    seniority = SeniorityLevel.JUNIOR
            else:
                seniority = SeniorityLevel.JUNIOR

            # Use validated skills from frontend, or fall back to extracted ones
            raw_skills = skills if skills is not None else extracted_data.get("skills", [])
            raw_skills_list = raw_skills if isinstance(raw_skills, list) else []

            skill_objects = []
            for item in raw_skills_list:
                if isinstance(item, dict) and "name" in item:
                    skill_objects.append(
                        Skill(
                            name=item["name"],
                            nature=_nature_from_category(item.get("category", "technical")),
                            normalized_name=item["name"].lower().replace(" ", "").replace(".", ""),
                            self_taught=bool(item.get("self_taught", False)),
                            personal_projects=bool(item.get("personal_projects", False)),
                            years_of_experience=int(item.get("years_of_experience", 0) or 0),
                            has_certification=bool(item.get("has_certification", False)),
                        )
                    )
                elif isinstance(item, dict) and "skill_type" in item:
                    # Handle SkillDTO format from frontend
                    skill_objects.append(
                        Skill(
                            name=item["name"],
                            nature=_nature_from_category(item.get("skill_type", "technical")),
                            normalized_name=item["name"].lower().replace(" ", "").replace(".", ""),
                            self_taught=bool(item.get("self_taught", False)),
                            personal_projects=bool(item.get("personal_projects", False)),
                            years_of_experience=int(item.get("years_of_experience", 0) or 0),
                            has_certification=bool(item.get("has_certification", False)),
                        )
                    )

            profile = UserProfile(
                user_id=uid,
                cv_id=cvid,
                embedding=[],
                detected_skills=skill_objects,
                seniority=seniority,
                primary_affinity=ClusterAffinity(
                    cluster_id=uuid4(),
                    cluster_name="Sin Diagnóstico",
                    affinity_score=0.0,
                    is_primary=True,
                ),
                secondary_affinities=[],
                skill_gaps=[],
                current_job_role=extracted_data.get("current_job_role") or None,
                professional_summary=extracted_data.get("professional_summary") or None,
                years_experience=int(years_exp) if isinstance(years_exp, (int, float)) else None,
                work_experience=extracted_data.get("work_experience") or [],
                education=extracted_data.get("education") or [],
                certifications=extracted_data.get("certifications") or [],
                cv_raw_text=extracted_data.get("cv_text", ""),
                is_diagnosed=False,
            )

            profile_repo = SQLUserProfileRepository(bg_session)
            await profile_repo.save(profile)

            use_case = ProfileUserFromCVUseCase(
                cv_parser=LocalCVParserService(),
                cluster_repository=SQLClusterRepository(bg_session),
                profile_repository=profile_repo,
                llm_service=get_llm_service(),
                skill_repository=SQLSkillRepository(bg_session),
            )
            try:
                await use_case.finalize_diagnosis(
                    user_id=uid,
                    cv_id=cvid,
                    validated_skills=skills,
                )
                await cv_repo_bg.update_status(cvid, "completed")
                await bg_session.commit()
                bg_logger.info(
                    "Phase 2 finalize complete",
                    user_id=str(uid),
                    cv_id=str(cvid),
                )
            except Exception as exc:
                bg_logger.exception(
                    "Phase 2 finalize failed",
                    user_id=str(uid),
                    cv_id=str(cvid),
                    error=str(exc),
                )
                await cv_repo_bg.update_status(cvid, "failed", error_message=str(exc))
                await bg_session.commit()

    background_tasks.add_task(
        run_finalize_in_background,
        uid=UUID(current_user_id),
        cvid=cv_id,
        skills=validated_skills,
    )

    return FinalizeResponseDTO(
        cv_id=cv_id,
        status="processing",
        message="Diagnóstico en proceso...",
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
