"""Delivery module use cases."""

from datetime import datetime
from uuid import UUID, uuid4

import structlog

from src.delivery.application.dtos import CVListDTO, CVUploadResultDTO, UserProfileDTO
from src.delivery.domain.entities import CVDocument, User
from src.delivery.domain.ports import CVRepository, StorageService, UserRepository
from src.ml_engine.domain.ports import UserProfileRepository
from src.shared.exceptions import (
    AuthorizationError,
    FileTooLargeError,
    NotFoundError,
    UnsupportedFileTypeError,
)

logger = structlog.get_logger(__name__)

# Constants
MAX_CV_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_CONTENT_TYPES = frozenset(
    [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
)
VALID_EXTENSIONS = frozenset(["pdf", "docx"])
PDF_MAGIC = b"%PDF"
DOCX_MAGIC = b"PK\x03\x04"


class GetCurrentUserUseCase:
    """Retrieve profile of the currently authenticated user.

    If the user does not exist in the local database (e.g., they just registered via Supabase),
    we implicitly provision them (Just-In-Time) using the provided token identity metadata.
    """

    def __init__(self, user_repository: UserRepository) -> None:
        self._users = user_repository

    async def execute(
        self,
        user_id: UUID,
        email: str,
        full_name: str | None = None,
        avatar_url: str | None = None,
    ) -> UserProfileDTO:
        user = await self._users.get_by_id(user_id)
        if user is None:
            logger.info("JIT Provisioning new user", user_id=str(user_id), email=email)
            user = User(
                id=user_id,
                email=email,
                full_name=full_name,
                avatar_url=avatar_url,
                created_at=datetime.utcnow(),
            )
            user = await self._users.upsert(user)

        return UserProfileDTO(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            avatar_url=user.avatar_url,
        )


class UploadCVUseCase:
    """Handle CV file upload to Supabase Storage and persist metadata."""

    def __init__(
        self,
        cv_repository: CVRepository,
        storage_service: StorageService,
    ) -> None:
        self._cvs = cv_repository
        self._storage = storage_service

    async def execute(
        self,
        user_id: UUID,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> CVUploadResultDTO:
        # Validate file type (content-type header)
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise UnsupportedFileTypeError(
                f"File type '{content_type}' not supported. Use PDF or DOCX."
            )

        # Validate file extension
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in VALID_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"File extension '.{ext}' not supported. Use .pdf or .docx."
            )

        # Validate file signature (magic bytes)
        if content[:4] == PDF_MAGIC:
            pass  # Valid PDF signature
        elif content[:4] == DOCX_MAGIC:
            pass  # Valid DOCX (ZIP) signature
        else:
            raise UnsupportedFileTypeError(
                "File content signature does not match PDF or DOCX format."
            )

        # Validate file size
        if len(content) > MAX_CV_SIZE_BYTES:
            raise FileTooLargeError(f"File size {len(content)} bytes exceeds the 5MB limit")

        logger.info("Uploading CV", user_id=str(user_id), filename=filename)

        # Upload to storage
        storage_path = await self._storage.upload_cv(
            user_id=user_id,
            filename=filename,
            content=content,
            content_type=content_type,
        )

        # Enforce history limit (max 5 CVs)
        existing_cvs = await self._cvs.get_by_user_id(user_id)
        if len(existing_cvs) >= 5:
            # Delete the oldest CVs so we make room for the new one (keep at most 4 existing CVs)
            cvs_to_delete = existing_cvs[4:]
            for old_cv in cvs_to_delete:
                try:
                    await self._storage.delete_cv(old_cv.storage_path)
                except Exception as exc:
                    logger.error(
                        "Failed to delete old CV from storage during upload pruning",
                        storage_path=old_cv.storage_path,
                        error=str(exc),
                    )
                await self._cvs.delete(old_cv.id)

        # Persist metadata
        cv = CVDocument(
            id=uuid4(),
            user_id=user_id,
            storage_path=storage_path,
            original_filename=filename,
            content_type=content_type,
            size_bytes=len(content),
        )
        saved_cv = await self._cvs.save(cv)

        # Generate download URL
        try:
            download_url = await self._storage.get_signed_url(storage_path)
        except Exception as exc:
            logger.error(
                "Failed to generate signed URL for uploaded CV",
                cv_id=str(saved_cv.id),
                error=str(exc),
            )
            download_url = None

        logger.info("CV uploaded successfully", cv_id=str(saved_cv.id))

        return CVUploadResultDTO(
            cv_id=saved_cv.id,
            user_id=saved_cv.user_id,
            storage_path=saved_cv.storage_path,
            original_filename=saved_cv.original_filename,
            size_bytes=saved_cv.size_bytes,
            status=saved_cv.status,
            download_url=download_url,
            uploaded_at=saved_cv.uploaded_at,
        )


class ListUserCVsUseCase:
    """List all CVs uploaded by a user."""

    def __init__(self, cv_repository: CVRepository, storage_service: StorageService) -> None:
        self._cvs = cv_repository
        self._storage = storage_service

    async def execute(self, user_id: UUID) -> CVListDTO:
        cv_docs = await self._cvs.get_by_user_id(user_id, limit=5)

        cv_results = []
        for cv in cv_docs:
            try:
                url = await self._storage.get_signed_url(cv.storage_path)
            except Exception:
                url = None

            cv_results.append(
                CVUploadResultDTO(
                    cv_id=cv.id,
                    user_id=cv.user_id,
                    storage_path=cv.storage_path,
                    original_filename=cv.original_filename,
                    size_bytes=cv.size_bytes,
                    status=cv.status,
                    download_url=url,
                    uploaded_at=cv.uploaded_at,
                )
            )

        return CVListDTO(user_id=user_id, cvs=cv_results, total=len(cv_results))


class DeleteCVUseCase:
    """Delete a CV document from storage and database."""

    def __init__(self, cv_repository: CVRepository, storage_service: StorageService) -> None:
        self._cvs = cv_repository
        self._storage = storage_service

    async def execute(self, user_id: UUID, cv_id: UUID) -> None:
        cv = await self._cvs.get_by_id(cv_id)
        if not cv:
            raise NotFoundError("CV not found")

        if cv.user_id != user_id:
            raise AuthorizationError("You do not have permission to delete this CV")

        # Delete from storage first
        await self._storage.delete_cv(cv.storage_path)

        # Delete from database (also clears profile references)
        await self._cvs.delete(cv.id)


class ResetAccountUseCase:
    """Reset user account: deletes all CV documents (storage + DB) and profile/diagnostics."""

    def __init__(
        self,
        cv_repository: CVRepository,
        profile_repository: UserProfileRepository,
        storage_service: StorageService,
    ) -> None:
        self._cvs = cv_repository
        self._profiles = profile_repository
        self._storage = storage_service

    async def execute(self, user_id: UUID) -> None:
        import asyncio

        logger.info("Resetting user account data", user_id=str(user_id))

        # 1. Fetch all CVs for the user
        cv_docs = await self._cvs.get_by_user_id(user_id)

        # 2. Delete CV files from storage in parallel
        async def _safe_delete_storage(storage_path: str) -> None:
            try:
                await self._storage.delete_cv(storage_path)
            except Exception as exc:
                logger.error(
                    "Failed to delete CV from storage during account reset",
                    storage_path=storage_path,
                    error=str(exc),
                )

        if cv_docs:
            await asyncio.gather(*[_safe_delete_storage(cv.storage_path) for cv in cv_docs])

        # 3. Delete CV records from database
        for cv in cv_docs:
            await self._cvs.delete(cv.id)

        # 4. Delete user profile (cascades to diagnostics and skills)
        await self._profiles.delete_by_user_id(user_id)
