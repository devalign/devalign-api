"""Delivery module use cases."""

from datetime import datetime
from uuid import UUID, uuid4

import structlog

from src.delivery.application.dtos import CVListDTO, CVUploadResultDTO, UserProfileDTO
from src.delivery.domain.entities import CVDocument, User
from src.delivery.domain.ports import CVRepository, StorageService, UserRepository
from src.shared.exceptions import FileTooLargeError, UnsupportedFileTypeError

logger = structlog.get_logger(__name__)

# Constants
MAX_CV_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_CONTENT_TYPES = frozenset(
    [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
)


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
        # Validate file type
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise UnsupportedFileTypeError(
                f"File type '{content_type}' not supported. Use PDF or DOCX."
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
        download_url = await self._storage.get_signed_url(storage_path)

        logger.info("CV uploaded successfully", cv_id=str(saved_cv.id))

        return CVUploadResultDTO(
            cv_id=saved_cv.id,
            user_id=saved_cv.user_id,
            storage_path=saved_cv.storage_path,
            original_filename=saved_cv.original_filename,
            size_bytes=saved_cv.size_bytes,
            download_url=download_url,
        )


class ListUserCVsUseCase:
    """List all CVs uploaded by a user."""

    def __init__(self, cv_repository: CVRepository, storage_service: StorageService) -> None:
        self._cvs = cv_repository
        self._storage = storage_service

    async def execute(self, user_id: UUID) -> CVListDTO:
        cv_docs = await self._cvs.get_by_user_id(user_id)

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
                    download_url=url,
                )
            )

        return CVListDTO(user_id=user_id, cvs=cv_results, total=len(cv_results))
