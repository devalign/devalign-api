"""Unit tests for DeleteCVUseCase."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.delivery.application.use_cases import DeleteCVUseCase
from src.delivery.domain.entities import CVDocument
from src.shared.exceptions import AuthorizationError, NotFoundError


@pytest.mark.asyncio
async def test_delete_cv_use_case_success():
    # Arrange
    user_id = uuid4()
    cv_id = uuid4()

    cv = CVDocument(
        id=cv_id,
        user_id=user_id,
        storage_path=f"cvs/{user_id}/1_cv.pdf",
        original_filename="cv1.pdf",
        content_type="application/pdf",
        size_bytes=1000,
    )

    cv_repo = MagicMock()
    cv_repo.get_by_id = AsyncMock(return_value=cv)
    cv_repo.delete = AsyncMock()

    storage_service = MagicMock()
    storage_service.delete_cv = AsyncMock()

    use_case = DeleteCVUseCase(
        cv_repository=cv_repo,
        storage_service=storage_service,
    )

    # Act
    await use_case.execute(user_id=user_id, cv_id=cv_id)

    # Assert
    cv_repo.get_by_id.assert_called_once_with(cv_id)
    storage_service.delete_cv.assert_called_once_with(cv.storage_path)
    cv_repo.delete.assert_called_once_with(cv.id)


@pytest.mark.asyncio
async def test_delete_cv_use_case_not_found():
    # Arrange
    user_id = uuid4()
    cv_id = uuid4()

    cv_repo = MagicMock()
    cv_repo.get_by_id = AsyncMock(return_value=None)
    storage_service = MagicMock()

    use_case = DeleteCVUseCase(
        cv_repository=cv_repo,
        storage_service=storage_service,
    )

    # Act & Assert
    with pytest.raises(NotFoundError):
        await use_case.execute(user_id=user_id, cv_id=cv_id)


@pytest.mark.asyncio
async def test_delete_cv_use_case_unauthorized():
    # Arrange
    owner_id = uuid4()
    other_user_id = uuid4()
    cv_id = uuid4()

    cv = CVDocument(
        id=cv_id,
        user_id=owner_id,
        storage_path=f"cvs/{owner_id}/1_cv.pdf",
        original_filename="cv1.pdf",
        content_type="application/pdf",
        size_bytes=1000,
    )

    cv_repo = MagicMock()
    cv_repo.get_by_id = AsyncMock(return_value=cv)
    storage_service = MagicMock()

    use_case = DeleteCVUseCase(
        cv_repository=cv_repo,
        storage_service=storage_service,
    )

    # Act & Assert
    with pytest.raises(AuthorizationError):
        await use_case.execute(user_id=other_user_id, cv_id=cv_id)


@pytest.mark.asyncio
async def test_delete_cv_use_case_storage_failure_still_deletes_db_record():
    # Arrange
    user_id = uuid4()
    cv_id = uuid4()

    cv = CVDocument(
        id=cv_id,
        user_id=user_id,
        storage_path=f"cvs/{user_id}/1_cv.pdf",
        original_filename="cv1.pdf",
        content_type="application/pdf",
        size_bytes=1000,
    )

    cv_repo = MagicMock()
    cv_repo.get_by_id = AsyncMock(return_value=cv)
    cv_repo.delete = AsyncMock()

    storage_service = MagicMock()
    storage_service.delete_cv = AsyncMock(side_effect=RuntimeError("Storage unreachable"))

    use_case = DeleteCVUseCase(
        cv_repository=cv_repo,
        storage_service=storage_service,
    )

    # Act - Should not raise and must still delete the DB record
    await use_case.execute(user_id=user_id, cv_id=cv_id)

    # Assert
    storage_service.delete_cv.assert_called_once_with(cv.storage_path)
    cv_repo.delete.assert_called_once_with(cv.id)
