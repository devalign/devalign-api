"""Unit tests for ResetAccountUseCase."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.delivery.application.use_cases import ResetAccountUseCase
from src.delivery.domain.entities import CVDocument


@pytest.mark.asyncio
async def test_reset_account_use_case_success():
    # Arrange
    user_id = uuid4()
    cv_id_1 = uuid4()
    cv_id_2 = uuid4()

    # Mock CVs
    cv_1 = CVDocument(
        id=cv_id_1,
        user_id=user_id,
        storage_path=f"cvs/{user_id}/1_cv.pdf",
        original_filename="cv1.pdf",
        content_type="application/pdf",
        size_bytes=1000,
    )
    cv_2 = CVDocument(
        id=cv_id_2,
        user_id=user_id,
        storage_path=f"cvs/{user_id}/2_cv.docx",
        original_filename="cv2.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        size_bytes=2000,
    )

    # Mock dependencies
    cv_repo = MagicMock()
    cv_repo.get_by_user_id = AsyncMock(return_value=[cv_1, cv_2])
    cv_repo.delete = AsyncMock()

    profile_repo = MagicMock()
    profile_repo.delete_by_user_id = AsyncMock()

    storage_service = MagicMock()
    storage_service.delete_cv = AsyncMock()

    use_case = ResetAccountUseCase(
        cv_repository=cv_repo,
        profile_repository=profile_repo,
        storage_service=storage_service,
    )

    # Act
    await use_case.execute(user_id)

    # Assert
    # Check that CVs are retrieved for the correct user
    cv_repo.get_by_user_id.assert_called_once_with(user_id)

    # Check that both CVs are deleted from storage
    assert storage_service.delete_cv.call_count == 2
    storage_service.delete_cv.assert_any_call(cv_1.storage_path)
    storage_service.delete_cv.assert_any_call(cv_2.storage_path)

    # Check that both CVs are deleted from the repository
    assert cv_repo.delete.call_count == 2
    cv_repo.delete.assert_any_call(cv_1.id)
    cv_repo.delete.assert_any_call(cv_2.id)

    # Check that profile_repository.delete_by_user_id is called
    profile_repo.delete_by_user_id.assert_called_once_with(user_id)
