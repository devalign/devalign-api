"""Unit tests for SQLAlchemyUserRepository and UserModel."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.delivery.domain.entities import User
from src.delivery.infrastructure.models import UserModel
from src.delivery.infrastructure.repository import SQLAlchemyUserRepository


@pytest.mark.asyncio
async def test_user_model_entity_conversion() -> None:
    """Test conversion between User domain entity and UserModel ORM."""
    user_id = uuid4()
    now = datetime.now(UTC)
    entity = User(
        id=user_id,
        email="dev@example.com",
        full_name="Test Dev",
        avatar_url="https://example.com/avatar.png",
        created_at=now,
    )

    model = UserModel.from_entity(entity)
    assert model.user_id == user_id
    assert model.email == "dev@example.com"
    assert model.full_name == "Test Dev"
    assert model.avatar_url == "https://example.com/avatar.png"

    converted = model.to_entity()
    assert converted.id == user_id
    assert converted.email == "dev@example.com"
    assert converted.full_name == "Test Dev"
    assert converted.avatar_url == "https://example.com/avatar.png"


@pytest.mark.asyncio
async def test_repository_get_by_id() -> None:
    """Test repository get_by_id execution."""
    session = AsyncMock()
    user_id = uuid4()
    mock_model = UserModel(
        user_id=user_id,
        email="dev@example.com",
        full_name="Test Dev",
        avatar_url=None,
        created_at=datetime.now(UTC),
    )
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = mock_model
    session.execute.return_value = result_mock

    repo = SQLAlchemyUserRepository(session)
    user = await repo.get_by_id(user_id)

    assert user is not None
    assert user.id == user_id
    assert user.email == "dev@example.com"
    assert user.full_name == "Test Dev"
