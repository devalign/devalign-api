"""PostgreSQL repository implementation for delivery module."""

from uuid import UUID

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.delivery.domain.entities import CVDocument, User
from src.delivery.domain.ports import CVRepository, UserRepository
from src.delivery.infrastructure.models import CVDocumentModel, UserModel

logger = structlog.get_logger(__name__)


class SQLAlchemyUserRepository(UserRepository):
    """Implements UserRepository using async SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, user_id: UUID) -> User | None:
        result = await self._session.execute(select(UserModel).where(UserModel.user_id == user_id))
        model = result.scalar_one_or_none()
        return model.to_entity() if model else None

    async def get_by_email(self, email: str) -> User | None:
        result = await self._session.execute(select(UserModel).where(UserModel.email == email))
        model = result.scalar_one_or_none()
        return model.to_entity() if model else None

    async def upsert(self, user: User) -> User:
        existing = await self.get_by_id(user.id)
        if existing:
            await self._session.execute(
                update(UserModel)
                .where(UserModel.user_id == user.id)
                .values(
                    email=user.email,
                    full_name=user.full_name,
                    avatar_url=user.avatar_url,
                )
            )
            await self._session.flush()
            return user
        else:
            model = UserModel.from_entity(user)
            self._session.add(model)
            await self._session.flush()
            return model.to_entity()


class SQLAlchemyCVRepository(CVRepository):
    """Implements CVRepository using async SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, cv: CVDocument) -> CVDocument:
        model = CVDocumentModel.from_entity(cv)
        self._session.add(model)
        await self._session.flush()
        return model.to_entity()

    async def get_by_user_id(self, user_id: UUID) -> list[CVDocument]:
        result = await self._session.execute(
            select(CVDocumentModel)
            .where(CVDocumentModel.user_id == user_id)
            .order_by(CVDocumentModel.uploaded_at.desc())
        )
        return [row.to_entity() for row in result.scalars().all()]

    async def get_latest_by_user_id(self, user_id: UUID) -> CVDocument | None:
        result = await self._session.execute(
            select(CVDocumentModel)
            .where(CVDocumentModel.user_id == user_id)
            .order_by(CVDocumentModel.uploaded_at.desc())
            .limit(1)
        )
        model = result.scalar_one_or_none()
        return model.to_entity() if model else None

    async def get_by_id(self, cv_id: UUID) -> CVDocument | None:
        result = await self._session.execute(
            select(CVDocumentModel).where(CVDocumentModel.id == cv_id)
        )
        model = result.scalar_one_or_none()
        return model.to_entity() if model else None
