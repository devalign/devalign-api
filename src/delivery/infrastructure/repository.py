"""PostgreSQL repository implementation for delivery module."""

from typing import Any
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
        existing = await self.get_by_id(cv.id)
        if existing:
            stmt = (
                update(CVDocumentModel)
                .where(CVDocumentModel.id == cv.id)
                .values(
                    storage_path=cv.storage_path,
                    original_filename=cv.original_filename,
                    content_type=cv.content_type,
                    size_bytes=cv.size_bytes,
                    status=cv.status,
                    error_message=cv.error_message,
                    extracted_data=cv.extracted_data,
                )
            )
            await self._session.execute(stmt)
            await self._session.flush()
            return cv
        else:
            model = CVDocumentModel.from_entity(cv)
            self._session.add(model)
            await self._session.flush()
            return model.to_entity()

    async def get_by_user_id(self, user_id: UUID, limit: int | None = None) -> list[CVDocument]:
        stmt = (
            select(CVDocumentModel)
            .where(CVDocumentModel.user_id == user_id)
            .order_by(CVDocumentModel.uploaded_at.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
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

    async def delete(self, cv_id: UUID) -> None:
        # Clear any active CV references in user profiles
        from src.ml_engine.infrastructure.models import ProfileModel

        await self._session.execute(
            update(ProfileModel).where(ProfileModel.cv_id == cv_id).values(cv_id=None)
        )

        result = await self._session.execute(
            select(CVDocumentModel).where(CVDocumentModel.id == cv_id)
        )
        model = result.scalar_one_or_none()
        if model:
            await self._session.delete(model)
            await self._session.flush()

    async def update_extracted_data(self, cv_id: UUID, extracted_data: dict[str, Any]) -> int:
        result = await self._session.execute(
            update(CVDocumentModel)
            .where(CVDocumentModel.id == cv_id)
            .values(extracted_data=extracted_data)
        )
        await self._session.flush()
        return int(result.rowcount)  # type: ignore

    async def update_status(self, cv_id: UUID, status: str, error_message: str | None = None) -> int:
        values: dict[str, object] = {"status": status}
        if error_message is not None:
            values["error_message"] = error_message
        result = await self._session.execute(
            update(CVDocumentModel).where(CVDocumentModel.id == cv_id).values(values)
        )
        await self._session.flush()
        return int(result.rowcount)  # type: ignore
