"""SQLAlchemy ORM models for the delivery module."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.delivery.domain.entities import CVDocument, User
from src.shared.database import Base


class UserModel(Base):
    """ORM model for the users table.

    This is the *public* user table, distinct from auth.users managed by Supabase Auth.
    user_id references auth.users(id) — Supabase Auth handles password hashing,
    OAuth providers, and session management internally. This table only stores
    application-level metadata.
    """

    __tablename__ = "users"

    # References auth.users(id) from Supabase Auth — NOT a self-managed auth column.
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    # full_name is denormalized here for quick access without joining profiles.
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Relationships
    cvs: Mapped[list["CVDocumentModel"]] = relationship(
        "CVDocumentModel", back_populates="user", lazy="select"
    )

    def to_entity(self) -> User:
        return User(
            id=self.user_id,
            email=self.email,
            full_name=self.full_name,
            avatar_url=self.avatar_url,
            created_at=self.created_at,
        )

    @classmethod
    def from_entity(cls, user: User) -> "UserModel":
        return cls(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            avatar_url=user.avatar_url,
            created_at=user.created_at,
        )


class CVDocumentModel(Base):
    """ORM model for the cv_documents table."""

    __tablename__ = "cv_documents"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    user: Mapped["UserModel"] = relationship("UserModel", back_populates="cvs")

    def to_entity(self) -> CVDocument:
        return CVDocument(
            id=self.id,
            user_id=self.user_id,
            storage_path=self.storage_path,
            original_filename=self.original_filename,
            content_type=self.content_type,
            size_bytes=self.size_bytes,
            uploaded_at=self.uploaded_at,
        )

    @classmethod
    def from_entity(cls, cv: CVDocument) -> "CVDocumentModel":
        return cls(
            id=cv.id,
            user_id=cv.user_id,
            storage_path=cv.storage_path,
            original_filename=cv.original_filename,
            content_type=cv.content_type,
            size_bytes=cv.size_bytes,
            uploaded_at=cv.uploaded_at,
        )
