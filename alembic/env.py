"""Alembic migration environment."""

import asyncio
import os

# Import settings and Base for autogenerate
import sys
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings

# Import all models so Alembic discovers them for autogenerate
from src.delivery.infrastructure.models import CVDocumentModel, UserModel  # noqa: F401
from src.ml_engine.infrastructure.models import (  # noqa: F401
    ClusterModel,
    ClusterSkillModel,
    DiagnosticModel,
    DiagnosticSkillModel,
    ProfileModel,
    SkillModel,
)
from src.scraper.infrastructure.models import JobOfferModel, OfferSkillModel  # noqa: F401
from src.shared.database import Base

config = context.config

# Override sqlalchemy.url with our settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in offline mode (no DB connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode (required for asyncpg)."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
