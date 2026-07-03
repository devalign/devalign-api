"""Add error_message column to cv_documents table if it doesn't exist."""
import sys
sys.path.insert(0, "src")
from config import settings
import asyncpg
import asyncio


async def main():
    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='cv_documents' AND column_name='error_message'"
        )
        if not exists:
            await conn.execute(
                "ALTER TABLE cv_documents ADD COLUMN error_message TEXT"
            )
            print("Column error_message added successfully")
        else:
            print("Column error_message already exists")
    finally:
        await conn.close()


asyncio.run(main())
