"""Add professional_summary column to profiles table."""

import asyncio
import sys
sys.path.insert(0, "src")
from config import settings


async def main():
    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    import asyncpg
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            "ALTER TABLE profiles ADD COLUMN IF NOT EXISTS professional_summary TEXT;"
        )
        print("Column professional_summary added successfully.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
