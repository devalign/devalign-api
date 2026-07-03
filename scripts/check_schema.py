import asyncio, asyncpg, sys
sys.path.insert(0, "src")
from config import settings

async def main():
    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    rows = await conn.fetch(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'cv_documents' ORDER BY ordinal_position"
    )
    for r in rows:
        print(f"{r['column_name']:30} {r['data_type']}")
    await conn.close()

asyncio.run(main())
