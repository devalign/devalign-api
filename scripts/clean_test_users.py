import asyncio, asyncpg, sys
sys.path.insert(0, "src")
from config import settings

async def main():
    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    result = await conn.execute("DELETE FROM users WHERE email LIKE 'test@%'")
    print(f"Deleted: {result}")
    await conn.close()

asyncio.run(main())
