import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def main():
    engine = create_async_engine(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/devalign_dev"
    )
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT name, market_insights FROM clusters"))
        for row in result:
            print(f"Cluster: {row[0]} | Insights: {row[1]}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
