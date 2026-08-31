import asyncio, asyncpg, sys
sys.path.insert(0, "src")
from config import settings

async def main():
    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    db_cols = await conn.fetch(
        "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public'"
    )
    db_table_cols = {}
    for r in db_cols:
        db_table_cols.setdefault(r['table_name'], set()).add(r['column_name'])
    
    db_cols = await conn.fetch(
        "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public'"
    )
    db_table_cols = {}
    for r in db_cols:
        db_table_cols.setdefault(r['table_name'], set()).add(r['column_name'])

    from src.shared.database import Base
    import src.delivery.infrastructure.models  # noqa: F401
    import src.ml_engine.infrastructure.models  # noqa: F401
    import src.scraper.infrastructure.models  # noqa: F401

    all_matched = True
    for tbl_name, table in Base.metadata.tables.items():
        if tbl_name not in db_table_cols:
            print(f"Table {tbl_name} NOT in DB!")
            all_matched = False
            continue
        model_cols = {c.name for c in table.columns}
        missing_in_db = model_cols - db_table_cols[tbl_name]
        extra_in_db = db_table_cols[tbl_name] - model_cols
        print(f"=== {tbl_name} ===")
        if missing_in_db:
            print(f"  MISSING IN DB: {missing_in_db}")
            all_matched = False
        if extra_in_db:
            print(f"  EXTRA IN DB: {extra_in_db}")
            all_matched = False
        if not missing_in_db and not extra_in_db:
            print("  PERFECT MATCH")

    print("\nOVERALL SCHEMA STATUS:", "ALL PERFECT MATCH" if all_matched else "MISMATCHES FOUND")
    await conn.close()

asyncio.run(main())
