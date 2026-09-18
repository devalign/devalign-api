"""Sync script: Update skill_standards with Lightcast Category, Subcategory, and Standard Type.

1. Ensures migration columns (standard_type, category_name, subcategory_name) exist in skill_standards.
2. Loads local lightcast_raw.json (for standard_type: Specialized Skill, Common Skill, Certification).
3. Fetches live skill-categories from https://lightcast.io/api/skills/skill-categories (for Category and Subcategory).
4. Updates all skill_standards rows matching standard_name = 'Lightcast'.
"""

import asyncio
import json
from pathlib import Path
import sys
import httpx
from sqlalchemy import text
import structlog

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.database import AsyncSessionLocal

logger = structlog.get_logger(__name__)

RAW_FILE = Path(__file__).resolve().parent / "data" / "lightcast_raw.json"
CATEGORIES_URL = "https://lightcast.io/api/skills/skill-categories"


async def ensure_schema_columns(session) -> None:
    """Add taxonomy columns and indexes to skill_standards if they do not exist."""
    print("Step 1: Ensuring columns exist in skill_standards...")
    ddl_statements = [
        "ALTER TABLE skill_standards ADD COLUMN IF NOT EXISTS standard_type VARCHAR(100);",
        "ALTER TABLE skill_standards ADD COLUMN IF NOT EXISTS category_name VARCHAR(150);",
        "ALTER TABLE skill_standards ADD COLUMN IF NOT EXISTS subcategory_name VARCHAR(150);",
        "CREATE INDEX IF NOT EXISTS idx_skill_standards_category ON skill_standards(category_name);",
        "CREATE INDEX IF NOT EXISTS idx_skill_standards_subcategory ON skill_standards(subcategory_name);",
        "CREATE INDEX IF NOT EXISTS idx_skill_standards_type ON skill_standards(standard_type);",
    ]
    for stmt in ddl_statements:
        await session.execute(text(stmt))
    await session.commit()
    print("Schema migration for skill_standards completed.")


def load_raw_types() -> dict[str, str]:
    """Load skill types (Specialized Skill, Common Skill, Certification) from local cache."""
    print(f"Step 2: Loading raw types from {RAW_FILE}...")
    if not RAW_FILE.exists():
        print(f"Warning: {RAW_FILE} not found.")
        return {}
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    type_map = {}
    for item in data:
        sid = item.get("id")
        tname = item.get("type", {}).get("name") if isinstance(item.get("type"), dict) else None
        if sid and tname:
            type_map[sid] = tname
    print(f"Loaded {len(type_map)} skill types from local raw dataset.")
    return type_map


async def load_categories_tree() -> dict[str, tuple[str, str]]:
    """Fetch or load categories tree mapping skill_id -> (category_name, subcategory_name)."""
    print(f"Step 3: Fetching categories from {CATEGORIES_URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(CATEGORIES_URL, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    cat_map = {}
    for cat in data.get("children", []):
        cat_name = cat.get("name")
        for subcat in cat.get("children", []):
            subcat_name = subcat.get("name")
            for skill in subcat.get("children", []):
                sid = skill.get("id")
                if sid:
                    cat_map[sid] = (cat_name, subcat_name)

    print(f"Extracted {len(cat_map)} skill category mappings from Lightcast API.")
    return cat_map


async def sync_standards():
    """Main synchronization logic."""
    async with AsyncSessionLocal() as session:
        await ensure_schema_columns(session)

        type_map = load_raw_types()
        cat_map = await load_categories_tree()

        print("Step 4: Fetching skill_standards from PostgreSQL...")
        res = await session.execute(
            text("SELECT id, standard_code FROM skill_standards WHERE standard_name = 'Lightcast';")
        )
        standards = res.fetchall()
        print(f"Found {len(standards)} Lightcast standard mappings in database.")

        updated_count = 0
        batch_updates = []

        for row in standards:
            row_id = row[0]
            code = row[1]
            if not code:
                continue

            # Resolve type
            stype = type_map.get(code)

            # Resolve category & subcategory
            cat_info = cat_map.get(code)
            cat_name = cat_info[0] if cat_info else None
            subcat_name = cat_info[1] if cat_info else None

            if stype or cat_name or subcat_name:
                batch_updates.append(
                    {
                        "p_id": row_id,
                        "p_type": stype,
                        "p_cat": cat_name,
                        "p_subcat": subcat_name,
                    }
                )

        print(f"Step 5: Applying {len(batch_updates)} updates to skill_standards in batches...")
        batch_size = 500
        for i in range(0, len(batch_updates), batch_size):
            chunk = batch_updates[i : i + batch_size]
            for item in chunk:
                await session.execute(
                    text(
                        """
                        UPDATE skill_standards
                        SET standard_type = :p_type,
                            category_name = :p_cat,
                            subcategory_name = :p_subcat
                        WHERE id = :p_id;
                        """
                    ),
                    item,
                )
            await session.commit()
            updated_count += len(chunk)
            print(f"  Updated {updated_count}/{len(batch_updates)} rows...")

        print("\n=== Sync Summary ===")
        print(f"Total Lightcast standards updated: {updated_count}")

        # Verification query
        stats_res = await session.execute(
            text(
                """
                SELECT 
                    COUNT(*) as total,
                    COUNT(category_name) as with_category,
                    COUNT(subcategory_name) as with_subcategory,
                    COUNT(standard_type) as with_type
                FROM skill_standards 
                WHERE standard_name = 'Lightcast';
                """
            )
        )
        stats = stats_res.fetchone()
        print(f"Stats in DB -> Total: {stats[0]}, With Category: {stats[1]}, With Subcategory: {stats[2]}, With Type: {stats[3]}")


if __name__ == "__main__":
    asyncio.run(sync_standards())
