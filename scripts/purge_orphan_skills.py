"""Script to safely purge orphan skills (Phase 1).

1. Identifies orphan skills without any standard and without any links in:
   - offer_skills
   - cluster_skills
   - diagnostic_skills
   - profile_skills
   - skill_relations
   - cluster_skill_trends
2. Rescues valuable aliases (e.g. 'python3', 'c sharp') that point to orphans
   and re-points them to the canonical Lightcast skill.
3. Deletes the unlinked orphan skills in a single transaction.

Usage:
    .venv/Scripts/python scripts/purge_orphan_skills.py [--dry-run]
"""

import asyncio
import os
import sys

from sqlalchemy import delete, select, text

# Append parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_engine.infrastructure.models import SkillAliasModel, SkillModel
from src.shared.database import AsyncSessionLocal


async def purge_orphans(dry_run: bool = False) -> None:
    async with AsyncSessionLocal() as session:
        print(f"=== PHASE 1: PURGE ORPHAN SKILLS (dry_run={dry_run}) ===")

        # 1. Fetch all orphan skill IDs
        find_orphans_query = text("""
            WITH linked_skills AS (
                SELECT skill_id FROM offer_skills
                UNION
                SELECT skill_id FROM cluster_skills
                UNION
                SELECT skill_id FROM diagnostic_skills
                UNION
                SELECT skill_id FROM profile_skills
                UNION
                SELECT source_skill_id AS skill_id FROM skill_relations
                UNION
                SELECT target_skill_id AS skill_id FROM skill_relations
                UNION
                SELECT skill_id FROM cluster_skill_trends
            )
            SELECT s.skill_id, s.name, s.nature, DATE(s.created_at) as created_date
            FROM skills s
            LEFT JOIN skill_standards ss ON s.skill_id = ss.skill_id
            LEFT JOIN linked_skills ls ON s.skill_id = ls.skill_id
            WHERE ss.skill_id IS NULL AND ls.skill_id IS NULL;
        """)

        res = await session.execute(find_orphans_query)
        orphan_rows = res.fetchall()
        orphan_ids = [row[0] for row in orphan_rows]
        print(f"Total orphan skills detected: {len(orphan_ids)}")

        esco_count = sum(1 for row in orphan_rows if str(row[3]) == "2026-07-05")
        concept_count = sum(1 for row in orphan_rows if row[2] == "concept")
        tech_count = sum(1 for row in orphan_rows if row[2] == "tech")
        print(f"  - Residual ESCO (2026-07-05): {esco_count}")
        print(f"  - Concepts: {concept_count}")
        print(f"  - Tech: {tech_count}")

        if not orphan_ids:
            print("No orphan skills to purge.")
            return

        # 2. Rescue valuable aliases from orphan skills to canonical Lightcast skills
        fetch_canonical_targets_query = text("""
            SELECT s.skill_id, s.name, ss.standard_code
            FROM skills s
            JOIN skill_standards ss ON s.skill_id = ss.skill_id
            WHERE ss.standard_name = 'Lightcast'
              AND s.name IN (
                  'Python (Programming Language)',
                  'C# (Programming Language)',
                  'C++ (Programming Language)',
                  'C (Programming Language)',
                  'Java (Programming Language)',
                  'JavaScript (Programming Language)',
                  'SQL (Programming Language)',
                  'Docker (Software)',
                  'Git (Version Control System)'
              );
        """)
        canon_res = await session.execute(fetch_canonical_targets_query)
        canon_skills = {row[1]: row[0] for row in canon_res.fetchall()}

        orphan_to_canon = {
            "python (programación informática)": canon_skills.get("Python (Programming Language)"),
            "c#": canon_skills.get("C# (Programming Language)"),
            "c++": canon_skills.get("C++ (Programming Language)"),
            "java (programación informática)": canon_skills.get("Java (Programming Language)"),
        }

        rescued_aliases_count = 0
        for orphan_name, target_id in orphan_to_canon.items():
            if not target_id:
                continue

            matching_orphan_ids = [row[0] for row in orphan_rows if row[1].lower() == orphan_name.lower()]
            if not matching_orphan_ids:
                continue

            for m_id in matching_orphan_ids:
                aliases_res = await session.execute(
                    select(SkillAliasModel).where(SkillAliasModel.skill_id == m_id)
                )
                aliases = aliases_res.scalars().all()
                for al in aliases:
                    existing_alias_res = await session.execute(
                        select(SkillAliasModel).where(
                            SkillAliasModel.skill_id == target_id,
                            SkillAliasModel.alias_name.ilike(al.alias_name),
                        )
                    )
                    if not existing_alias_res.scalars().first():
                        print(f"  [RESCUE ALIAS] '{al.alias_name}' -> reassigning to canonical target ({target_id})")
                        if not dry_run:
                            al.skill_id = target_id
                        rescued_aliases_count += 1

        print(f"Rescued {rescued_aliases_count} valuable aliases.")

        # 3. Purge orphan skills
        if not dry_run:
            print(f"Deleting {len(orphan_ids)} orphan skills...")
            chunk_size = 500
            for i in range(0, len(orphan_ids), chunk_size):
                chunk = orphan_ids[i : i + chunk_size]
                await session.execute(delete(SkillModel).where(SkillModel.skill_id.in_(chunk)))
            await session.commit()
            print("Purge committed successfully.")
        else:
            print("[DRY-RUN] No changes were committed.")


if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    asyncio.run(purge_orphans(dry_run=is_dry_run))
