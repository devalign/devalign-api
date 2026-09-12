"""Script to resolve typographic duplicates and remaining canonical mergers (Phase 3).

1. Merges remaining non-standard skills into Lightcast:
   - css, CSS -> Cascading Style Sheets (CSS)
   - html, HTML -> HyperText Markup Language (HTML)
   - api, API -> Application Programming Interface (API)
   - agile, Agile -> Agile Software Development
   - vue, Vue -> Vue.js
   - sqlserver, sql server, SQL Server -> Microsoft SQL Servers
   - golang, Golang -> Go (Programming Language)
   - front-endprogramming -> Front End (Software Engineering)
   - back-endprogramming -> Back End (Software Engineering)
   - fullstack, full-stackprogramming -> Full Stack Development
   - rest, REST API, apisrest -> RESTful API
2. Merges non-standard case variations into canonical clean title:
   - nextjs -> Next.js
   - nosql -> NoSQL
   - powerbi -> Power BI
   - ux/ui -> UX/UI

Usage:
    .venv/Scripts/python scripts/cleanup_typographic_and_hallucinations.py [--dry-run]
"""

import asyncio
import os
import sys
from uuid import UUID

from sqlalchemy import text

# Append parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.database import AsyncSessionLocal


async def cleanup_phase_3(dry_run: bool = False) -> None:
    async with AsyncSessionLocal() as session:
        print(f"=== PHASE 3: TYPOGRAPHIC & REMAINING CANONICAL MERGES (dry_run={dry_run}) ===")

        # Map source skill name (lowercase) -> target skill name (exact)
        target_mappings = {
            # Lightcast targets
            "css": "Cascading Style Sheets (CSS)",
            "html": "HyperText Markup Language (HTML)",
            "api": "Application Programming Interface (API)",
            "agile": "Agile Software Development",
            "vue": "Vue.js",
            "sqlserver": "Microsoft SQL Servers",
            "sql server": "Microsoft SQL Servers",
            "golang": "Go (Programming Language)",
            "front-endprogramming": "Front End (Software Engineering)",
            "back-endprogramming": "Back End (Software Engineering)",
            "fullstack": "Full Stack Development",
            "full-stackprogramming": "Full Stack Development",
            "rest": "RESTful API",
            "rest api": "RESTful API",
            "apisrest": "RESTful API",
            # Non-standard title targets
            "nextjs": "Next.js",
            "nosql": "NoSQL",
            "powerbi": "Power BI",
            "ux/ui": "UX/UI",
        }

        # Find target skill IDs
        target_names = list(set(target_mappings.values()))
        tgt_query = text("""
            SELECT skill_id, name FROM skills WHERE name = ANY(:names);
        """)
        tgt_rows = (await session.execute(tgt_query, {"names": target_names})).fetchall()
        tgt_name_to_id = {row[1]: row[0] for row in tgt_rows}

        # Check if Next.js, NoSQL, Power BI, UX/UI exist
        # If not, find by ILIKE
        for t_name in ["Next.js", "NoSQL", "Power BI", "UX/UI"]:
            if t_name not in tgt_name_to_id:
                res = (await session.execute(
                    text("SELECT skill_id, name FROM skills WHERE LOWER(name) = LOWER(:t_name) ORDER BY length(name) LIMIT 1;"),
                    {"t_name": t_name}
                )).fetchone()
                if res:
                    tgt_name_to_id[t_name] = res[0]

        # Fetch all skills to match sources
        all_skills_rows = (await session.execute(text("SELECT skill_id, name FROM skills;"))).fetchall()
        skills_by_name = {row[1].lower().strip(): (row[0], row[1]) for row in all_skills_rows}

        merges_to_execute: list[tuple[UUID, UUID, str, str]] = []
        for src_name_lower, tgt_name in target_mappings.items():
            tgt_id = tgt_name_to_id.get(tgt_name)
            if not tgt_id:
                print(f"Warning: Target skill '{tgt_name}' not found. Skipping.")
                continue

            # Check if source skill exists
            if src_name_lower in skills_by_name:
                src_id, actual_src_name = skills_by_name[src_name_lower]
                if src_id != tgt_id:
                    merges_to_execute.append((src_id, tgt_id, actual_src_name, tgt_name))

        print(f"Total skills to merge in Phase 3: {len(merges_to_execute)}")
        for src_id, tgt_id, src_name, tgt_name in merges_to_execute:
            print(f"  Mapping: '{src_name}' -> '{tgt_name}'")

        if not merges_to_execute:
            print("No skills to merge.")
            return

        for src_id, tgt_id, src_name, tgt_name in merges_to_execute:
            # 1. offer_skills
            await session.execute(text("""
                DELETE FROM offer_skills os_src
                WHERE os_src.skill_id = :src_id
                  AND EXISTS (
                      SELECT 1 FROM offer_skills os_tgt
                      WHERE os_tgt.job_offer_id = os_src.job_offer_id
                        AND os_tgt.skill_id = :tgt_id
                  );
            """), {"src_id": src_id, "tgt_id": tgt_id})
            await session.execute(text("""
                UPDATE offer_skills SET skill_id = :tgt_id WHERE skill_id = :src_id;
            """), {"src_id": src_id, "tgt_id": tgt_id})

            # 2. cluster_skills
            await session.execute(text("""
                DELETE FROM cluster_skills cs_src
                WHERE cs_src.skill_id = :src_id
                  AND EXISTS (
                      SELECT 1 FROM cluster_skills cs_tgt
                      WHERE cs_tgt.cluster_id = cs_src.cluster_id
                        AND cs_tgt.skill_id = :tgt_id
                  );
            """), {"src_id": src_id, "tgt_id": tgt_id})
            await session.execute(text("""
                UPDATE cluster_skills SET skill_id = :tgt_id WHERE skill_id = :src_id;
            """), {"src_id": src_id, "tgt_id": tgt_id})

            # 3. profile_skills
            await session.execute(text("""
                DELETE FROM profile_skills ps_src
                WHERE ps_src.skill_id = :src_id
                  AND EXISTS (
                      SELECT 1 FROM profile_skills ps_tgt
                      WHERE ps_tgt.profile_id = ps_src.profile_id
                        AND ps_tgt.skill_id = :tgt_id
                  );
            """), {"src_id": src_id, "tgt_id": tgt_id})
            await session.execute(text("""
                UPDATE profile_skills SET skill_id = :tgt_id WHERE skill_id = :src_id;
            """), {"src_id": src_id, "tgt_id": tgt_id})

            # 4. diagnostic_skills
            await session.execute(text("""
                DELETE FROM diagnostic_skills ds_src
                WHERE ds_src.skill_id = :src_id
                  AND EXISTS (
                      SELECT 1 FROM diagnostic_skills ds_tgt
                      WHERE ds_tgt.diagnostic_id = ds_src.diagnostic_id
                        AND ds_tgt.skill_id = :tgt_id
                  );
            """), {"src_id": src_id, "tgt_id": tgt_id})
            await session.execute(text("""
                UPDATE diagnostic_skills SET skill_id = :tgt_id WHERE skill_id = :src_id;
            """), {"src_id": src_id, "tgt_id": tgt_id})

            # 5. cluster_skill_trends
            await session.execute(text("""
                DELETE FROM cluster_skill_trends cst_src
                WHERE cst_src.skill_id = :src_id
                  AND EXISTS (
                      SELECT 1 FROM cluster_skill_trends cst_tgt
                      WHERE cst_tgt.cluster_id = cst_src.cluster_id
                        AND cst_tgt.recorded_at = cst_src.recorded_at
                        AND cst_tgt.skill_id = :tgt_id
                  );
            """), {"src_id": src_id, "tgt_id": tgt_id})
            await session.execute(text("""
                UPDATE cluster_skill_trends SET skill_id = :tgt_id WHERE skill_id = :src_id;
            """), {"src_id": src_id, "tgt_id": tgt_id})

            # 6. skill_relations
            await session.execute(text("""
                DELETE FROM skill_relations 
                WHERE (source_skill_id = :src_id AND target_skill_id = :tgt_id)
                   OR (source_skill_id = :tgt_id AND target_skill_id = :src_id);
            """), {"src_id": src_id, "tgt_id": tgt_id})

            await session.execute(text("""
                DELETE FROM skill_relations sr_src
                WHERE sr_src.source_skill_id = :src_id
                  AND EXISTS (
                      SELECT 1 FROM skill_relations sr_tgt
                      WHERE sr_tgt.source_skill_id = :tgt_id
                        AND sr_tgt.target_skill_id = sr_src.target_skill_id
                        AND sr_tgt.relation_type = sr_src.relation_type
                  );
            """), {"src_id": src_id, "tgt_id": tgt_id})
            await session.execute(text("""
                UPDATE skill_relations SET source_skill_id = :tgt_id WHERE source_skill_id = :src_id;
            """), {"src_id": src_id, "tgt_id": tgt_id})

            await session.execute(text("""
                DELETE FROM skill_relations sr_src
                WHERE sr_src.target_skill_id = :src_id
                  AND EXISTS (
                      SELECT 1 FROM skill_relations sr_tgt
                      WHERE sr_tgt.target_skill_id = :tgt_id
                        AND sr_tgt.source_skill_id = sr_src.source_skill_id
                        AND sr_tgt.relation_type = sr_src.relation_type
                  );
            """), {"src_id": src_id, "tgt_id": tgt_id})
            await session.execute(text("""
                UPDATE skill_relations SET target_skill_id = :tgt_id WHERE target_skill_id = :src_id;
            """), {"src_id": src_id, "tgt_id": tgt_id})

            # 7. skill_aliases
            await session.execute(text("""
                DELETE FROM skill_aliases sa_src
                WHERE sa_src.skill_id = :src_id
                  AND (
                      LOWER(sa_src.alias_name) LIKE '%no mencionado%'
                      OR EXISTS (
                          SELECT 1 FROM skill_aliases sa_tgt
                          WHERE sa_tgt.skill_id = :tgt_id AND LOWER(sa_tgt.alias_name) = LOWER(sa_src.alias_name)
                      )
                  );
            """), {"src_id": src_id, "tgt_id": tgt_id})

            await session.execute(text("""
                UPDATE skill_aliases SET skill_id = :tgt_id WHERE skill_id = :src_id;
            """), {"src_id": src_id, "tgt_id": tgt_id})

            clean_alias = src_name.strip()
            if "no mencionado" not in clean_alias.lower() and len(clean_alias) < 50:
                await session.execute(text("""
                    INSERT INTO skill_aliases (alias_id, skill_id, alias_name)
                    SELECT gen_random_uuid(), :tgt_id, CAST(:alias_name AS VARCHAR)
                    WHERE NOT EXISTS (
                        SELECT 1 FROM skill_aliases WHERE LOWER(alias_name) = LOWER(CAST(:alias_name AS VARCHAR))
                    );
                """), {"tgt_id": tgt_id, "alias_name": clean_alias})

            # 8. Delete source skill
            await session.execute(text("""
                DELETE FROM skills WHERE skill_id = :src_id;
            """), {"src_id": src_id})

        if not dry_run:
            await session.commit()
            print(f"Successfully committed all {len(merges_to_execute)} Phase 3 skill merges!")
        else:
            await session.rollback()
            print("[DRY-RUN] Rollback executed successfully.")


if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    asyncio.run(cleanup_phase_3(dry_run=is_dry_run))
