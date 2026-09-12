"""High-performance script to relink and merge redundant non-standard skills into Lightcast (Phase 2 & 3).

Executes set-based transactional SQL updates to safely migrate foreign keys:
- offer_skills
- cluster_skills
- profile_skills
- diagnostic_skills
- cluster_skill_trends
- skill_relations
- skill_aliases

Usage:
    .venv/Scripts/python scripts/relink_and_merge_lightcast.py [--dry-run]
"""

import asyncio
import os
import re
import sys
from uuid import UUID

from sqlalchemy import text

# Append parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.database import AsyncSessionLocal


def normalize_str(s: str) -> str:
    """Strip parentheses and keep only alphanumeric lowercase chars."""
    base = re.sub(r"\s*\(.*?\)\s*", "", s)
    return re.sub(r"[^a-z0-9]", "", base.lower())


async def relink_and_merge(dry_run: bool = False) -> None:
    async with AsyncSessionLocal() as session:
        print(f"=== PHASE 2 & 3: FAST SET-BASED RELINK & MERGE (dry_run={dry_run}) ===")

        # 1. Fetch all Lightcast skills
        lc_query = text("""
            SELECT s.skill_id, s.name, ss.standard_code
            FROM skills s
            JOIN skill_standards ss ON s.skill_id = ss.skill_id
            WHERE ss.standard_name = 'Lightcast';
        """)
        lc_rows = (await session.execute(lc_query)).fetchall()
        lc_by_norm: dict[str, tuple[UUID, str]] = {}
        lc_by_full_norm: dict[str, tuple[UUID, str]] = {}
        for row in lc_rows:
            sk_id, sk_name, _ = row
            norm = normalize_str(sk_name)
            full_norm = re.sub(r"[^a-z0-9]", "", sk_name.lower())
            if norm and norm not in lc_by_norm:
                lc_by_norm[norm] = (sk_id, sk_name)
            if full_norm and full_norm not in lc_by_full_norm:
                lc_by_full_norm[full_norm] = (sk_id, sk_name)

        # 2. Fetch all remaining non-standard skills
        ns_query = text("""
            SELECT s.skill_id, s.name, s.nature
            FROM skills s
            LEFT JOIN skill_standards ss ON s.skill_id = ss.skill_id
            WHERE ss.skill_id IS NULL;
        """)
        ns_rows = (await session.execute(ns_query)).fetchall()
        print(f"Total non-standard skills in database: {len(ns_rows)}")

        # Explicit known mappings
        explicit_mappings = {
            "aws": "Amazon Web Services",
            "azure": "Microsoft Azure",
            "gcp": "Google Cloud Platform (GCP)",
            "react": "React.js",
            "node": "Node.js",
            "mysql": "MySql",
            "postgresql": "postgresql",
            "postgres": "postgresql",
            "c#": "C# (Programming Language)",
            "c++": "C++ (Programming Language)",
            "c": "C (Programming Language)",
            "python (no mencionado explícitamente pero se asume por el contexto de la empresa)": "Python (Programming Language)",
            "python(nomencionadoexplícitamenteperoseasumeporelcontextodelaempresa)": "Python (Programming Language)",
            "docker(nomencionadoexplícitamenteperoseasumeporelcontextodelaempresa)": "Docker (Software)",
            "aws(nomencionadoexplícitamenteperoseasumeporelcontextodelaempresa)": "Amazon Web Services",
            "amazonwebservices(aws)": "Amazon Web Services",
            "jenkins (herramientas para la gestión de la configuración del software)": "jenkins",
            "java (programación informática)": "Java (Programming Language)",
            "aspnetcore": "ASP.NET Core",
            "apacheairflow": "Apache Airflow",
        }

        # Build list of (source_id, target_id, source_name, target_name)
        merge_list: list[tuple[UUID, UUID, str, str]] = []

        for row in ns_rows:
            s_id, s_name, _ = row
            s_lower = s_name.lower().strip()
            s_norm = normalize_str(s_name)
            s_full_norm = re.sub(r"[^a-z0-9]", "", s_lower)

            target_lc: tuple[UUID, str] | None = None

            if s_lower in explicit_mappings:
                target_name = explicit_mappings[s_lower]
                t_norm = normalize_str(target_name)
                t_full = re.sub(r"[^a-z0-9]", "", target_name.lower())
                target_lc = lc_by_norm.get(t_norm) or lc_by_full_norm.get(t_full)
            elif s_norm in lc_by_norm and s_norm not in ("c", "r"):
                target_lc = lc_by_norm[s_norm]
            elif s_full_norm in lc_by_full_norm and s_full_norm not in ("c", "r"):
                target_lc = lc_by_full_norm[s_full_norm]

            if target_lc and target_lc[0] != s_id:
                merge_list.append((s_id, target_lc[0], s_name, target_lc[1]))

        print(f"Total skills targeted for merge: {len(merge_list)}")

        # Print sample mappings
        for src_id, tgt_id, src_name, tgt_name in merge_list[:15]:
            print(f"  Mapping: '{src_name}' -> '{tgt_name}'")

        if not merge_list:
            print("No skills to merge.")
            return

        # 3. Perform Set-Based Migrations
        for src_id, tgt_id, src_name, tgt_name in merge_list:
            # offer_skills
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

            # cluster_skills
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

            # profile_skills
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

            # diagnostic_skills
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

            # cluster_skill_trends
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

            # skill_relations
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

            # skill_aliases:
            # 1. Delete aliases on src_id that already exist on tgt_id or are hallucinations
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

            # 2. Move remaining aliases from source to target
            await session.execute(text("""
                UPDATE skill_aliases SET skill_id = :tgt_id WHERE skill_id = :src_id;
            """), {"src_id": src_id, "tgt_id": tgt_id})

            # 3. If src_name is not already an alias anywhere, add it for tgt_id
            clean_alias = src_name.strip()
            if "no mencionado" not in clean_alias.lower() and len(clean_alias) < 50:
                await session.execute(text("""
                    INSERT INTO skill_aliases (alias_id, skill_id, alias_name)
                    SELECT gen_random_uuid(), :tgt_id, CAST(:alias_name AS VARCHAR)
                    WHERE NOT EXISTS (
                        SELECT 1 FROM skill_aliases WHERE LOWER(alias_name) = LOWER(CAST(:alias_name AS VARCHAR))
                    );
                """), {"tgt_id": tgt_id, "alias_name": clean_alias})

            # Delete source skill
            await session.execute(text("""
                DELETE FROM skills WHERE skill_id = :src_id;
            """), {"src_id": src_id})

        if not dry_run:
            await session.commit()
            print(f"Successfully committed all {len(merge_list)} skill merges!")
        else:
            await session.rollback()
            print("[DRY-RUN] Rollback executed successfully.")


if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    asyncio.run(relink_and_merge(dry_run=is_dry_run))
