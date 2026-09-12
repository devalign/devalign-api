"""Transactionally relink and merge non-standard skills into Lightcast and consolidate internal duplicates.

Migrates foreign keys across:
- offer_skills
- cluster_skills
- profile_skills
- diagnostic_skills
- cluster_skill_trends
- skill_relations
- skill_aliases

Consolidates:
- Grupo 1: Non-standard skills to Lightcast canonicals (documentation -> Software Documentation, agile -> Agile Methodology, etc.)
- Grupo 2: Internal duplicates (Excel, excelvba -> Microsoft Excel, gsuite -> Google Workspace, qa -> Quality Assurance (QA))
- Grupo 3: Junk scraper strings (herramientassimilaresapowerbi -> Power BI, Manejo de herramientas... -> Microsoft Excel, delete garbage)
"""

import asyncio
import os
import sys
from uuid import UUID

from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.database import AsyncSessionLocal


MAPPINGS = {
    # Grupo 1: Non-standard to Lightcast
    "documentation": "Software Documentation",
    "agile": "Agile Methodology",
    "softwaredecódigoabierto": "Open Source Development",
    "herramientas para la gestión de la configuración del software": "Software Configuration Management",
    "basesdedatosrelacionales": "Relational Database Management Systems",
    "puertasdeenlaceapi": "Api Gateway",
    "pruebasdecargayrendimiento": "Load Testing",
    "productmanagement": "Software Product Management",
    "productdevelopment": "Agile Product Development",
    "cloudformation": "AWS CloudFormation",
    "microsoftsql": "Microsoft SQL Servers",
    "sharepoint": "SharePoint Development",
    "jwt": "JSON Web Token (JWT)",
    "Airflow": "Apache Airflow",
    "jmeter": "Apache JMeter",
    "Kafka": "Apache Kafka",
    "TDD": "Test-Driven Development (TDD)",
    "OWASP": "Open Web Application Security Project (OWASP)",
    "Lambda": "AWS Lambda",
    "vpc": "Amazon Virtual Private Cloud (VPC)",
    "xray": "AWS X-Ray",
    "eventdriven": "Event-Driven Programming",
    "prototyping": "Software Prototyping",
    "linq": "LINQ To SQL",
    "weblogic": "Oracle WebLogic Server",
    "rails": "Ruby On Rails",
    "ruby/rails": "Ruby On Rails",
    "Jira Software": "jira",
    "Maven": "apache maven",
    "Postman": "Postman API Platform",
    "SQL Server": "Microsoft SQL Servers",
    "Stripe": "Stripe.net",
    "Mongo": "mongodb",
    "net": ".NET Development",
    "netcore": "ASP.NET Core",
    "aspnetc": "ASP.NET Core",
    "c#/net": ".NET Development",
    "ios": "Apple IOS",
    "apple": "Apple Developer Tools",
    "Windows Server": "Microsoft Windows Server Administration",
    "Windows": "Microsoft Windows Server Administration",
    "qlik": "Qlik Sense (Data Analytics Software)",
    "routing": "Network Routing",
    "switching": "Cisco Certified Network Associate (CCNA) Routing And Switching",
    "ciberseguridad": "Cybersecurity Compliance",
    "modeladodebasesdedatos": "Relational Database Design",
    "infrastructure": "Cloud Infrastructure",
    "azureai": "Microsoft Certified: Azure AI Fundamentals",
    "azureaifoundry": "Microsoft Certified: Azure AI Fundamentals",
    "apirest": "API Design",
    "apis": "Java APIs",
    "api": "API Design",
    "CSS3": "Cascading Style Sheets (CSS)",
    "css": "Cascading Style Sheets (CSS)",
    "HTML5": "HTML",
    "html": "HTML",
    "vue": "Vue Components",
    "JBoss": "JBoss Developer Studio",
    "crm": "Customer Relationship Management (CRM) Software",
    "research": "Research And Development",
    "testing": "Software Testing",
    # Grupo 2: Internal consolidations
    "Excel": "Microsoft Excel",
    "excelvba": "Microsoft Excel",
    "Manejo de herramientas ofimáticas (Word y Excel)": "Microsoft Excel",
    "gsuite": "Google Workspace",
    # Grupo 3: Junk to valid
    "herramientassimilaresapowerbi(noespecificado)": "Power BI",
}

PURE_JUNK_TO_DELETE = [
    "lenguajedeprogramación",
    "estrategiasdepruebasdealtonivel",
]


async def run_consolidation(dry_run: bool = False) -> None:
    async with AsyncSessionLocal() as session:
        print(f"=== STARTING NON-STANDARD SKILLS CONSOLIDATION (dry_run={dry_run}) ===")

        # 1. Resolve source and target IDs
        resolved_merges: list[tuple[UUID, UUID, str, str]] = []

        for src_name, tgt_name in MAPPINGS.items():
            s_res = await session.execute(
                text("SELECT skill_id, name FROM skills WHERE LOWER(name) = LOWER(:n)"),
                {"n": src_name},
            )
            s_row = s_res.fetchone()
            t_res = await session.execute(
                text("SELECT skill_id, name FROM skills WHERE LOWER(name) = LOWER(:n)"),
                {"n": tgt_name},
            )
            t_row = t_res.fetchone()

            if s_row and t_row:
                if s_row[0] != t_row[0]:
                    resolved_merges.append((s_row[0], t_row[0], s_row[1], t_row[1]))
            else:
                print(f"WARNING: Could not resolve pair '{src_name}' -> '{tgt_name}'")

        print(f"Total resolved skill merges to execute: {len(resolved_merges)}")

        # 2. Process each merge
        for idx, (src_id, tgt_id, src_name, tgt_name) in enumerate(resolved_merges, 1):
            print(f"[{idx}/{len(resolved_merges)}] Merging '{src_name}' -> '{tgt_name}'")

            # A. Deduplicate offer_skills where (job_offer_id, tgt_id) already exists
            await session.execute(
                text("""
                DELETE FROM offer_skills
                WHERE skill_id = :src_id
                  AND job_offer_id IN (
                      SELECT job_offer_id FROM offer_skills WHERE skill_id = :tgt_id
                  );
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )

            # Relink remaining offer_skills
            res_off = await session.execute(
                text("""
                UPDATE offer_skills
                SET skill_id = :tgt_id
                WHERE skill_id = :src_id;
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )
            if res_off.rowcount > 0:
                print(f"   -> Relinked {res_off.rowcount} offer_skills")

            # B. Deduplicate cluster_skills
            await session.execute(
                text("""
                DELETE FROM cluster_skills
                WHERE skill_id = :src_id
                  AND cluster_id IN (
                      SELECT cluster_id FROM cluster_skills WHERE skill_id = :tgt_id
                  );
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )

            res_clust = await session.execute(
                text("""
                UPDATE cluster_skills
                SET skill_id = :tgt_id
                WHERE skill_id = :src_id;
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )
            if res_clust.rowcount > 0:
                print(f"   -> Relinked {res_clust.rowcount} cluster_skills")

            # C. Deduplicate profile_skills
            await session.execute(
                text("""
                DELETE FROM profile_skills
                WHERE skill_id = :src_id
                  AND profile_id IN (
                      SELECT profile_id FROM profile_skills WHERE skill_id = :tgt_id
                  );
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )

            res_prof = await session.execute(
                text("""
                UPDATE profile_skills
                SET skill_id = :tgt_id
                WHERE skill_id = :src_id;
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )
            if res_prof.rowcount > 0:
                print(f"   -> Relinked {res_prof.rowcount} profile_skills")

            # D. Deduplicate diagnostic_skills
            await session.execute(
                text("""
                DELETE FROM diagnostic_skills
                WHERE skill_id = :src_id
                  AND diagnostic_id IN (
                      SELECT diagnostic_id FROM diagnostic_skills WHERE skill_id = :tgt_id
                  );
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )

            res_diag = await session.execute(
                text("""
                UPDATE diagnostic_skills
                SET skill_id = :tgt_id
                WHERE skill_id = :src_id;
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )
            if res_diag.rowcount > 0:
                print(f"   -> Relinked {res_diag.rowcount} diagnostic_skills")

            # E. Cluster skill trends
            await session.execute(
                text("""
                DELETE FROM cluster_skill_trends
                WHERE skill_id = :src_id
                  AND (cluster_id, recorded_at) IN (
                      SELECT cluster_id, recorded_at FROM cluster_skill_trends WHERE skill_id = :tgt_id
                  );
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )
            await session.execute(
                text("""
                UPDATE cluster_skill_trends
                SET skill_id = :tgt_id
                WHERE skill_id = :src_id;
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )

            # F. Skill relations
            await session.execute(
                text("""
                DELETE FROM skill_relations
                WHERE source_skill_id = :src_id
                  AND (target_skill_id, relation_type) IN (
                      SELECT target_skill_id, relation_type FROM skill_relations WHERE source_skill_id = :tgt_id
                  );
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )
            await session.execute(
                text("""
                UPDATE skill_relations
                SET source_skill_id = :tgt_id
                WHERE source_skill_id = :src_id;
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )

            await session.execute(
                text("""
                DELETE FROM skill_relations
                WHERE target_skill_id = :src_id
                  AND (source_skill_id, relation_type) IN (
                      SELECT source_skill_id, relation_type FROM skill_relations WHERE target_skill_id = :tgt_id
                  );
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )
            await session.execute(
                text("""
                UPDATE skill_relations
                SET target_skill_id = :tgt_id
                WHERE target_skill_id = :src_id;
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )

            # Prevent self-referential relations
            await session.execute(
                text("""
                DELETE FROM skill_relations
                WHERE source_skill_id = target_skill_id;
            """)
            )

            # G. Skill aliases
            # First, delete any alias on source that already exists on target or globally
            await session.execute(
                text("""
                DELETE FROM skill_aliases
                WHERE skill_id = :src_id
                  AND LOWER(alias_name) IN (
                      SELECT LOWER(alias_name) FROM skill_aliases WHERE skill_id = :tgt_id
                  );
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )

            await session.execute(
                text("""
                UPDATE skill_aliases
                SET skill_id = :tgt_id
                WHERE skill_id = :src_id;
            """),
                {"src_id": src_id, "tgt_id": tgt_id},
            )

            # Add source name as alias on target if not long/junk and not already present
            clean_alias = src_name.strip().lower()
            if len(clean_alias) <= 40 and not any(
                p in clean_alias
                for p in ["herramientas", "manejo", "gestión", "código", "programación"]
            ):
                await session.execute(
                    text("""
                    INSERT INTO skill_aliases (alias_id, skill_id, alias_name)
                    SELECT gen_random_uuid(), :tgt_id, CAST(:alias AS VARCHAR)
                    WHERE NOT EXISTS (
                        SELECT 1 FROM skill_aliases WHERE LOWER(alias_name) = LOWER(CAST(:alias AS VARCHAR))
                    );
                """),
                    {"tgt_id": tgt_id, "alias": clean_alias},
                )

            # H. Delete source skill
            await session.execute(
                text("DELETE FROM skills WHERE skill_id = :src_id"), {"src_id": src_id}
            )

        # 3. Rename 'qa' to 'Quality Assurance (QA)' and add 'qa' alias
        qa_res = await session.execute(
            text("SELECT skill_id FROM skills WHERE LOWER(name) = 'qa'")
        )
        qa_row = qa_res.fetchone()
        if qa_row:
            qa_id = qa_row[0]
            print("Renaming 'qa' -> 'Quality Assurance (QA)'")
            await session.execute(
                text("""
                UPDATE skills
                SET name = 'Quality Assurance (QA)'
                WHERE skill_id = :id;
            """),
                {"id": qa_id},
            )
            await session.execute(
                text("""
                INSERT INTO skill_aliases (alias_id, skill_id, alias_name)
                SELECT gen_random_uuid(), :id, 'qa'
                WHERE NOT EXISTS (
                    SELECT 1 FROM skill_aliases WHERE LOWER(alias_name) = 'qa'
                );
            """),
                {"id": qa_id},
            )

        # 4. Purge pure junk skills
        for junk_name in PURE_JUNK_TO_DELETE:
            j_res = await session.execute(
                text("SELECT skill_id, name FROM skills WHERE LOWER(name) = LOWER(:n)"),
                {"n": junk_name},
            )
            j_row = j_res.fetchone()
            if j_row:
                print(f"Purging pure junk skill '{j_row[1]}'")
                await session.execute(
                    text("DELETE FROM offer_skills WHERE skill_id = :id"), {"id": j_row[0]}
                )
                await session.execute(
                    text("DELETE FROM skills WHERE skill_id = :id"), {"id": j_row[0]}
                )

        if dry_run:
            print("\n[DRY-RUN] Rolling back all changes. Zero modifications written.")
            await session.rollback()
        else:
            print("\n[COMMIT] Committing all consolidations to database...")
            await session.commit()
            print("Successfully committed!")

        # 5. Final count recap
        res_count = await session.execute(text("SELECT COUNT(*) FROM skills;"))
        total_skills = res_count.scalar()
        res_lc = await session.execute(
            text("SELECT COUNT(*) FROM skill_standards WHERE standard_name = 'Lightcast';")
        )
        total_lc = res_lc.scalar()
        res_non = await session.execute(
            text("""
            SELECT COUNT(*) FROM skills s
            LEFT JOIN skill_standards ss ON ss.skill_id = s.skill_id
            WHERE ss.skill_id IS NULL;
        """)
        )
        total_non = res_non.scalar()
        print(f"\nFinal State: Total={total_skills}, Lightcast={total_lc}, Non-Standard={total_non}")


if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    asyncio.run(run_consolidation(dry_run=is_dry_run))
