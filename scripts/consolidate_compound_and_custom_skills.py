"""Consolidate compound, lowercase, and redundant Custom skills in Devalign.

Performs:
1. Decoupling of slashed skills:
   - 'javascript/typescript' -> re-link offer_skills to 'JavaScript (Programming Language)' and 'TypeScript'
   - 'ai/ml' -> re-link offer_skills to 'Artificial Intelligence' and 'Machine Learning'
   - 'net8+/aspnetcore' -> re-link to 'ASP.NET Core' and '.NET'
   - 'sappi/po' -> re-link to 'SAP'
2. Merging concatenated lowercase custom skills into canonical Lightcast/Custom skills
3. Renaming prominent lowercase skills to Title Case / canonical casing (e.g. typescript -> TypeScript)
4. Purging corrupted phrases, scraping noise, and non-technical generic terms

Usage:
    .venv/Scripts/python scripts/consolidate_compound_and_custom_skills.py [--dry-run]
"""

import argparse
import asyncio
import os
import sys
from uuid import UUID

from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.shared.database import AsyncSessionLocal


MERGES = {
    # Concatenated lowercase to canonical
    "javaweb": "Java (Programming Language)",
    "html": "HyperText Markup Language (HTML)",
    "phyton": "Python (Programming Language)",
    "adobexd": "Adobe XD",
    "aftereffects": "Adobe After Effects",
    "photoshop": "Adobe Photoshop",
    "illustrator": "Adobe Illustrator",
    "codereview": "Code Review",
    "datavisualization": "Data Visualization",
    "dataentry": "Data Entry",
    "engineeringmanagement": "Engineering Management",
    "mobiledevelopment": "Mobile Application Development",
    "sitereliability": "Site Reliability Engineering (SRE)",
    "systemdesign": "System Design",
    "teamlead": "Team Leadership",
    "r": "R (Programming Language)",
    "ror": "Ruby On Rails",
    "sass": "Sass (Stylesheet Language)",
    "sftp": "SSH File Transfer Protocol (SFTP)",
    "soap": "Simple Object Access Protocol (SOAP)",
    "sns": "Amazon Simple Notification Service (SNS)",
    "sqs": "Amazon Simple Queue Service (SQS)",
    "tcp/ip": "Transmission Control Protocol / Internet Protocol (TCP/IP)",
    "unity": "Unity (Game Engine)",
    "vmware": "VMware",
    "voip": "Voice Over IP (VOIP)",
    "webhooks": "Webhooks",
    "webpack": "Webpack",
    "wordpress": "WordPress",
    "xml": "XML",
    "confluence": "Atlassian Confluence",
    "datadog": "Datadog",
    "dynamodb": "Amazon DynamoDB",
    "mariadb": "MariaDB",
    "shopify": "Shopify",
    "tensorflow": "TensorFlow",
    "appsscript": "Google Apps Script",
    "awsbedrock": "Amazon Bedrock",
    "awsrds": "Amazon Relational Database Service (RDS)",
    "awss3": "Amazon S3",
    "azurekeyvault": "Azure Key Vault",
    "eventbridge": "Amazon EventBridge",
    "googlecloudvertexai": "Vertex AI",
    "googlevertexai": "Vertex AI",
    "stepfunctions": "AWS Step Functions",
    "microsoft365": "Microsoft 365",
    "microsoftcopilotstudio": "Microsoft Copilot Studio",
    "opentelemetrycollector": "OpenTelemetry",
    "nxmonorepo": "Nx",
    "oauth2": "OAuth",
    "postman": "Postman API Platform",
    "Event-Driven Architecture (EDA)": "Event-Driven Architecture",
    "SOLID Principles": "SOLID",
}

# Proper Casing Renames (name in DB -> new casing)
RENAMES = {
    "typescript": "TypeScript",
    "postgresql": "PostgreSQL",
    "devops": "DevOps",
    "kubernetes": "Kubernetes",
    "microservicios": "Microservicios",
    "linux": "Linux",
    "jira": "Jira",
    "kanban": "Kanban",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "graphql": "GraphQL",
    "jenkins": "Jenkins",
    "cypress": "Cypress",
    "kotlin": "Kotlin",
    "ci/cd": "CI/CD",
    "gitlab": "GitLab",
    "symfony": "Symfony",
    "electron": "Electron",
    "dbeaver": "DBeaver",
    "duckdb": "DuckDB",
    "mixpanel": "Mixpanel",
    "n8n": "n8n",
    "net8": ".NET 8",
    "passport": "Passport.js",
    "prestashop": "PrestaShop",
    "pytest": "PyTest",
    "serilog": "Serilog",
    "sip": "SIP",
    "temporal": "Temporal",
    "typeorm": "TypeORM",
    "rxjs": "RxJS",
    "sdlc": "SDLC",
    "xslt": "XSLT",
    "mvc/hexagonal architecture": "MVC / Hexagonal Architecture",
}

# Corrupted scraping phrases and non-technical noise to purge
PURGES = [
    "macros(deseable)",
    "pruebasdeaplicacionesmvilesyweb",
    "pruebasdeaplicacionesmóvilesyweb",
    "herramientasdeautomatizacin",
    "herramientasdeautomatización",
    "herramientasdegestindepruebas",
    "herramientasdegestióndepruebas",
    "herramientasdeinspeccindecdigo",
    "herramientasdeinspeccióndecódigo",
    "generadoresdegrficosestadsticos",
    "generadoresdegráficosestadísticos",
    "orquestacindeflujosdetrabajo",
    "orquestacióndeflujosdetrabajo",
    "sistemasdepublicacin-suscripcin",
    "sistemasdepublicación-suscripción",
    "serviciosdeinfraestructura",
    "marcosrpc",
    "controldecalidad",
    "descartes",
    "csvlod",
    "aicodingagents",
    "genexus16",
    "mapasgeoreferenciados",
    "chat",
    "diversity",
    "editing",
    "insurance",
    "startup",
    "travel",
    "video",
    "videoproduction",
    "youtube",
    "onboarding",
    "fundraising",
    "stem",
    "themes",
    "rfc",
    "development",
    "digitalcontent",
    "c-suite",
    "bookkeeping",
    "accounting",
    "advertising",
    "customeracquisition",
    "organicmarketing",
    "productstrategy",
    "peoplemanagement",
    "socialmedia",
    "spreadsheets",
    "traditional web",
    "reactive web",
]


async def decouple_slashed_skills(session, dry_run: bool):
    """Decouple compound skills into multiple canonical skills."""
    compound_mappings = [
        ("javascript/typescript", ["JavaScript (Programming Language)", "TypeScript"]),
        ("ai/ml", ["Artificial Intelligence", "Machine Learning"]),
        ("net8+/aspnetcore", [".NET Development", "ASP.NET"]),
        ("sappi/po", ["SAP"]),
    ]

    for source_name, target_names in compound_mappings:
        res = await session.execute(
            text("SELECT skill_id FROM skills WHERE LOWER(name) = LOWER(:s)"),
            {"s": source_name},
        )
        source_id = res.scalar()
        if not source_id:
            continue

        print(f"Decoupling compound skill '{source_name}' -> {target_names}")
        # Find target skill IDs
        target_ids = []
        for tname in target_names:
            tres = await session.execute(
                text("SELECT skill_id FROM skills WHERE LOWER(name) = LOWER(:t)"),
                {"t": tname},
            )
            tid = tres.scalar()
            if tid:
                target_ids.append(tid)

        if not target_ids:
            print(f"  WARNING: No targets found for '{source_name}', skipping.")
            continue

        # Get all offer_skills linked to source
        offers_res = await session.execute(
            text("SELECT job_offer_id, skill_type FROM offer_skills WHERE skill_id = :sid"),
            {"sid": source_id},
        )
        offers = offers_res.all()
        print(f"  Found {len(offers)} offers to re-link to {len(target_ids)} targets")

        if not dry_run:
            for job_offer_id, skill_type in offers:
                for tid in target_ids:
                    await session.execute(
                        text("""
                            INSERT INTO offer_skills (job_offer_id, skill_id, skill_type)
                            VALUES (:jid, :sid, :stype)
                            ON CONFLICT DO NOTHING;
                        """),
                        {"jid": job_offer_id, "sid": tid, "stype": skill_type or "tech"},
                    )

            # Delete source from all junction tables and skills
            await session.execute(text("DELETE FROM offer_skills WHERE skill_id = :sid"), {"sid": source_id})
            await session.execute(text("DELETE FROM cluster_skills WHERE skill_id = :sid"), {"sid": source_id})
            await session.execute(text("DELETE FROM profile_skills WHERE skill_id = :sid"), {"sid": source_id})
            await session.execute(text("DELETE FROM diagnostic_skills WHERE skill_id = :sid"), {"sid": source_id})
            await session.execute(text("DELETE FROM cluster_skill_trends WHERE skill_id = :sid"), {"sid": source_id})
            await session.execute(text("DELETE FROM skill_relations WHERE source_skill_id = :sid OR target_skill_id = :sid"), {"sid": source_id})
            await session.execute(text("DELETE FROM skill_aliases WHERE skill_id = :sid"), {"sid": source_id})
            await session.execute(text("DELETE FROM skill_standards WHERE skill_id = :sid"), {"sid": source_id})
            await session.execute(text("DELETE FROM skills WHERE skill_id = :sid"), {"sid": source_id})


async def merge_skill(session, source_name: str, target_name: str, dry_run: bool):
    """Merge source skill into target skill, preserving aliases and junction rows."""
    res_source = await session.execute(
        text("SELECT skill_id, name FROM skills WHERE LOWER(name) = LOWER(:s)"),
        {"s": source_name},
    )
    source = res_source.fetchone()
    if not source:
        return

    res_target = await session.execute(
        text("SELECT skill_id, name FROM skills WHERE LOWER(name) = LOWER(:t)"),
        {"t": target_name},
    )
    target = res_target.fetchone()
    if not target:
        print(f"Target '{target_name}' not found for source '{source_name}', creating or keeping as Custom.")
        # Capitalize the source skill to target_name if it doesn't exist
        if not dry_run:
            await session.execute(
                text("UPDATE skills SET name = :tname WHERE skill_id = :sid"),
                {"tname": target_name, "sid": source.skill_id},
            )
            await session.execute(
                text("""
                    INSERT INTO skill_aliases (alias_id, skill_id, alias_name)
                    SELECT gen_random_uuid(), :sid, CAST(:alias AS VARCHAR)
                    WHERE NOT EXISTS (SELECT 1 FROM skill_aliases WHERE LOWER(alias_name) = LOWER(CAST(:alias AS VARCHAR)));
                """),
                {"sid": source.skill_id, "alias": source_name.lower()},
            )
        return

    source_id, target_id = source.skill_id, target.skill_id
    if source_id == target_id:
        return

    print(f"Merging '{source.name}' -> '{target.name}'")
    if dry_run:
        return

    # 1. Re-link offer_skills
    await session.execute(
        text("""
            DELETE FROM offer_skills WHERE skill_id = :tid AND job_offer_id IN (
                SELECT job_offer_id FROM offer_skills WHERE skill_id = :sid
            );
        """),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(
        text("UPDATE offer_skills SET skill_id = :tid WHERE skill_id = :sid;"),
        {"tid": target_id, "sid": source_id},
    )

    # 2. Re-link cluster_skills
    await session.execute(
        text("""
            DELETE FROM cluster_skills WHERE skill_id = :tid AND cluster_id IN (
                SELECT cluster_id FROM cluster_skills WHERE skill_id = :sid
            );
        """),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(
        text("UPDATE cluster_skills SET skill_id = :tid WHERE skill_id = :sid;"),
        {"tid": target_id, "sid": source_id},
    )

    # 3. Re-link profile_skills
    await session.execute(
        text("""
            DELETE FROM profile_skills WHERE skill_id = :tid AND profile_id IN (
                SELECT profile_id FROM profile_skills WHERE skill_id = :sid
            );
        """),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(
        text("UPDATE profile_skills SET skill_id = :tid WHERE skill_id = :sid;"),
        {"tid": target_id, "sid": source_id},
    )

    # 4. Re-link diagnostic_skills
    await session.execute(
        text("""
            DELETE FROM diagnostic_skills WHERE skill_id = :tid AND diagnostic_id IN (
                SELECT diagnostic_id FROM diagnostic_skills WHERE skill_id = :sid
            );
        """),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(
        text("UPDATE diagnostic_skills SET skill_id = :tid WHERE skill_id = :sid;"),
        {"tid": target_id, "sid": source_id},
    )

    # 5. Re-link cluster_skill_trends
    await session.execute(
        text("""
            DELETE FROM cluster_skill_trends WHERE skill_id = :tid AND (cluster_id, recorded_at) IN (
                SELECT cluster_id, recorded_at FROM cluster_skill_trends WHERE skill_id = :sid
            );
        """),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(
        text("UPDATE cluster_skill_trends SET skill_id = :tid WHERE skill_id = :sid;"),
        {"tid": target_id, "sid": source_id},
    )

    # 6. Re-link skill_relations
    await session.execute(
        text("""
            DELETE FROM skill_relations WHERE source_skill_id = :tid AND (target_skill_id, relation_type) IN (
                SELECT target_skill_id, relation_type FROM skill_relations WHERE source_skill_id = :sid
            );
        """),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(
        text("UPDATE skill_relations SET source_skill_id = :tid WHERE source_skill_id = :sid;"),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(
        text("""
            DELETE FROM skill_relations WHERE target_skill_id = :tid AND (source_skill_id, relation_type) IN (
                SELECT source_skill_id, relation_type FROM skill_relations WHERE target_skill_id = :sid
            );
        """),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(
        text("UPDATE skill_relations SET target_skill_id = :tid WHERE target_skill_id = :sid;"),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(text("DELETE FROM skill_relations WHERE source_skill_id = target_skill_id;"))

    # 7. Re-link skill_aliases
    await session.execute(
        text("""
            DELETE FROM skill_aliases WHERE skill_id = :tid AND LOWER(alias_name) IN (
                SELECT LOWER(alias_name) FROM skill_aliases WHERE skill_id = :sid
            );
        """),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(
        text("UPDATE skill_aliases SET skill_id = :tid WHERE skill_id = :sid;"),
        {"tid": target_id, "sid": source_id},
    )
    await session.execute(
        text("""
            INSERT INTO skill_aliases (alias_id, skill_id, alias_name)
            SELECT gen_random_uuid(), :tid, CAST(:alias AS VARCHAR)
            WHERE NOT EXISTS (SELECT 1 FROM skill_aliases WHERE LOWER(alias_name) = LOWER(CAST(:alias AS VARCHAR)));
        """),
        {"tid": target_id, "alias": source_name.lower()},
    )

    # 8. Delete source skill_standards and skill
    await session.execute(text("DELETE FROM skill_standards WHERE skill_id = :sid"), {"sid": source_id})
    await session.execute(text("DELETE FROM skills WHERE skill_id = :sid"), {"sid": source_id})


async def rename_skills(session, dry_run: bool):
    """Rename lowercase skills to Title Case / canonical casing."""
    for old_name, new_name in RENAMES.items():
        res = await session.execute(
            text("SELECT skill_id, name FROM skills WHERE LOWER(name) = LOWER(:s)"),
            {"s": old_name},
        )
        row = res.fetchone()
        if not row:
            continue

        print(f"Renaming '{row.name}' -> '{new_name}'")
        if not dry_run:
            await session.execute(
                text("UPDATE skills SET name = :n WHERE skill_id = :sid"),
                {"n": new_name, "sid": row.skill_id},
            )
            await session.execute(
                text("""
                    INSERT INTO skill_aliases (alias_id, skill_id, alias_name)
                    SELECT gen_random_uuid(), :sid, CAST(:alias AS VARCHAR)
                    WHERE NOT EXISTS (SELECT 1 FROM skill_aliases WHERE LOWER(alias_name) = LOWER(CAST(:alias AS VARCHAR)));
                """),
                {"sid": row.skill_id, "alias": old_name.lower()},
            )


async def purge_junk_skills(session, dry_run: bool):
    """Purge junk and non-technical skills."""
    for junk in PURGES:
        res = await session.execute(
            text("SELECT skill_id, name FROM skills WHERE LOWER(name) = LOWER(:j)"),
            {"j": junk},
        )
        row = res.fetchone()
        if not row:
            continue

        print(f"Purging junk skill '{row.name}'")
        if not dry_run:
            sid = row.skill_id
            await session.execute(text("DELETE FROM offer_skills WHERE skill_id = :sid"), {"sid": sid})
            await session.execute(text("DELETE FROM cluster_skills WHERE skill_id = :sid"), {"sid": sid})
            await session.execute(text("DELETE FROM profile_skills WHERE skill_id = :sid"), {"sid": sid})
            await session.execute(text("DELETE FROM diagnostic_skills WHERE skill_id = :sid"), {"sid": sid})
            await session.execute(text("DELETE FROM cluster_skill_trends WHERE skill_id = :sid"), {"sid": sid})
            await session.execute(text("DELETE FROM skill_relations WHERE source_skill_id = :sid OR target_skill_id = :sid"), {"sid": sid})
            await session.execute(text("DELETE FROM skill_aliases WHERE skill_id = :sid"), {"sid": sid})
            await session.execute(text("DELETE FROM skill_standards WHERE skill_id = :sid"), {"sid": sid})
            await session.execute(text("DELETE FROM skills WHERE skill_id = :sid"), {"sid": sid})


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without committing")
    args = parser.parse_args()

    async with AsyncSessionLocal() as session:
        print(f"=== Starting Skill Consolidation (dry_run={args.dry_run}) ===")

        # 1. Decouple slashed skills
        await decouple_slashed_skills(session, args.dry_run)

        # 2. Merge redundant skills
        for source_name, target_name in MERGES.items():
            await merge_skill(session, source_name, target_name, args.dry_run)

        # 3. Rename lowercase skills
        await rename_skills(session, args.dry_run)

        # 4. Purge junk skills
        await purge_junk_skills(session, args.dry_run)

        if not args.dry_run:
            print("Committing changes...")
            await session.commit()
            print("Successfully committed!")
        else:
            await session.rollback()
            print("Dry-run complete (rolled back).")

        # Report final status
        res_total = await session.execute(text("SELECT COUNT(*) FROM skills;"))
        res_std = await session.execute(text("SELECT standard_name, COUNT(*) FROM skill_standards GROUP BY standard_name;"))
        print(f"Final Count: Total Skills = {res_total.scalar()}")
        print(f"Standards: {res_std.all()}")


if __name__ == "__main__":
    asyncio.run(main())
