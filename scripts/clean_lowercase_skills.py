"""Clean, rename and normalize the 69 lowercase skills in Supabase.

1. Spanish concepts and corrupted slugs are re-pointed to canonical Lightcast entities,
   added to skill_aliases, and removed from skills table.
2. Legitimate technical skills currently in lowercase (e.g. wordpress -> WordPress,
   ansible -> Ansible) are renamed to proper Title Case.
"""

import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import delete, select, update
from src.ml_engine.infrastructure.models import (
    ClusterSkillModel,
    DiagnosticSkillModel,
    SkillAliasModel,
    SkillModel,
)
from src.scraper.infrastructure.models import OfferSkillModel
from src.shared.database import AsyncSessionLocal

# Spanish / corrupted terms to re-point to existing Lightcast canonical skills
MERGE_TO_CANONICAL = {
    "base de datos": "Database Design",
    "sistemas de gestión de bases de datos": "Database Management Systems",
    "ciencia de datos": "Data Science",
    "gestión de proyectos": "Software Project Management",
    "gestión de riesgos": "Risk Management",
    "agentes": "Artificial Intelligence",
    "angular18+": "Angular (Web Framework)",
    "arquitecturascloudnative": "Cloud Architecture",
    "iagenerativa": "Generative Artificial Intelligence",
    "testintegrados": "Integration Testing",
    "microsoftdynamics365customerservicecloud": "Microsoft Dynamics 365",
    "responsive": "Responsive Web Design",
}

# Standalone technologies to rename to Title Case
RENAME_TO_TITLE_CASE = {
    "ansible": "Ansible",
    "apache maven": "Apache Maven",
    "apache tomcat": "Apache Tomcat",
    "api-first": "API-First Design",
    "asp.net": "ASP.NET",
    "asterisk": "Asterisk",
    "auth0": "Auth0",
    "bantotal": "Bantotal",
    "bigquery": "Google BigQuery",
    "bull": "Bull",
    "bullmq": "BullMQ",
    "cad": "Computer-Aided Design (CAD)",
    "computer vision": "Computer Vision",
    "dashboards": "Dashboards",
    "data analysis": "Data Analysis",
    "data engineering": "Data Engineering",
    "datadog": "Datadog",
    "eks": "Amazon EKS",
    "elasticsearch": "Elasticsearch",
    "es6": "ECMAScript 6 (ES6)",
    "ethereum": "Ethereum",
    "financialservices": "Financial Services",
    "fintech": "Fintech",
    "gemini": "Google Gemini",
    "gradio": "Gradio",
    "graphicdesign": "Graphic Design",
    "grpc": "gRPC",
    "healthcare": "Healthcare IT",
    "hyper-v": "Microsoft Hyper-V",
    "iac": "Infrastructure as Code (IaC)",
    "idoc": "SAP IDoc",
    "json": "JSON",
    "kali linux": "Kali Linux",
    "kql": "Kusto Query Language (KQL)",
    "magento": "Magento",
    "mariadb": "MariaDB",
    "marketplace": "Marketplace Platforms",
    "mastra": "Mastra",
    "microservices": "Microservices",
    "mvvm": "Model-View-ViewModel (MVVM)",
    "n8n": "N8N",
    "nlp": "Natural Language Processing (NLP)",
    "opentelemetry": "OpenTelemetry",
    "oracle application development framework": "Oracle Application Development Framework (ADF)",
    "outlook": "Microsoft Outlook",
    "pmi": "Project Management Institute (PMI)",
    "security": "Information Security",
    "shopify": "Shopify",
    "system architecture": "System Architecture",
    "tensorflow": "TensorFlow",
    "togaf": "TOGAF",
    "transact-sql": "Transact-SQL",
    "vmware": "VMware",
    "webhooks": "Webhooks",
    "webpack": "Webpack",
    "wordpress": "WordPress",
    "xml": "XML",
}


from sqlalchemy.orm import defer


async def run_clean():
    async with AsyncSessionLocal() as session:
        print("=== Step 1: Loading all skills (deferring heavy embeddings) ===")
        res = await session.execute(select(SkillModel).options(defer(SkillModel.embedding)))
        all_skills = res.scalars().all()
        name_to_skill = {s.name.lower().strip(): s for s in all_skills}
        id_to_skill = {s.skill_id: s for s in all_skills}

        print(f"Total skills in DB: {len(all_skills)}")

        # Step 2: Process Merges / Aliases
        print("\n=== Step 2: Merging Spanish and corrupted skills into canonical skills ===")
        merged_count = 0
        for old_name_lower, target_canonical_name in MERGE_TO_CANONICAL.items():
            old_skill = name_to_skill.get(old_name_lower)
            if not old_skill:
                continue

            target_skill = name_to_skill.get(target_canonical_name.lower().strip())
            if not target_skill:
                # Search partial
                partials = [s for s in all_skills if target_canonical_name.lower() in s.name.lower()]
                if partials:
                    target_skill = partials[0]

            if not target_skill:
                print(f"[WARN] Target canonical skill '{target_canonical_name}' not found for '{old_name_lower}'. Skipping.")
                continue

            old_id = old_skill.skill_id
            target_id = target_skill.skill_id

            print(f"Merging '{old_skill.name}' ({old_id}) -> '{target_skill.name}' ({target_id})")

            # 1. Add old_name as alias in skill_aliases if not already present
            existing_alias = await session.execute(
                select(SkillAliasModel).where(
                    SkillAliasModel.alias_name.ilike(old_skill.name)
                )
            )
            existing_obj = existing_alias.scalars().first()
            if not existing_obj:
                session.add(SkillAliasModel(skill_id=target_id, alias_name=old_skill.name))
                print(f"  + Added alias '{old_skill.name}' -> '{target_skill.name}'")
            else:
                existing_obj.skill_id = target_id
                print(f"  ~ Updated existing alias '{old_skill.name}' to point to '{target_skill.name}'")

            # 2. Re-point offer_skills
            # Check existing to avoid unique constraint violations
            existing_os = await session.execute(
                select(OfferSkillModel.job_offer_id).where(OfferSkillModel.skill_id == target_id)
            )
            existing_job_ids = set(existing_os.scalars().all())

            # Delete duplicates where offer already has target_id
            await session.execute(
                delete(OfferSkillModel).where(
                    OfferSkillModel.skill_id == old_id,
                    OfferSkillModel.job_offer_id.in_(existing_job_ids),
                )
            )
            # Re-point remaining
            await session.execute(
                update(OfferSkillModel)
                .where(OfferSkillModel.skill_id == old_id)
                .values(skill_id=target_id)
            )

            # 3. Re-point cluster_skills
            existing_cs = await session.execute(
                select(ClusterSkillModel.cluster_id).where(ClusterSkillModel.skill_id == target_id)
            )
            existing_cluster_ids = set(existing_cs.scalars().all())
            await session.execute(
                delete(ClusterSkillModel).where(
                    ClusterSkillModel.skill_id == old_id,
                    ClusterSkillModel.cluster_id.in_(existing_cluster_ids),
                )
            )
            await session.execute(
                update(ClusterSkillModel)
                .where(ClusterSkillModel.skill_id == old_id)
                .values(skill_id=target_id)
            )

            # 4. Re-point diagnostic_skills
            existing_ds = await session.execute(
                select(DiagnosticSkillModel.diagnostic_id).where(DiagnosticSkillModel.skill_id == target_id)
            )
            existing_diag_ids = set(existing_ds.scalars().all())
            await session.execute(
                delete(DiagnosticSkillModel).where(
                    DiagnosticSkillModel.skill_id == old_id,
                    DiagnosticSkillModel.diagnostic_id.in_(existing_diag_ids),
                )
            )
            await session.execute(
                update(DiagnosticSkillModel)
                .where(DiagnosticSkillModel.skill_id == old_id)
                .values(skill_id=target_id)
            )

            # 5. Delete aliases pointing to old skill
            await session.execute(delete(SkillAliasModel).where(SkillAliasModel.skill_id == old_id))

            # 6. Delete the old skill
            await session.delete(old_skill)
            merged_count += 1

        await session.commit()
        print(f"Successfully merged and removed {merged_count} skills.")

        # Step 3: Rename standalone technical skills to Title Case
        print("\n=== Step 3: Renaming standalone technical skills to Title Case ===")
        renamed_count = 0
        for old_name_lower, new_title in RENAME_TO_TITLE_CASE.items():
            skill = name_to_skill.get(old_name_lower)
            if skill and skill.name != new_title:
                old_name = skill.name
                skill.name = new_title
                # Also add lowercase version as alias if not already in aliases
                existing_alias = await session.execute(
                    select(SkillAliasModel).where(
                        SkillAliasModel.alias_name.ilike(old_name)
                    )
                )
                if not existing_alias.scalars().first():
                    session.add(SkillAliasModel(skill_id=skill.skill_id, alias_name=old_name))
                print(f"Renamed: '{old_name}' -> '{new_title}'")
                renamed_count += 1

        await session.commit()
        print(f"Successfully renamed {renamed_count} skills.")

        # Step 4: Verification check
        res_after = await session.execute(select(SkillModel.name))
        after_names = res_after.scalars().all()
        remaining_lowers = [
            n for n in after_names if n == n.lower() and any(c.isalpha() for c in n)
        ]
        print(f"\nVerification: Remaining lowercase skills in DB: {len(remaining_lowers)}")
        if remaining_lowers:
            print(f"Remaining: {remaining_lowers}")
        else:
            print("SUCCESS! Exactly 0 lowercase skills remain in the database!")


if __name__ == "__main__":
    asyncio.run(run_clean())
