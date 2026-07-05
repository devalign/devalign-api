"""Seed script: Parse ESCO IT skills based on IT Occupations (ISCO 25 & 35) in Spanish and English and seed the database.

Usage:
    python scripts/seed_standards.py
"""

import asyncio
import os
import sys
import pandas as pd
import structlog
from uuid import uuid4
from sqlalchemy.future import select

# Setup path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.ml_engine.infrastructure.models import SkillModel, SkillAliasModel, SkillStandardModel
from src.shared.database import AsyncSessionLocal
from src.ml_engine.infrastructure.embeddings import VoyageEmbeddingService

logger = structlog.get_logger()

# Paths to the CSV files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ES_CSV_PATH = os.path.join(BASE_DIR, "ESCO dataset - v1.2.1 - classification - es - csv", "skills_es.csv")
EN_CSV_PATH = os.path.join(BASE_DIR, "ESCO dataset - v1.2.1 - classification - en - csv", "skills_en.csv")
HIERARCHY_ES_PATH = os.path.join(BASE_DIR, "ESCO dataset - v1.2.1 - classification - es - csv", "skillsHierarchy_es.csv")
BROADER_ES_PATH = os.path.join(BASE_DIR, "ESCO dataset - v1.2.1 - classification - es - csv", "broaderRelationsSkillPillar_es.csv")
OCCUPATIONS_CSV_PATH = os.path.join(BASE_DIR, "ESCO dataset - v1.2.1 - classification - es - csv", "occupations_es.csv")
RELATIONS_CSV_PATH = os.path.join(BASE_DIR, "ESCO dataset - v1.2.1 - classification - es - csv", "occupationSkillRelations_es.csv")


async def seed_standards():
    logger.info("Starting ESCO IT skills seeding based on TI Occupations...")

    # 1. Verify files exist
    paths = [ES_CSV_PATH, EN_CSV_PATH, HIERARCHY_ES_PATH, BROADER_ES_PATH, OCCUPATIONS_CSV_PATH, RELATIONS_CSV_PATH]
    for p in paths:
        if not os.path.exists(p):
            logger.error(f"Required CSV file not found: {p}. Ensure ESCO datasets are extracted in C:\\Projects\\Devalign")
            return

    # 2. Filter ESCO occupations for IT roles (ISCO 25 & 35)
    logger.info("Filtering ESCO occupations for IT roles (ISCO 25 and 35)...")
    df_occ = pd.read_csv(OCCUPATIONS_CSV_PATH)
    df_occ['iscoGroup_str'] = df_occ['iscoGroup'].astype(str)
    it_occs = df_occ[df_occ['iscoGroup_str'].str.startswith(('25', '35'))]['conceptUri'].unique()
    it_occs_set = set(it_occs)
    logger.info(f"Found {len(it_occs_set)} IT occupations.")

    # 3. Filter ESCO skill relations for IT occupations
    logger.info("Filtering ESCO skill relations for IT occupations...")
    df_rel = pd.read_csv(RELATIONS_CSV_PATH)
    it_skills_uris = df_rel[df_rel['occupationUri'].isin(it_occs_set)]['skillUri'].unique()
    it_skills_set = set(it_skills_uris)
    logger.info(f"Found {len(it_skills_set)} unique skills mapped to IT occupations.")

    # 4. Parse hierarchy to resolve group codes
    logger.info("Parsing ESCO hierarchies to resolve group codes...")
    df_hier = pd.read_csv(HIERARCHY_ES_PATH)
    all_groups = {}  # URI -> code
    for _, row in df_hier.iterrows():
        for level in range(4):
            uri_col = f"Level {level} URI"
            code_col = f"Level {level} code"
            if uri_col in df_hier.columns and code_col in df_hier.columns:
                uri = row[uri_col]
                code = row[code_col]
                if pd.notna(uri) and pd.notna(code):
                    all_groups[uri] = str(code)

    # 5. Parse broader relations to map leaf skills to parent groups
    logger.info("Parsing broader relations to map leaf skills to parent groups...")
    df_broad = pd.read_csv(BROADER_ES_PATH)
    leaf_skills = {}  # conceptUri -> parent_code
    for _, row in df_broad.iterrows():
        source_uri = row["conceptUri"]
        target_uri = row["broaderUri"]
        if target_uri in all_groups:
            leaf_skills[source_uri] = all_groups[target_uri]

    # 6. Load full Spanish and English skill details, dropping duplicates
    logger.info("Reading skill details (ES & EN)...")
    df_es = pd.read_csv(ES_CSV_PATH).dropna(subset=["conceptUri", "preferredLabel"]).drop_duplicates(subset=["conceptUri"])
    df_en = pd.read_csv(EN_CSV_PATH).dropna(subset=["conceptUri", "preferredLabel"]).drop_duplicates(subset=["conceptUri"])

    merged_df = pd.merge(
        df_es[["conceptUri", "preferredLabel", "altLabels", "description", "skillType"]],
        df_en[["conceptUri", "preferredLabel", "altLabels"]],
        on="conceptUri",
        suffixes=("_es", "_en")
    )
    
    # Filter to only keep skills mapped to IT occupations
    it_details = merged_df[merged_df["conceptUri"].isin(it_skills_set)]
    logger.info(f"Loaded details for {len(it_details)} aligned IT skills.")

    # 7. Initialize Voyage Embedding Service if configured
    embedding_service = None
    if settings.VOYAGE_API_KEY:
        logger.info("Initializing Voyage Embedding Service...")
        embedding_service = VoyageEmbeddingService(
            api_key=settings.VOYAGE_API_KEY,
            model=settings.EMBEDDING_MODEL
        )
    else:
        logger.warning("VOYAGE_API_KEY not configured. Embeddings will not be generated.")

    async with AsyncSessionLocal() as session:
        # A. Clean up old ESCO skills first
        logger.info("Cleaning up existing ESCO standard mappings and skills...")
        existing_esco_standards = await session.execute(
            select(SkillStandardModel.skill_id, SkillStandardModel.id).where(SkillStandardModel.standard_name == "ESCO")
        )
        esco_mappings = existing_esco_standards.all()
        if esco_mappings:
            esco_skill_ids = [m[0] for m in esco_mappings]
            esco_standard_ids = [m[1] for m in esco_mappings]
            
            # Delete standard mappings
            for sid in esco_standard_ids:
                standard_obj = await session.get(SkillStandardModel, sid)
                if standard_obj:
                    await session.delete(standard_obj)
            
            # Delete skills (which cascades to aliases and relations)
            for skid in esco_skill_ids:
                skill_obj = await session.get(SkillModel, skid)
                if skill_obj:
                    await session.delete(skill_obj)
            
            await session.commit()
            logger.info(f"Cleaned up {len(esco_skill_ids)} old ESCO skills and their mappings.")

        # Load remaining skills and standards to avoid duplicates
        logger.info("Querying existing database state...")
        existing_skills_query = await session.execute(select(SkillModel))
        existing_skills = existing_skills_query.scalars().all()
        existing_by_name = {s.name: s for s in existing_skills}

        existing_standards_query = await session.execute(
            select(SkillStandardModel).where(SkillStandardModel.standard_name == "ESCO")
        )
        existing_standards = {s.standard_uri: s.skill_id for s in existing_standards_query.scalars().all()}

        new_skills_to_insert = []
        standards_to_insert = []
        aliases_to_insert = []
        skills_to_update_standard = []
        seen_names_to_insert = {}  # name_es -> skill_id

        for _, row in it_details.iterrows():
            uri = row["conceptUri"]
            name_es = row["preferredLabel_es"].strip().lower()
            name_en = row["preferredLabel_en"].strip().lower()
            
            # Resolve the standard code
            code = all_groups.get(uri) or leaf_skills.get(uri) or "IT"

            # If standard already exists, skip
            if uri in existing_standards:
                continue

            # If skill name exists in DB (e.g. tech skill or manually added concept) but lacks ESCO standard mapping
            if name_es in existing_by_name:
                existing_skill = existing_by_name[name_es]
                # Map standard to this existing skill
                standards_to_insert.append(
                    SkillStandardModel(
                        id=uuid4(),
                        skill_id=existing_skill.skill_id,
                        standard_name="ESCO",
                        standard_uri=uri,
                        standard_code=code
                    )
                )
                skills_to_update_standard.append((existing_skill, name_es, name_en, row["altLabels_es"], row["altLabels_en"]))
            elif name_es in seen_names_to_insert:
                # Skill name is duplicate, map standard URI to already planned new skill
                planned_skill_id = seen_names_to_insert[name_es]
                standards_to_insert.append(
                    SkillStandardModel(
                        id=uuid4(),
                        skill_id=planned_skill_id,
                        standard_name="ESCO",
                        standard_uri=uri,
                        standard_code=code
                    )
                )
            else:
                # Create a new skill completely
                skill_id = uuid4()
                skill = SkillModel(
                    skill_id=skill_id,
                    name=name_es,
                    nature="concept",
                    weight=1.00
                )
                new_skills_to_insert.append((skill, uri, code, name_es, name_en, row["altLabels_es"], row["altLabels_en"]))
                seen_names_to_insert[name_es] = skill_id

        # B. Process Existing Skills linking to ESCO
        if standards_to_insert:
            logger.info(f"Adding ESCO standards to {len(standards_to_insert)} existing or duplicate-name skills...")
            session.add_all(standards_to_insert)
            for skill_info in skills_to_update_standard:
                skill, name_es, name_en, alts_es, alts_en = skill_info
                # Add aliases
                if name_es != name_en:
                    aliases_to_insert.append(SkillAliasModel(alias_name=name_en, skill_id=skill.skill_id))
                if pd.notna(alts_es):
                    for alt in str(alts_es).split("\n"):
                        alt_clean = alt.strip().lower()
                        if alt_clean and alt_clean != name_es:
                            aliases_to_insert.append(SkillAliasModel(alias_name=alt_clean, skill_id=skill.skill_id))
                if pd.notna(alts_en):
                    for alt in str(alts_en).split("\n"):
                        alt_clean = alt.strip().lower()
                        if alt_clean and alt_clean != name_en:
                            aliases_to_insert.append(SkillAliasModel(alias_name=alt_clean, skill_id=skill.skill_id))
            await session.commit()

        # C. Process New Skill Inserts in batches (and generate embeddings if possible)
        batch_size = 50
        total_seeded = 0
        logger.info(f"Inserting {len(new_skills_to_insert)} new IT skills...")

        for i in range(0, len(new_skills_to_insert), batch_size):
            batch = new_skills_to_insert[i:i + batch_size]
            skills_batch = [item[0] for item in batch]

            # Generate embeddings in batch
            if embedding_service:
                texts_to_embed = [item[3] for item in batch]  # Spanish labels
                try:
                    logger.info(f"Generating embeddings for batch {i // batch_size + 1}...")
                    embeddings = await embedding_service.embed_batch(texts_to_embed)
                    for idx, emb in enumerate(embeddings):
                        skills_batch[idx].embedding = emb
                    
                    logger.info("Sleeping 22 seconds to respect Voyage API free tier rate limit...")
                    await asyncio.sleep(22)
                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}. Seeding batch without embeddings.")
                    await asyncio.sleep(10)

            session.add_all(skills_batch)
            await session.commit()  # Generates skill_ids

            # Build Standard Mappings and Aliases
            for skill, uri, code, name_es, name_en, alts_es, alts_en in batch:
                session.add(
                    SkillStandardModel(
                        id=uuid4(),
                        skill_id=skill.skill_id,
                        standard_name="ESCO",
                        standard_uri=uri,
                        standard_code=code
                    )
                )

                if name_es != name_en:
                    aliases_to_insert.append(SkillAliasModel(alias_name=name_en, skill_id=skill.skill_id))
                if pd.notna(alts_es):
                    for alt in str(alts_es).split("\n"):
                        alt_clean = alt.strip().lower()
                        if alt_clean and alt_clean != name_es:
                            aliases_to_insert.append(SkillAliasModel(alias_name=alt_clean, skill_id=skill.skill_id))
                if pd.notna(alts_en):
                    for alt in str(alts_en).split("\n"):
                        alt_clean = alt.strip().lower()
                        if alt_clean and alt_clean != name_en:
                            aliases_to_insert.append(SkillAliasModel(alias_name=alt_clean, skill_id=skill.skill_id))

            await session.commit()
            total_seeded += len(skills_batch)
            logger.info(f"Seeded {total_seeded}/{len(new_skills_to_insert)} skills.")

        # Batch insert aliases, avoiding duplicates
        if aliases_to_insert:
            logger.info(f"Inserting {len(aliases_to_insert)} aliases...")
            unique_aliases = {alias.alias_name: alias for alias in aliases_to_insert}

            existing_aliases_query = await session.execute(select(SkillAliasModel.alias_name))
            existing_alias_names = set(existing_aliases_query.scalars().all())

            filtered_aliases = [v for k, v in unique_aliases.items() if k not in existing_alias_names]
            session.add_all(filtered_aliases)
            await session.commit()
            logger.info(f"Seeded {len(filtered_aliases)} unique aliases.")

    logger.info("ESCO Seeding completed successfully!")


if __name__ == "__main__":
    asyncio.run(seed_standards())
