"""Seed script: Parse ESCO digital skills in Spanish and English and seed the database.

Usage:
    python scripts/seed_esco.py
"""

import asyncio
import os
import sys
import pandas as pd
import structlog
from sqlalchemy.future import select

# Setup path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.ml_engine.infrastructure.models import SkillModel, SkillAliasModel, SkillRelationModel
from src.shared.database import AsyncSessionLocal
from src.ml_engine.infrastructure.embeddings import VoyageEmbeddingService

logger = structlog.get_logger()

# Paths to the CSV files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ES_CSV_PATH = os.path.join(
    BASE_DIR,
    "ESCO dataset - v1.2.1 - classification - es - csv",
    "digitalSkillsCollection_es.csv"
)
EN_CSV_PATH = os.path.join(
    BASE_DIR,
    "ESCO dataset - v1.2.1 - classification - en - csv",
    "digitalSkillsCollection_en.csv"
)
REL_SKILL_SKILL_PATH = os.path.join(
    BASE_DIR,
    "ESCO dataset - v1.2.1 - classification - en - csv",
    "skillSkillRelations_en.csv"
)
REL_BROADER_PATH = os.path.join(
    BASE_DIR,
    "ESCO dataset - v1.2.1 - classification - en - csv",
    "broaderRelationsSkillPillar_en.csv"
)


async def seed_esco():
    logger.info("Starting ESCO digital skills seeding...")

    # 1. Load CSV files
    if not os.path.exists(ES_CSV_PATH) or not os.path.exists(EN_CSV_PATH):
        logger.error(
            "ESCO dataset files not found. Ensure folders are extracted in C:\\Projects\\Devalign"
        )
        return

    logger.info("Reading ESCO CSVs...")
    df_es = pd.read_csv(ES_CSV_PATH)
    df_en = pd.read_csv(EN_CSV_PATH)

    # Clean columns
    df_es = df_es.dropna(subset=["conceptUri", "preferredLabel"])
    df_en = df_en.dropna(subset=["conceptUri", "preferredLabel"])

    # Merge on conceptUri to align Spanish and English labels
    logger.info("Merging ES and EN datasets...")
    merged_df = pd.merge(
        df_es[["conceptUri", "preferredLabel", "altLabels", "description", "skillType"]],
        df_en[["conceptUri", "preferredLabel", "altLabels"]],
        on="conceptUri",
        suffixes=("_es", "_en")
    )
    logger.info(f"Aligned {len(merged_df)} digital skills.")

    # Initialize Voyage embedding service
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
        # Load all existing skills to avoid unique name violations and handle updates
        logger.info("Querying existing skills from DB...")
        existing_skills_query = await session.execute(select(SkillModel))
        existing_skills = existing_skills_query.scalars().all()

        existing_by_name = {s.name: s for s in existing_skills}
        existing_by_uri = {s.esco_uri: s for s in existing_skills if s.esco_uri}

        skills_to_update = []
        new_skills_to_insert = []
        aliases_to_insert = []
        skills_lacking_embeddings = []

        # Iterate and separate into new inserts vs updates
        for _, row in merged_df.iterrows():
            uri = row["conceptUri"]
            name_es = row["preferredLabel_es"].strip().lower()
            name_en = row["preferredLabel_en"].strip().lower()

            # Determine nature
            skill_type = str(row["skillType"]).lower()
            nature = "tech" if "skill" in skill_type or "competence" in skill_type else "concept"

            if uri in existing_by_uri:
                existing_skill = existing_by_uri[uri]
                if existing_skill.embedding is None:
                    skills_lacking_embeddings.append((existing_skill, name_es))
                continue

            if name_es in existing_by_name:
                # Skill exists in DB by name but has no esco_uri. Update it!
                existing_skill = existing_by_name[name_es]
                existing_skill.esco_uri = uri
                existing_skill.nature = nature
                skills_to_update.append((existing_skill, name_es, name_en, row["altLabels_es"], row["altLabels_en"]))
                if existing_skill.embedding is None:
                    skills_lacking_embeddings.append((existing_skill, name_es))
            else:
                # New skill entirely
                skill = SkillModel(
                    name=name_es,
                    esco_uri=uri,
                    nature=nature,
                    weight=1.00
                )
                new_skills_to_insert.append((skill, name_es, name_en, row["altLabels_es"], row["altLabels_en"]))

        # A. Process Updates (Merge ESCO URI to existing skills)
        if skills_to_update:
            logger.info(f"Merging ESCO URIs into {len(skills_to_update)} existing skills...")
            for skill, name_es, name_en, alts_es, alts_en in skills_to_update:
                session.add(skill)

                # English label as alias
                if name_es != name_en:
                    aliases_to_insert.append(SkillAliasModel(alias_name=name_en, skill_id=skill.skill_id))

                # Parse Spanish altLabels
                if pd.notna(alts_es):
                    for alt in str(alts_es).split("\n"):
                        alt_clean = alt.strip().lower()
                        if alt_clean and alt_clean != name_es:
                            aliases_to_insert.append(SkillAliasModel(alias_name=alt_clean, skill_id=skill.skill_id))

                # Parse English altLabels
                if pd.notna(alts_en):
                    for alt in str(alts_en).split("\n"):
                        alt_clean = alt.strip().lower()
                        if alt_clean and alt_clean != name_en:
                            aliases_to_insert.append(SkillAliasModel(alias_name=alt_clean, skill_id=skill.skill_id))

            await session.commit()

        # B. Backfill missing embeddings for existing/updated skills (rate-limited)
        if skills_lacking_embeddings and embedding_service:
            logger.info(f"Found {len(skills_lacking_embeddings)} skills lacking embeddings. Backfilling...")
            batch_size = 50
            for i in range(0, len(skills_lacking_embeddings), batch_size):
                batch = skills_lacking_embeddings[i:i + batch_size]
                texts_to_embed = [item[1] for item in batch]
                try:
                    logger.info(f"Backfilling embeddings batch {i // batch_size + 1}/{len(skills_lacking_embeddings)//batch_size + 1}...")
                    embeddings = await embedding_service.embed_batch(texts_to_embed)
                    for idx, emb in enumerate(embeddings):
                        batch[idx][0].embedding = emb
                    
                    # Commit this batch
                    for item in batch:
                        session.add(item[0])
                    await session.commit()

                    # Sleep to respect rate limits
                    logger.info("Sleeping 22 seconds to respect Voyage API rate limits...")
                    await asyncio.sleep(22)
                except Exception as e:
                    logger.error(f"Error backfilling embeddings: {e}")
                    await asyncio.sleep(10)
                    break

        # C. Process New Skill Inserts in batches (generate embeddings)
        batch_size = 50
        total_seeded = 0

        logger.info(f"Need to insert {len(new_skills_to_insert)} new skills.")

        for i in range(0, len(new_skills_to_insert), batch_size):
            batch = new_skills_to_insert[i:i + batch_size]
            skills_batch = [item[0] for item in batch]

            # Generate embeddings
            if embedding_service:
                texts_to_embed = [item[1] for item in batch]  # Embed Spanish names
                try:
                    logger.info(f"Generating embeddings for batch {i // batch_size + 1}/{len(new_skills_to_insert)//batch_size + 1}...")
                    embeddings = await embedding_service.embed_batch(texts_to_embed)
                    for idx, emb in enumerate(embeddings):
                        skills_batch[idx].embedding = emb
                    
                    # Sleep to respect Voyage API free tier rate limit (3 RPM)
                    logger.info("Sleeping 22 seconds to respect Voyage API free tier rate limit...")
                    await asyncio.sleep(22)
                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}. Seeding without embeddings.")
                    # Sleep anyway to prevent immediate failure in next batch
                    await asyncio.sleep(10)

            session.add_all(skills_batch)
            await session.commit()  # Commit to generate skill_ids

            # Now add aliases
            for skill, name_es, name_en, alts_es, alts_en in batch:
                # Add English preferred label as alias
                if name_es != name_en:
                    aliases_to_insert.append(
                        SkillAliasModel(alias_name=name_en, skill_id=skill.skill_id)
                    )

                # Parse Spanish altLabels
                if pd.notna(alts_es):
                    for alt in str(alts_es).split("\n"):
                        alt_clean = alt.strip().lower()
                        if alt_clean and alt_clean != name_es:
                            aliases_to_insert.append(
                                SkillAliasModel(alias_name=alt_clean, skill_id=skill.skill_id)
                            )

                # Parse English altLabels
                if pd.notna(alts_en):
                    for alt in str(alts_en).split("\n"):
                        alt_clean = alt.strip().lower()
                        if alt_clean and alt_clean != name_en:
                            aliases_to_insert.append(
                                SkillAliasModel(alias_name=alt_clean, skill_id=skill.skill_id)
                            )

            total_seeded += len(skills_batch)
            logger.info(f"Seeded {total_seeded}/{len(new_skills_to_insert)} skills.")

        # Batch insert aliases, avoiding duplicates
        if aliases_to_insert:
            logger.info(f"Inserting {len(aliases_to_insert)} aliases...")
            # Deduplicate aliases list in memory first
            unique_aliases = {}
            for alias in aliases_to_insert:
                unique_aliases[alias.alias_name] = alias

            # Query existing aliases to avoid conflict
            existing_aliases_query = await session.execute(select(SkillAliasModel.alias_name))
            existing_alias_names = set(existing_aliases_query.scalars().all())

            filtered_aliases = [
                v for k, v in unique_aliases.items() if k not in existing_alias_names
            ]
            session.add_all(filtered_aliases)
            await session.commit()
            logger.info(f"Seeded {len(filtered_aliases)} unique aliases.")

        # 3. Seed Relations
        logger.info("Seeding relations (Graph)...")
        # Load all seeded ESCO skills to map URI -> skill_id
        esco_skills_query = await session.execute(
            select(SkillModel.esco_uri, SkillModel.skill_id).where(SkillModel.esco_uri.isnot(None))
        )
        uri_to_id = {row[0]: row[1] for row in esco_skills_query.all()}

        # 3a. Broader relations (belongs_to)
        if os.path.exists(REL_BROADER_PATH):
            logger.info("Reading broader relations...")
            df_broad = pd.read_csv(REL_BROADER_PATH)
            relations_to_add = []

            # Find existing relations to avoid duplicate keys
            existing_rels_query = await session.execute(
                select(SkillRelationModel.source_skill_id, SkillRelationModel.target_skill_id)
            )
            existing_rels = {(row[0], row[1]) for row in existing_rels_query.all()}

            for _, row in df_broad.iterrows():
                source_uri = row["conceptUri"]
                target_uri = row["broaderUri"]

                if source_uri in uri_to_id and target_uri in uri_to_id:
                    source_id = uri_to_id[source_uri]
                    target_id = uri_to_id[target_uri]

                    if (source_id, target_id) not in existing_rels:
                        relations_to_add.append(
                            SkillRelationModel(
                                source_skill_id=source_id,
                                target_skill_id=target_id,
                                relation_type="belongs_to"
                            )
                        )
                        existing_rels.add((source_id, target_id))

            if relations_to_add:
                session.add_all(relations_to_add)
                await session.commit()
                logger.info(f"Seeded {len(relations_to_add)} hierarchical (belongs_to) relations.")

        # 3b. Skill-Skill relations (requires/optional)
        if os.path.exists(REL_SKILL_SKILL_PATH):
            logger.info("Reading skill-skill relations...")
            df_ss = pd.read_csv(REL_SKILL_SKILL_PATH)
            relations_to_add = []

            # Refresh existing relations set
            existing_rels_query = await session.execute(
                select(SkillRelationModel.source_skill_id, SkillRelationModel.target_skill_id)
            )
            existing_rels = {(row[0], row[1]) for row in existing_rels_query.all()}

            for _, row in df_ss.iterrows():
                source_uri = row["originalSkillUri"]
                target_uri = row["relatedSkillUri"]
                rel_type_raw = str(row["relationType"]).lower()
                rel_type = "requires" if "essential" in rel_type_raw or "requires" in rel_type_raw else "optional"

                if source_uri in uri_to_id and target_uri in uri_to_id:
                    source_id = uri_to_id[source_uri]
                    target_id = uri_to_id[target_uri]

                    if (source_id, target_id) not in existing_rels:
                        relations_to_add.append(
                            SkillRelationModel(
                                source_skill_id=source_id,
                                target_skill_id=target_id,
                                relation_type=rel_type
                            )
                        )
                        existing_rels.add((source_id, target_id))

            if relations_to_add:
                session.add_all(relations_to_add)
                await session.commit()
                logger.info(f"Seeded {len(relations_to_add)} cross (requires/optional) relations.")

    logger.info("ESCO Seeding completed successfully!")


if __name__ == "__main__":
    asyncio.run(seed_esco())
