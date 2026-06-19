import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.ml_engine.application.skill_catalog_service import SkillCatalogService
from src.ml_engine.domain.entities import Skill
from src.ml_engine.infrastructure.llm_client import get_llm_service
from src.ml_engine.infrastructure.models import SkillModel, SkillAliasModel
from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository
from src.shared.database import AsyncSessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migrate_skills")

async def run_migration():
    logger.info("Starting Skill Graph Migration...")
    
    async with AsyncSessionLocal() as session:
        # 1. Fetch all existing models to update them
        logger.info("Fetching existing skills from database...")
        result = await session.execute(
            select(SkillModel).options(selectinload(SkillModel.aliases))
        )
        existing_models = result.scalars().all()
        
        raw_names = [m.name for m in existing_models]
        
        if not raw_names:
            logger.info("No skills found to migrate.")
            return
            
        logger.info(f"Found {len(raw_names)} distinct skills to migrate.")
        
        # 2. Setup services
        skill_repo = SQLSkillRepository(session)
        llm_service = get_llm_service()
        catalog_service = SkillCatalogService(skill_repo, llm_service)
        
        logger.info("Classifying skills with LLM in batches of 20...")
        batch_size = 20
        all_new_skills = []
        for i in range(0, len(raw_names), batch_size):
            batch = raw_names[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(raw_names)-1)//batch_size + 1}")
            try:
                classified_skills = await catalog_service._classify_with_llm(batch)
                all_new_skills.extend(classified_skills)
            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")

        # Now update the existing skills in the DB
        logger.info("Updating database records...")
        
        # Build a lookup from classified skills
        classified_lookup = {}
        for s in all_new_skills:
            classified_lookup[s.normalized_name] = s
            for alias in s.aliases:
                classified_lookup[alias.lower().replace(" ", "")] = s
        
        # Load all existing aliases from the DB to avoid duplicates
        global_seen_aliases = set()
        for model in existing_models:
            if getattr(model, 'aliases', None):
                for a in model.aliases:
                    global_seen_aliases.add(a.alias_name)
                    
        updates_count = 0
        seen_canonical_names = set()
        
        for model in existing_models:
            norm_name = model.name.lower().replace(" ", "").replace(".", "")
            
            # Find matching classification
            classification: Skill = classified_lookup.get(norm_name)
            
            if classification:
                # We do NOT update model.name to avoid unique constraint violations
                # The original name is kept, but it will be enriched with nature, domains and aliases.
                
                model.nature = classification.nature.value
                model.domain_tags = classification.domain_tags
                
                existing_aliases = {a.alias_name for a in model.aliases}
                
                for alias in classification.aliases:
                    if alias not in global_seen_aliases:
                        model.aliases.append(SkillAliasModel(alias_name=alias))
                        global_seen_aliases.add(alias)
                
                updates_count += 1
            else:
                # Fallback if LLM missed it
                model.nature = "tech"
                model.domain_tags = ["Unknown"]
                seen_canonical_names.add(model.name)
                
        await session.commit()
        logger.info(f"Migration complete. Updated {updates_count} skills with Graph attributes.")


if __name__ == "__main__":
    asyncio.run(run_migration())
