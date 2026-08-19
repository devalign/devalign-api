import asyncio
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select, delete
from src.ml_engine.infrastructure.llm_client import get_llm_service
from src.ml_engine.infrastructure.models import SkillModel, SkillStandardModel, ClusterSkillModel, DiagnosticSkillModel, ProfileSkillModel
from src.scraper.infrastructure.models import OfferSkillModel
from src.shared.database import AsyncSessionLocal

async def main():
    session = AsyncSessionLocal()
    llm = get_llm_service()
    
    print("Fetching ESCO skills for safe cleaning...")
    try:
        # Get all skills that have an ESCO standard mapping
        stmt = (
            select(SkillModel)
            .join(SkillStandardModel, SkillModel.skill_id == SkillStandardModel.skill_id)
            .where(SkillStandardModel.standard_name == "ESCO")
        )
        result = await session.execute(stmt)
        esco_skills = result.scalars().all()
        print(f"Loaded {len(esco_skills)} ESCO skills from database.")
        
        # Get linked skill IDs (to prevent foreign key errors and protect active data)
        offer_skills_res = await session.execute(select(OfferSkillModel.skill_id))
        linked_offer_skills = set(offer_skills_res.scalars().all())
        cluster_skills_res = await session.execute(select(ClusterSkillModel.skill_id))
        linked_cluster_skills = set(cluster_skills_res.scalars().all())
        diag_skills_res = await session.execute(select(DiagnosticSkillModel.skill_id))
        linked_diag_skills = set(diag_skills_res.scalars().all())
        profile_skills_res = await session.execute(select(ProfileSkillModel.skill_id))
        linked_profile_skills = set(profile_skills_res.scalars().all())
        
        linked_ids = linked_offer_skills | linked_cluster_skills | linked_diag_skills | linked_profile_skills
        
        # Classify them in batches of 250 using the LLM (OpenAI)
        batch_size = 250
        deleted_count = 0
        deselected_count = 0
        
        for i in range(0, len(esco_skills), batch_size):
            batch = esco_skills[i : i + batch_size]
            names = [s.name for s in batch]
            print(f"Processing batch {i // batch_size + 1}/{len(esco_skills)//batch_size + 1}...")
            
            prompt = (
                "You are an expert technical recruiter and software architect.\n"
                "Classify the following list of skills as either 'keep' (relevant to Software Development, IT, DevOps, QA, Cloud, Data, or Digital Tech) or 'delete' (completely unrelated to IT, e.g. arts, music, manual labor, non-technical business management, acting, physical crafts, photography, aviation, non-digital things).\n"
                "Return ONLY a valid JSON object where keys are the skill names (exactly as provided) and values are either 'keep' or 'delete'.\n"
                "Do not include any explanation or markdown formatting outside the JSON.\n\n"
                f"Skills to classify: {json.dumps(names)}"
            )
            
            # Sleep between batches to avoid rate limits
            if i > 0:
                await asyncio.sleep(8)
                
            response_text = ""
            for attempt in range(4):
                try:
                    response_text = await llm.generate(prompt)
                    break
                except Exception as exc:
                    print(f"Rate limit or error hit, waiting {10 + 5*attempt}s... ({exc})")
                    await asyncio.sleep(10 + 5*attempt)
            
            if not response_text:
                print("Failed to get response, skipping batch.")
                continue
                
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                classifications = json.loads(response_text[start:end])
                
                for s in batch:
                    action = classifications.get(s.name, "delete")
                    is_linked = s.skill_id in linked_ids
                    
                    if is_linked or action == "keep":
                        # Keep the skill, but delete its ESCO mapping from skill_standards
                        stmt_del_std = (
                            delete(SkillStandardModel)
                            .where(SkillStandardModel.skill_id == s.skill_id)
                            .where(SkillStandardModel.standard_name == "ESCO")
                        )
                        await session.execute(stmt_del_std)
                        deselected_count += 1
                        print(f"  [KEEP / DESELECT] '{s.name}' (linked={is_linked}) - removed ESCO tag.")
                    else:
                        # Unlinked and irrelevant -> delete completely from skills table
                        await session.delete(s)
                        deleted_count += 1
                        print(f"  [DELETE] '{s.name}' - deleted skill completely.")
                        
                # Commit after each batch to avoid massive transaction blocks
                await session.commit()
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                await session.rollback()
                raise e
                
        print("\n=== CLEANUP COMPLETED ===")
        print(f"Total skills deleted from catalog: {deleted_count}")
        print(f"Total skills kept (but ESCO association removed): {deselected_count}")
        
    except Exception as e:
        print(f"Failed to clean skills: {e}")
    finally:
        await session.close()

if __name__ == "__main__":
    asyncio.run(main())
