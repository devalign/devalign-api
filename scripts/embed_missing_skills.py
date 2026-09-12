import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import func, select

from src.ml_engine.infrastructure.embeddings import get_embedding_service
from src.ml_engine.infrastructure.models import SkillModel
from src.scraper.infrastructure.models import OfferSkillModel
from src.shared.database import AsyncSessionLocal


async def embed_missing():
    session = AsyncSessionLocal()
    embedding_service = get_embedding_service()

    print(f"Starting DB Missing Skills Embedding using: {embedding_service.__class__.__name__}")

    try:
        print("\n--- Querying Skills with Missing Embeddings (Prioritizing Most Demanded) ---")
        stmt = (
            select(SkillModel)
            .where(SkillModel.embedding.is_(None))
            .outerjoin(OfferSkillModel, OfferSkillModel.skill_id == SkillModel.skill_id)
            .group_by(SkillModel.skill_id)
            .order_by(func.count(OfferSkillModel.offer_skill_id).desc(), SkillModel.name.asc())
        )
        skills_res = await session.execute(stmt)
        skills = skills_res.scalars().all()
        total_skills = len(skills)
        print(f"Found {total_skills} skills with missing embeddings in database.")

        if skills:
            batch_size = 100
            total_batches = (total_skills + batch_size - 1) // batch_size
            for idx, i in enumerate(range(0, total_skills, batch_size)):
                batch_num = idx + 1
                # Rate limit delay for Voyage free tier (3 RPM = ~20s between requests)
                if idx > 0 and "Voyage" in embedding_service.__class__.__name__:
                    print(f"Waiting 21s to respect Voyage API rate limits (Batch {batch_num}/{total_batches})...")
                    await asyncio.sleep(21)

                batch = skills[i : i + batch_size]
                names = [s.name for s in batch]
                print(f"[{batch_num}/{total_batches}] Embedding {len(batch)} skills (First 3: {names[:3]})...")

                try:
                    vectors = await embedding_service.embed_batch(names)
                    for skill, vector in zip(batch, vectors, strict=True):
                        skill.embedding = vector
                    await session.commit()
                    processed = min(i + batch_size, total_skills)
                    print(f"-> Committed batch {batch_num}/{total_batches}. Total processed: {processed}/{total_skills} ({processed*100/total_skills:.1f}%).")
                except Exception as e:
                    print(f"Error embedding skill batch {batch_num}: {e}")
                    await session.rollback()
                    raise e
            print(f"\nAll {total_skills} skills successfully embedded and committed to database!")
        else:
            print("No skills with missing embeddings found. Database is 100% complete!")

    except Exception as e:
        print(f"\nEmbedding failed: {e}")
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(embed_missing())
