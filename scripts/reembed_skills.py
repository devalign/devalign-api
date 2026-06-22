"""Script to re-embed all skills and profile CVs in the database using the currently active embedding provider.

Use this script to migrate the database vectors if you switch between Voyage AI and OpenAI.
"""

import asyncio
import os
import sys

# Append src/ to path so we can import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select

from src.ml_engine.infrastructure.embeddings import get_embedding_service
from src.ml_engine.infrastructure.models import ProfileModel, SkillModel
from src.shared.database import AsyncSessionLocal


async def reembed_all() -> None:
    session = AsyncSessionLocal()
    embedding_service = get_embedding_service()

    print(f"Starting DB Re-embedding using: {embedding_service.__class__.__name__}")

    try:
        # 1. Re-embed Skills
        print("\n--- Processing Skills ---")
        skills_res = await session.execute(select(SkillModel))
        skills = skills_res.scalars().all()
        print(f"Found {len(skills)} skills in database.")

        if skills:
            batch_size = 50
            for i in range(0, len(skills), batch_size):
                # Simple rate limit delay for free tier Voyage API (3 RPM limit)
                if i > 0 and "Voyage" in embedding_service.__class__.__name__:
                    print("Waiting 21 seconds to respect Voyage API free tier rate limits...")
                    await asyncio.sleep(21)

                batch = skills[i : i + batch_size]
                names = [s.name for s in batch]
                print(f"Embedding skill batch {i//batch_size + 1}: {names}")

                try:
                    vectors = await embedding_service.embed_batch(names)
                    for skill, vector in zip(batch, vectors, strict=True):
                        skill.embedding = vector
                    await session.commit()
                except Exception as e:
                    print(f"Error embedding skill batch: {e}")
                    await session.rollback()
                    raise e

        # 2. Re-embed Profiles (cv_raw_text)
        print("\n--- Processing Profile CVs ---")
        profiles_res = await session.execute(select(ProfileModel).where(ProfileModel.cv_raw_text != None))
        profiles = profiles_res.scalars().all()
        print(f"Found {len(profiles)} profiles with raw CV text in database.")

        if profiles:
            # Re-embedding profiles one by one or in small batches because CV texts are large
            for idx, profile in enumerate(profiles):
                # Rate limit spacing for Voyage
                if idx > 0 and "Voyage" in embedding_service.__class__.__name__:
                    print("Waiting 10 seconds between profile embeddings...")
                    await asyncio.sleep(10)

                print(f"Embedding CV for profile: {profile.full_name or profile.profile_id}")
                try:
                    vector = await embedding_service.embed_text(profile.cv_raw_text)
                    profile.cv_embedding = vector
                    await session.commit()
                except Exception as e:
                    print(f"Error embedding profile {profile.profile_id}: {e}")
                    await session.rollback()
                    raise e

        print("\nDatabase re-embedding completed successfully!")
        print("IMPORTANT: Centroids for existing clusters are now out of sync. Please run the")
        print("clustering pipeline in the 'devalign-ml' repository to retrain clusters and")
        print("calculate correct centroids in the new vector space.")

    except Exception as e:
        print(f"\nMigration failed: {e}")
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(reembed_all())
