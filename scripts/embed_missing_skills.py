import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select

from src.ml_engine.infrastructure.embeddings import get_embedding_service
from src.ml_engine.infrastructure.models import SkillModel
from src.shared.database import AsyncSessionLocal


async def embed_missing():
    session = AsyncSessionLocal()
    embedding_service = get_embedding_service()

    print(f"Starting DB Missing Skills Embedding using: {embedding_service.__class__.__name__}")

    try:
        print("\n--- Processing Skills with Missing Embeddings ---")
        skills_res = await session.execute(select(SkillModel).where(SkillModel.embedding.is_(None)))
        skills = skills_res.scalars().all()
        print(f"Found {len(skills)} skills with missing embeddings in database.")

        if skills:
            batch_size = 50
            for i in range(0, len(skills), batch_size):
                # Rate limit delay for Voyage free tier
                if i > 0 and "Voyage" in embedding_service.__class__.__name__:
                    print("Waiting 21 seconds to respect Voyage API rate limits...")
                    await asyncio.sleep(21)

                batch = skills[i : i + batch_size]
                names = [s.name for s in batch]
                print(f"Embedding skill batch {i // batch_size + 1}: {names}")

                try:
                    vectors = await embedding_service.embed_batch(names)
                    for skill, vector in zip(batch, vectors, strict=True):
                        skill.embedding = vector
                    await session.commit()
                    print(f"Successfully embedded and committed batch {i // batch_size + 1}.")
                except Exception as e:
                    print(f"Error embedding skill batch: {e}")
                    await session.rollback()
                    raise e
        else:
            print("No skills with missing embeddings found.")

    except Exception as e:
        print(f"\nEmbedding failed: {e}")
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(embed_missing())
