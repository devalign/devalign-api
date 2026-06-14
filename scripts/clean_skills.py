"""Script to clean up duplicate skills in the database using Voyage/OpenAI embeddings."""

import asyncio
import os
import sys

import numpy as np
from sqlalchemy import delete, select, update

# Append src/ to path so we can import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_engine.infrastructure.embeddings import get_embedding_service
from src.ml_engine.infrastructure.models import ClusterSkillModel, DiagnosticSkillModel, SkillModel
from src.scraper.infrastructure.models import OfferSkillModel
from src.shared.database import AsyncSessionLocal


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

async def clean_skills() -> None:
    session = AsyncSessionLocal()
    embedding_service = get_embedding_service()

    print("Fetching all skills from database...")
    result = await session.execute(select(SkillModel))
    skills = result.scalars().all()
    print(f"Found {len(skills)} total skills.")

    # 1. Generate embeddings for any skills that lack them
    skills_lacking_embeddings = [s for s in skills if s.embedding is None]
    if skills_lacking_embeddings:
        print(f"Generating embeddings for {len(skills_lacking_embeddings)} skills...")
        batch_size = 50
        for i in range(0, len(skills_lacking_embeddings), batch_size):
            if i > 0:
                print("Waiting 21 seconds to respect Voyage API free tier rate limit (3 RPM)...")
                await asyncio.sleep(21)
            batch = skills_lacking_embeddings[i : i + batch_size]
            names = [s.name for s in batch]
            print(f"Embedding batch {i//batch_size + 1}: {names}")
            try:
                vectors = await embedding_service.embed_batch(names)
                for skill, vector in zip(batch, vectors, strict=True):
                    skill.embedding = vector
                await session.commit()
            except Exception as e:
                print(f"Error embedding batch: {e}")
                await session.rollback()
                return

    # Refresh skills list
    result = await session.execute(select(SkillModel))
    skills = result.scalars().all()

    # 2. Find and merge duplicates
    print("Analyzing skills for duplicates (threshold similarity >= 0.92)...")
    merged_ids = set()

    for i in range(len(skills)):
        skill_a = skills[i]
        if skill_a.skill_id in merged_ids:
            continue

        for j in range(i + 1, len(skills)):
            skill_b = skills[j]
            if skill_b.skill_id in merged_ids:
                continue

            # Double check names are not exactly equal (already handled by DB constraints but just in case)
            if skill_a.name.lower() == skill_b.name.lower():
                similarity = 1.0
            elif skill_a.embedding is not None and skill_b.embedding is not None:
                similarity = _cosine_similarity(skill_a.embedding, skill_b.embedding)
            else:
                similarity = 0.0

            if similarity >= 0.92:
                print(f"Duplicate detected: '{skill_b.name}' (ID: {skill_b.skill_id}) matches '{skill_a.name}' (ID: {skill_a.skill_id}) with similarity {similarity:.3f}")

                # Merge skill_b into skill_a
                try:
                    # Merge in offer_skills
                    b_offers_res = await session.execute(
                        select(OfferSkillModel.job_offer_id).where(OfferSkillModel.skill_id == skill_b.skill_id)
                    )
                    b_offer_ids = b_offers_res.scalars().all()

                    for offer_id in b_offer_ids:
                        # Check if offer already has skill_a
                        exists_res = await session.execute(
                            select(OfferSkillModel).where(
                                OfferSkillModel.job_offer_id == offer_id,
                                OfferSkillModel.skill_id == skill_a.skill_id
                            )
                        )
                        exists = exists_res.scalars().first()
                        if exists:
                            # If duplicate link exists, delete the link to skill_b
                            await session.execute(
                                delete(OfferSkillModel).where(
                                    OfferSkillModel.job_offer_id == offer_id,
                                    OfferSkillModel.skill_id == skill_b.skill_id
                                )
                            )
                        else:
                            # Update link from B to A
                            await session.execute(
                                update(OfferSkillModel)
                                .where(OfferSkillModel.job_offer_id == offer_id, OfferSkillModel.skill_id == skill_b.skill_id)
                                .values(skill_id=skill_a.skill_id)
                            )

                    # Merge in cluster_skills
                    b_clusters_res = await session.execute(
                        select(ClusterSkillModel.cluster_id).where(ClusterSkillModel.skill_id == skill_b.skill_id)
                    )
                    b_cluster_ids = b_clusters_res.scalars().all()

                    for cluster_id in b_cluster_ids:
                        exists_res = await session.execute(
                            select(ClusterSkillModel).where(
                                ClusterSkillModel.cluster_id == cluster_id,
                                ClusterSkillModel.skill_id == skill_a.skill_id
                            )
                        )
                        exists = exists_res.scalars().first()
                        if exists:
                            await session.execute(
                                delete(ClusterSkillModel).where(
                                    ClusterSkillModel.cluster_id == cluster_id,
                                    ClusterSkillModel.skill_id == skill_b.skill_id
                                )
                            )
                        else:
                            await session.execute(
                                update(ClusterSkillModel)
                                .where(ClusterSkillModel.cluster_id == cluster_id, ClusterSkillModel.skill_id == skill_b.skill_id)
                                .values(skill_id=skill_a.skill_id)
                            )

                    # Merge in diagnostic_skills
                    b_diagnostics_res = await session.execute(
                        select(DiagnosticSkillModel.diagnostic_id).where(DiagnosticSkillModel.skill_id == skill_b.skill_id)
                    )
                    b_diagnostic_ids = b_diagnostics_res.scalars().all()

                    for diag_id in b_diagnostic_ids:
                        exists_res = await session.execute(
                            select(DiagnosticSkillModel).where(
                                DiagnosticSkillModel.diagnostic_id == diag_id,
                                DiagnosticSkillModel.skill_id == skill_a.skill_id
                            )
                        )
                        exists = exists_res.scalars().first()
                        if exists:
                            await session.execute(
                                delete(DiagnosticSkillModel).where(
                                    DiagnosticSkillModel.diagnostic_id == diag_id,
                                    DiagnosticSkillModel.skill_id == skill_b.skill_id
                                )
                            )
                        else:
                            await session.execute(
                                update(DiagnosticSkillModel)
                                .where(DiagnosticSkillModel.diagnostic_id == diag_id, DiagnosticSkillModel.skill_id == skill_b.skill_id)
                                .values(skill_id=skill_a.skill_id)
                            )

                    # Delete skill_b
                    await session.execute(delete(SkillModel).where(SkillModel.skill_id == skill_b.skill_id))
                    await session.commit()
                    merged_ids.add(skill_b.skill_id)
                    print(f"Successfully merged '{skill_b.name}' into '{skill_a.name}'.")
                except Exception as e:
                    print(f"Error merging skills: {e}")
                    await session.rollback()

    await session.close()
    print("Database cleaning complete.")

if __name__ == "__main__":
    asyncio.run(clean_skills())
