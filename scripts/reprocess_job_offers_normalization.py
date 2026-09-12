"""Reprocess and normalize all job offers in the database.

Matches raw_hard_skills against the canonical skills catalog (Lightcast + curated Custom),
expands slash-separated compounds, creates offer_skills relations,
assigns best-matching cluster_id, and sets is_normalized = True.
"""

import asyncio
import logging
import sys
from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert

from src.ml_engine.infrastructure.models import (
    ClusterModel,
    ClusterSkillModel,
    SkillAliasModel,
    SkillModel,
)
from src.scraper.infrastructure.models import JobOfferModel, OfferSkillModel
from src.shared.database import AsyncSessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SINGLE_SLASH_TERMS = {
    "ci/cd",
    "tcp/ip",
    "i/o",
    "pl/sql",
    "client/server",
    "lan/wan",
    "etl/elt",
    "os/2",
    "ui/ux",
}


def clean_norm_key(s: str) -> str:
    """Normalize string by removing whitespace, punctuation, and lowercasing."""
    return (
        s.strip()
        .lower()
        .replace(" ", "")
        .replace(".", "")
        .replace("-", "")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
    )


async def main():
    logger.info("Starting high-performance job offers batch normalization...")

    async with AsyncSessionLocal() as session:
        # 1. Load canonical skills and aliases
        logger.info("Loading canonical skills and aliases...")
        skills_query = select(SkillModel.skill_id, SkillModel.name).where(
            SkillModel.status == "canonical"
        )
        skills_res = await session.execute(skills_query)
        canonical_skills = skills_res.all()

        aliases_query = select(SkillAliasModel.alias_name, SkillAliasModel.skill_id)
        aliases_res = await session.execute(aliases_query)
        skill_aliases = aliases_res.all()

        # Build lookup table
        norm_to_skill: dict[str, UUID] = {}
        for s_id, s_name in canonical_skills:
            clean = s_name.strip().lower()
            norm_to_skill[clean] = s_id
            norm_to_skill[clean.replace(" ", "").replace(".", "")] = s_id
            norm_to_skill[clean_norm_key(s_name)] = s_id

        for a_name, s_id in skill_aliases:
            norm_to_skill[a_name.strip().lower()] = s_id
            norm_to_skill[clean_norm_key(a_name)] = s_id

        logger.info(f"Loaded {len(canonical_skills)} canonical skills with {len(norm_to_skill)} index keys.")

        # 2. Load clusters and cluster_skills
        logger.info("Loading clusters and centroid skills...")
        clusters_query = select(ClusterModel.cluster_id, ClusterModel.name)
        clusters_res = await session.execute(clusters_query)
        clusters = {c.cluster_id: c.name for c in clusters_res.all()}

        cluster_skills_query = select(
            ClusterSkillModel.cluster_id,
            ClusterSkillModel.skill_id,
            ClusterSkillModel.importance_score,
        )
        cs_res = await session.execute(cluster_skills_query)

        # cluster_id -> dict[skill_id, importance_score]
        cluster_skill_profiles: dict[UUID, dict[UUID, float]] = {c_id: {} for c_id in clusters}
        for c_id, s_id, imp in cs_res.all():
            if c_id in cluster_skill_profiles:
                cluster_skill_profiles[c_id][s_id] = float(imp) if imp is not None else 1.0

        logger.info(f"Loaded {len(clusters)} clusters with their skill profiles.")

        # 3. Load all job offers
        logger.info("Fetching all job offers...")
        offers_query = select(
            JobOfferModel.job_offer_id,
            JobOfferModel.job_title,
            JobOfferModel.raw_hard_skills,
            JobOfferModel.cluster_id,
        )
        offers_res = await session.execute(offers_query)
        all_offers = offers_res.all()
        total_offers = len(all_offers)
        logger.info(f"Retrieved {total_offers} total job offers.")

        # 4. Clear existing offer_skills to avoid any duplicates and re-link cleanly
        logger.info("Clearing existing offer_skills table...")
        await session.execute(delete(OfferSkillModel))
        await session.commit()
        logger.info("Cleared offer_skills table.")

        # 5. Process all offers in memory
        logger.info("Processing offers in memory...")
        all_offer_skills_to_insert = []
        cluster_to_offer_ids: dict[UUID | None, list[UUID]] = defaultdict(list)
        offers_with_skills = 0
        offers_assigned_cluster = 0
        cluster_assignment_counts: dict[UUID, int] = {c_id: 0 for c_id in clusters}

        for offer in all_offers:
            o_id = offer.job_offer_id
            raw_skills = offer.raw_hard_skills or []

            # Expand slashes unless in whitelist
            expanded_skills: list[str] = []
            for item in raw_skills:
                if not isinstance(item, str):
                    continue
                clean_item = item.strip()
                if "/" in clean_item and clean_item.lower() not in SINGLE_SLASH_TERMS:
                    parts = [p.strip() for p in clean_item.split("/") if p.strip()]
                    expanded_skills.extend(parts)
                else:
                    expanded_skills.append(clean_item)

            # Match against canonical index
            matched_skill_ids: set[UUID] = set()
            for term in expanded_skills:
                term_lower = term.lower()
                term_clean = clean_norm_key(term)

                target_id = norm_to_skill.get(term_lower) or norm_to_skill.get(term_clean)
                if target_id:
                    matched_skill_ids.add(target_id)

            if matched_skill_ids:
                offers_with_skills += 1
                for s_id in matched_skill_ids:
                    all_offer_skills_to_insert.append(
                        {
                            "job_offer_id": o_id,
                            "skill_id": s_id,
                            "skill_type": "hard_skill",
                            "importance_score": 1.0,
                        }
                    )

            # Determine best-matching cluster
            best_cluster_id = None
            best_score = 0.0

            if matched_skill_ids:
                for c_id, c_skills in cluster_skill_profiles.items():
                    intersect = matched_skill_ids & c_skills.keys()
                    if intersect:
                        score = sum(c_skills[sid] for sid in intersect)
                        if score > best_score:
                            best_score = score
                            best_cluster_id = c_id

            if best_cluster_id:
                offers_assigned_cluster += 1
                cluster_assignment_counts[best_cluster_id] += 1
            elif offer.cluster_id and offer.cluster_id in clusters:
                # Keep existing cluster if valid and no better skill match found
                best_cluster_id = offer.cluster_id
                cluster_assignment_counts[best_cluster_id] += 1

            cluster_to_offer_ids[best_cluster_id].append(o_id)

        # 6. Bulk insert offer_skills in chunks
        chunk_size = 2500
        total_links = len(all_offer_skills_to_insert)
        logger.info(f"Inserting {total_links} offer_skills in chunks of {chunk_size}...")
        for i in range(0, total_links, chunk_size):
            chunk = all_offer_skills_to_insert[i : i + chunk_size]
            await session.execute(insert(OfferSkillModel), chunk)
            logger.info(f"Inserted offer_skills: {min(i + chunk_size, total_links)}/{total_links}")

        await session.commit()

        # 7. Bulk update job_offers grouped by assigned cluster (at most 37 queries)
        logger.info(f"Updating job_offers table across {len(cluster_to_offer_ids)} cluster groups...")
        for c_id, o_ids in cluster_to_offer_ids.items():
            # Process in sub-batches of 1000 IDs to avoid query param overflow
            sub_batch_size = 1000
            for j in range(0, len(o_ids), sub_batch_size):
                sub_ids = o_ids[j : j + sub_batch_size]
                await session.execute(
                    update(JobOfferModel)
                    .where(JobOfferModel.job_offer_id.in_(sub_ids))
                    .values(
                        cluster_id=c_id,
                        is_normalized=True,
                    )
                )
        await session.commit()

        # 8. Update clusters.job_offer_count with new counts
        logger.info("Updating cluster job_offer_count statistics...")
        for c_id, count in cluster_assignment_counts.items():
            await session.execute(
                update(ClusterModel)
                .where(ClusterModel.cluster_id == c_id)
                .values(job_offer_count=count)
            )
        await session.commit()

        logger.info("=== Batch Normalization Summary ===")
        logger.info(f"Total job offers: {total_offers}")
        logger.info(
            f"Offers with canonical skills: {offers_with_skills} ({offers_with_skills/total_offers*100:.1f}%)"
        )
        logger.info(
            f"Offers assigned to clusters: {offers_assigned_cluster} ({offers_assigned_cluster/total_offers*100:.1f}%)"
        )
        logger.info(f"Total offer_skills links created: {total_links}")


if __name__ == "__main__":
    asyncio.run(main())
