"""Re-extract skills for job offers with deficient skill counts using Groq LLM (openai/gpt-oss-120b).

Normalizes all extracted skills against the canonical Lightcast catalog in Supabase,
persisting canonical links into offer_skills and updating raw_hard_skills in job_offers.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

import httpx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import delete, func, select, update
from src.config import settings
from src.ml_engine.application.skill_catalog_service import (
    build_skill_lookup_indexes,
    match_single_skill,
)
from src.ml_engine.infrastructure.embeddings import get_embedding_service
from src.ml_engine.infrastructure.models import SkillModel
from src.scraper.infrastructure.models import JobOfferModel, OfferSkillModel
from src.shared.database import AsyncSessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReExtractSkills")


def parse_reset_time(val: str | None) -> float:
    """Parses reset time strings like '22.597s', '300ms', '1m10s', or plain floats."""
    if not val:
        return 2.0
    val = val.strip().lower()
    try:
        if val.endswith("ms"):
            return float(val[:-2]) / 1000.0
        if val.endswith("s"):
            return float(val[:-1])
        if val.endswith("m"):
            return float(val[:-1]) * 60.0
        return float(val)
    except ValueError:
        return 5.0


async def call_groq_extract(
    client: httpx.AsyncClient, title: str, description: str
) -> tuple[list[str], float]:
    """Extracts technical hard skills from job title and description via Groq API.
    
    Returns:
        (extracted_skills_list, recommended_adaptive_sleep_seconds)
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Slice description to 1800 chars to maximize relevance and stay within TPM budget
    prompt = (
        f"Título del puesto: {title}\n\n"
        f"Descripción de la vacante:\n{description[:1800]}\n\n"
        "INSTRUCCIÓN:\n"
        "Extrae una lista JSON de las competencias técnicas y hard skills reales requeridas o valoradas (lenguajes, frameworks, herramientas, bases de datos, nubes, metodologías técnicas como Scrum/Kanban, y disciplinas de ingeniería como QA Manual, API Testing, Microservicios).\n"
        "REGLAS:\n"
        "1. NO inventes habilidades que no aparezcan ni se requieran en el texto.\n"
        "2. NO agregues habilidades blandas genéricas (como puntualidad o liderazgo).\n"
        "3. NO agregues términos hiper-genéricos como 'Software' o 'Tecnología de la información'.\n"
        "4. Devuelve ÚNICAMENTE un JSON con esta estructura: {\"hard_skills\": [\"Python\", \"FastAPI\", ...]}"
    )

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": "Eres un extractor técnico especializado en perfiles de ingeniería de software.",
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

    for attempt in range(5):
        try:
            resp = await client.post(url, headers=headers, json=payload, timeout=20.0)
            if resp.status_code == 200:
                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"]
                parsed = json.loads(raw_content)
                skills = parsed.get("hard_skills", [])

                # Adaptive rate-limiting inspection
                rem_tokens = int(resp.headers.get("x-ratelimit-remaining-tokens", 8000))
                reset_str = resp.headers.get("x-ratelimit-reset-tokens", "2s")
                if rem_tokens < 1500:
                    recommended_sleep = parse_reset_time(reset_str) + 0.5
                else:
                    recommended_sleep = 1.0

                return skills, recommended_sleep

            elif resp.status_code == 429:
                retry_after_str = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-tokens")
                wait_time = max(parse_reset_time(retry_after_str), 5.0) + 1.0
                logger.warning(f"Groq 429 Rate Limit. Adaptive pause {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.warning(f"Groq error {resp.status_code}: {resp.text[:150]}")
                await asyncio.sleep(2.0)
        except Exception as e:
            logger.warning(f"Request exception on attempt {attempt+1}: {e}")
            await asyncio.sleep(2.0)

    return [], 2.0


async def run_reextraction(max_skills_filter: int = 2, max_offers: int = 600):
    """Re-extracts skills for LATAM offers with <= max_skills_filter skills."""
    embedding_service = get_embedding_service()

    async with AsyncSessionLocal() as session:
        logger.info("Loading canonical skills catalog from DB...")
        from sqlalchemy.orm.attributes import flag_modified
        from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository

        skill_repo = SQLSkillRepository(session)
        all_skills = await skill_repo.get_all_skills()
        alias_to_skill, norm_to_skill = build_skill_lookup_indexes(all_skills)
        logger.info(f"Loaded {len(all_skills)} canonical skills.")

        # Find target offers from LATAM with deficient skill counts
        logger.info(
            f"Querying LATAM offers with <= {max_skills_filter} skills in raw_hard_skills..."
        )
        query = (
            select(JobOfferModel)
            .where(JobOfferModel.portal.in_(["computrabajo", "getonboard"]))
            .where(JobOfferModel.full_description.is_not(None))
        )
        offers_res = await session.execute(query)
        all_latam_offers = offers_res.scalars().all()

        target_offers = [
            o
            for o in all_latam_offers
            if len(o.raw_hard_skills or []) <= max_skills_filter
        ][:max_offers]

        logger.info(
            f"Found {len(target_offers)} target offers to enrich with Groq LLM (out of {len(all_latam_offers)} total)."
        )

        if not target_offers:
            logger.info("No offers match criteria.")
            return

        enriched_count = 0
        new_links_count = 0

        async with httpx.AsyncClient() as http_client:
            for idx, offer in enumerate(target_offers):
                logger.info(
                    f"[{idx+1}/{len(target_offers)}] Re-extracting: '{offer.job_title[:45]}' (Current skills: {offer.raw_hard_skills})"
                )

                extracted_skills, recommended_sleep = await call_groq_extract(
                    http_client, offer.job_title, offer.full_description
                )

                if not extracted_skills:
                    logger.info("  -> No skills extracted, keeping existing.")
                    await asyncio.sleep(recommended_sleep)
                    continue

                # Clean and combine with existing
                merged_raw = list(
                    dict.fromkeys((offer.raw_hard_skills or []) + extracted_skills)
                )
                offer.raw_hard_skills = merged_raw
                flag_modified(offer, "raw_hard_skills")

                # Match against canonical catalog
                canonical_skill_ids = set()
                for skill_name in merged_raw:
                    if not isinstance(skill_name, str) or len(skill_name.strip()) < 2:
                        continue
                    matched = match_single_skill(
                        skill_name.strip(), alias_to_skill, norm_to_skill
                    )
                    if matched and matched.id:
                        canonical_skill_ids.add(matched.id)

                # Persist offer_skills relations
                if canonical_skill_ids:
                    # Clear old links for this offer
                    await session.execute(
                        delete(OfferSkillModel).where(
                            OfferSkillModel.job_offer_id == offer.job_offer_id
                        )
                    )
                    for sid in canonical_skill_ids:
                        session.add(
                            OfferSkillModel(
                                job_offer_id=offer.job_offer_id,
                                skill_id=sid,
                                skill_type="hard_skill",
                                importance_score=1.0,
                            )
                        )
                        new_links_count += 1

                enriched_count += 1
                logger.info(
                    f"  -> Extracted {len(extracted_skills)} skills. Linked {len(canonical_skill_ids)} canonical skills."
                )

                # Commit every 10 offers to save progress
                if (idx + 1) % 10 == 0:
                    await session.commit()
                    logger.info(
                        f"*** Checkpoint saved: {enriched_count} offers enriched so far. ***"
                    )

                # Short delay to respect Groq rate limits (adaptive)
                await asyncio.sleep(recommended_sleep)

        await session.commit()
        logger.info(
            f"=== COMPLETED: Enriched {enriched_count} offers with {new_links_count} canonical offer_skills links! ==="
        )


if __name__ == "__main__":
    max_filter = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    max_count = int(sys.argv[2]) if len(sys.argv) > 2 else 550
    asyncio.run(
        run_reextraction(max_skills_filter=max_filter, max_offers=max_count)
    )
