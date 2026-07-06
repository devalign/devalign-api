import asyncio
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import or_, select

from src.ml_engine.infrastructure.llm_client import get_llm_service
from src.ml_engine.infrastructure.models import SkillModel
from src.shared.database import AsyncSessionLocal


async def main():
    session = AsyncSessionLocal()
    llm = get_llm_service()

    print(f"Starting DB domains and tags backfill using: {llm.__class__.__name__}")

    try:
        # Fetch skills with empty core_domains or empty domain_tags
        stmt = select(SkillModel).where(
            or_(
                SkillModel.core_domains.is_(None),
                SkillModel.core_domains == [],
                SkillModel.domain_tags.is_(None),
                SkillModel.domain_tags == [],
            )
        )
        res = await session.execute(stmt)
        skills = res.scalars().all()

        print(f"Found {len(skills)} skills with missing core_domains or domain_tags.")

        if not skills:
            print("No skills need backfilling.")
            return

        domains = ["Backend", "Frontend", "Mobile", "QA", "DevOps", "Cloud", "Data"]
        batch_size = 50

        for i in range(0, len(skills), batch_size):
            batch = skills[i : i + batch_size]
            names = [s.name for s in batch]
            print(f"Classifying batch {i // batch_size + 1} ({len(batch)} skills): {names}")

            prompt = (
                f"You are a technical expert. Classify the following IT skills into one or more of these core domains: {', '.join(domains)}.\n"
                "Each skill can belong to multiple domains (e.g. 'AWS' -> ['Cloud', 'DevOps']). If a skill is unrelated, use an empty list [].\n"
                "Additionally, generate 2-4 lowercase tag strings (domain_tags) for each skill to help with indexing (e.g. 'PostgreSQL' -> ['database', 'postgresql']).\n"
                "Return a JSON object where keys are the skill names (exactly as provided) and values are objects containing 'core_domains' (array) and 'domain_tags' (array).\n"
                "Example:\n"
                '{"python": {"core_domains": ["Backend"], "domain_tags": ["python", "backend"]}, '
                '"aws": {"core_domains": ["Cloud", "DevOps"], "domain_tags": ["cloud", "aws", "infrastructure"]}}\n\n'
                f"Skills to classify: {json.dumps(names)}"
            )

            # Sleep between batches to proactively avoid rate limits
            if i > 0:
                print("Waiting 15 seconds between batches to avoid rate limits...")
                await asyncio.sleep(15)

            try:
                llm_response = None
                for attempt in range(5):
                    try:
                        llm_response = await llm.generate(prompt)
                        break
                    except Exception as exc:
                        if (
                            "rate limit" in str(exc).lower()
                            or "429" in str(exc)
                            or "too many requests" in str(exc).lower()
                        ):
                            wait_time = (2**attempt) + 10
                            print(
                                f"Rate limit hit. Waiting {wait_time} seconds before retrying (attempt {attempt + 1}/5)..."
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            raise exc

                if not llm_response:
                    raise Exception(
                        "Failed to get response from LLM after 5 attempts due to rate limits."
                    )

                classifications = json.loads(llm_response)

                for s in batch:
                    res_cls = classifications.get(s.name) or {}
                    assigned_domains = res_cls.get("core_domains") or []
                    assigned_tags = res_cls.get("domain_tags") or []

                    s.core_domains = [d for d in assigned_domains if d in domains]
                    s.domain_tags = [str(t).lower().strip() for t in assigned_tags]

                    print(f"  Skill '{s.name}' -> core: {s.core_domains}, tags: {s.domain_tags}")

                await session.commit()
                print(f"Successfully classified and committed batch {i // batch_size + 1}.")
            except Exception as e:
                print(f"Error classifying batch: {e}")
                await session.rollback()
                raise e

    except Exception as e:
        print(f"\nBackfill failed: {e}")
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
