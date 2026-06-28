import json
import logging
import os
import sys
import requests
from dotenv import load_dotenv
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GROQ_API_KEY:
    logger.error("Missing credentials in .env")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

DOMAINS = ["Backend", "Frontend", "Mobile", "QA", "DevOps", "Cloud", "Data"]


def classify_batch(skill_names: list[str]) -> dict[str, list[str]]:
    """Classifies a batch of skills using Groq LLM."""
    prompt = f"""You are a technical expert. Classify the following IT skills into one or more of these core domains: {", ".join(DOMAINS)}.
Each skill can belong to multiple domains (e.g. 'AWS' -> ['Cloud', 'DevOps']). If a skill is unrelated or does not fit any domain, return an empty list [].
Return a JSON object where keys are the skill names (exactly as provided) and values are arrays of domains.
Example:
{{"python": ["Backend"], "react": ["Frontend"], "aws": ["Cloud", "DevOps"], "marketing": []}}

Skills to classify:
{json.dumps(skill_names)}
"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a JSON returning bot. Output only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            return result
        else:
            logger.error(f"API Error: {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Failed to call LLM: {e}")
        return {}


def main():
    logger.info("Fetching skills from Supabase...")
    # Fetch skill_id, name, and core_domains
    response = supabase.table("skills").select("skill_id, name, core_domains").execute()
    skills = response.data

    if not skills:
        logger.error("No skills found.")
        sys.exit(1)

    # Filter skills that have empty or null core_domains
    skills_to_update = []
    for s in skills:
        core_doms = s.get("core_domains")
        if core_doms is None or core_doms == [] or core_doms == "[]":
            skills_to_update.append(s)

    logger.info(f"Loaded {len(skills)} total skills. Found {len(skills_to_update)} skills with empty core_domains.")

    if not skills_to_update:
        logger.info("No skills need backfilling.")
        return

    # Process in batches of 50
    batch_size = 50
    for i in range(0, len(skills_to_update), batch_size):
        chunk = skills_to_update[i : i + batch_size]
        chunk_names = [s["name"] for s in chunk]
        
        logger.info(f"Classifying batch {i//batch_size + 1} ({len(chunk)} skills)...")
        classifications = classify_batch(chunk_names)
        
        for s in chunk:
            name = s["name"]
            skill_id = s["skill_id"]
            assigned_domains = classifications.get(name, [])
            
            # Validate assigned domains are part of our official list
            valid_assigned = [d for d in assigned_domains if d in DOMAINS]
            
            logger.info(f"Updating '{name}' -> {valid_assigned}")
            supabase.table("skills").update({"core_domains": valid_assigned}).eq(
                "skill_id", skill_id
            ).execute()

    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
