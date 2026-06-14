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

DOMAINS = ["Backend", "Frontend", "Data", "DevOps", "Cloud", "Mobile", "QA"]


def main():
    logger.info("Fetching skills from Supabase...")
    response = supabase.table("skills").select("skill_id, name").execute()
    skills = response.data

    if not skills:
        logger.error("No skills found.")
        sys.exit(1)

    skill_names = [s["name"] for s in skills]
    logger.info(f"Loaded {len(skills)} skills to classify.")

    # Call LLM
    prompt = f"""You are a technical expert. Classify the following IT skills into EXACTLY ONE of these domains: {", ".join(DOMAINS)}.
If a skill is completely unrelated (e.g. 'marketing'), use 'Other'.
Return a JSON object where keys are the skill names and values are the domains.
Example: {{"python": "Backend", "react": "Frontend", "aws": "Cloud"}}

Skills to classify:
{", ".join(skill_names)}
"""

    logger.info("Calling Groq LLM...")
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
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            classifications = json.loads(content)

            logger.info("Updating skills in database...")
            for skill in skills:
                name = skill["name"]
                domain = classifications.get(name)
                if domain in DOMAINS:
                    supabase.table("skills").update({"domain": domain}).eq(
                        "skill_id", skill["skill_id"]
                    ).execute()
                else:
                    # Fallback mapping if LLM fails
                    supabase.table("skills").update({"domain": "Backend"}).eq(
                        "skill_id", skill["skill_id"]
                    ).execute()

            logger.info("Successfully classified and updated all skills.")
        else:
            logger.error(f"API Error: {response.text}")
    except Exception as e:
        logger.error(f"Failed to classify domains: {e}")


if __name__ == "__main__":
    main()
