"""Seed script: Parse and seed Lightcast Open Skills into PostgreSQL.

Populates:
- skills: Canonical technical skills with core_domains and domain_tags
- skill_standards: Official Lightcast standard mapping (standard_name="Lightcast", standard_code, standard_uri)
- skill_aliases: Common aliases and simplified variants

Usage:
    .venv/Scripts/python scripts/seed_lightcast_skills.py
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

import httpx
import structlog
from sqlalchemy import select

# Setup path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_engine.domain.entities import SkillNature
from src.ml_engine.infrastructure.models import SkillAliasModel, SkillModel, SkillStandardModel
from src.shared.database import AsyncSessionLocal

logger = structlog.get_logger(__name__)

LIGHTCAST_GIST_URL = (
    "https://gist.githubusercontent.com/ThatGuySam/8a6e7bd152793ac12b7f60420d1017c8/raw/skills.json"
)
CACHE_DIR = Path(__file__).parent / "data"
CACHE_FILE = CACHE_DIR / "lightcast_raw.json"

TECH_KEYWORDS = [
    "software",
    "programming",
    "developer",
    "development",
    "web",
    "database",
    "sql",
    "cloud",
    "devops",
    "aws",
    "azure",
    "gcp",
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "golang",
    "rust",
    "ruby",
    "php",
    "html",
    "css",
    "react",
    "angular",
    "vue",
    "node.js",
    "nodejs",
    "docker",
    "kubernetes",
    "linux",
    "git",
    "api",
    "rest",
    "graphql",
    "kafka",
    "rabbitmq",
    "spring",
    "django",
    "flask",
    "fastapi",
    "microservices",
    "frontend",
    "backend",
    "full stack",
    "qa",
    "testing",
    "cybersecurity",
    "machine learning",
    "artificial intelligence",
    "data science",
    "data engineering",
    "etl",
    "bigquery",
    "snowflake",
    "postgresql",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "ci/cd",
    "jenkins",
    "agile",
    "scrum",
    "jira",
    "bitbucket",
    "github",
    "gitlab",
    "terraform",
    "ansible",
    ".net",
    "ios",
    "android",
    "flutter",
    "react native",
    "swift",
    "kotlin",
    "scala",
    "grpc",
    "graphql",
    "oauth",
    "jwt",
    "linux",
    "unix",
    "bash",
    "powershell",
    "server",
    "security",
    "network",
    "networking",
    "distributed systems",
    "elasticsearch",
    "cassandra",
    "graphql",
    "helm",
    "prometheus",
    "grafana",
    "nginx",
    "apache",
]

TECH_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in TECH_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

NON_TECH_CERT_KEYWORDS = [
    "fire",
    "pilot",
    "captain",
    "concrete",
    "personal trainer",
    "aerobics",
    "medical",
    "nurse",
    "dental",
    "welding",
    "cosmetology",
    "plumbing",
    "crane",
    "forklift",
]


def is_tech_skill(item: dict) -> bool:
    """Determine if a Lightcast skill entry is a relevant technical skill."""
    name = item.get("name", "")
    type_info = item.get("type", {})
    type_name = type_info.get("name", "") if isinstance(type_info, dict) else ""

    name_lower = name.lower()

    # Exclude non-IT certifications
    if type_name == "Certification":
        if any(bad in name_lower for bad in NON_TECH_CERT_KEYWORDS):
            return False
        # Only allow tech-related certifications
        if not any(
            k in name_lower
            for k in [
                "aws",
                "azure",
                "google",
                "cloud",
                "cisco",
                "developer",
                "scrum",
                "agile",
                "security",
                "comptia",
                "linux",
                "python",
                "java",
                "database",
                "oracle",
                "microsoft",
            ]
        ):
            return False

    return bool(TECH_PATTERN.search(name))


def infer_domains(name: str) -> tuple[list[str], list[str]]:
    """Infer core_domains and domain_tags based on skill name patterns."""
    nl = name.lower()
    core_domains = set()
    domain_tags = set()

    if any(
        k in nl
        for k in [
            "react",
            "angular",
            "vue",
            "frontend",
            "html",
            "css",
            "ui",
            "javascript",
            "typescript",
        ]
    ):
        core_domains.add("frontend")
        domain_tags.add("Frontend Development")

    if any(
        k in nl
        for k in [
            "backend",
            "api",
            "rest",
            "graphql",
            "grpc",
            "spring",
            "django",
            "fastapi",
            "express",
            "node",
            "java",
            "c#",
            "golang",
            "php",
            "ruby",
        ]
    ):
        core_domains.add("backend")
        domain_tags.add("Application Programming Interface (API)")
        domain_tags.add("Back-End Development")

    if any(
        k in nl
        for k in [
            "database",
            "sql",
            "postgres",
            "mysql",
            "mongodb",
            "redis",
            "cassandra",
            "data storage",
        ]
    ):
        core_domains.add("data")
        domain_tags.add("Databases")
        domain_tags.add("Data Storage")

    if any(
        k in nl
        for k in [
            "data",
            "etl",
            "machine learning",
            "ai",
            "artificial intelligence",
            "bigquery",
            "snowflake",
            "spark",
        ]
    ):
        core_domains.add("data")
        domain_tags.add("Artificial Intelligence and Machine Learning (AI/ML)")
        domain_tags.add("Data Management")

    if any(
        k in nl
        for k in [
            "cloud",
            "aws",
            "azure",
            "gcp",
            "docker",
            "kubernetes",
            "devops",
            "ci/cd",
            "terraform",
            "jenkins",
            "git",
        ]
    ):
        core_domains.add("cloud_devops")
        domain_tags.add("Cloud Computing")
        domain_tags.add("DevOps")

    if any(k in nl for k in ["test", "testing", "qa", "selenium", "cypress", "junit"]):
        core_domains.add("qa")
        domain_tags.add("Quality Assurance and Testing")

    if any(k in nl for k in ["security", "cyber", "oauth", "jwt", "keycloak"]):
        core_domains.add("security")
        domain_tags.add("Cybersecurity")

    if not core_domains:
        core_domains.add("software_engineering")
        domain_tags.add("Software Engineering")

    return sorted(core_domains), sorted(domain_tags)


async def fetch_lightcast_skills() -> list[dict]:
    """Load skills from local cache or download from public gist."""
    if CACHE_FILE.exists():
        logger.info("Loading Lightcast skills from local cache", path=str(CACHE_FILE))
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)

    logger.info("Downloading Lightcast Open Skills from public gist...", url=LIGHTCAST_GIST_URL)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(LIGHTCAST_GIST_URL)
        resp.raise_for_status()
        data = resp.json()
        skills = data.get("data", [])

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False)
    logger.info("Saved raw Lightcast skills to cache", count=len(skills))
    return skills


async def seed():
    """Main seeding function."""
    raw_skills = await fetch_lightcast_skills()
    logger.info("Total raw skills downloaded", total=len(raw_skills))

    tech_skills = [s for s in raw_skills if is_tech_skill(s)]
    logger.info("Filtered technical skills", count=len(tech_skills))

    async with AsyncSessionLocal() as session:
        # 1. Fetch existing skills
        res_skills = await session.execute(select(SkillModel))
        existing_skills = res_skills.scalars().all()
        skill_by_name = {s.name.lower().strip(): s for s in existing_skills}

        # 2. Fetch existing standards
        res_standards = await session.execute(
            select(SkillStandardModel).where(SkillStandardModel.standard_name == "Lightcast")
        )
        existing_standards = {s.standard_code: s for s in res_standards.scalars().all()}

        # 3. Fetch existing aliases
        res_aliases = await session.execute(select(SkillAliasModel))
        existing_aliases = {a.alias_name.lower().strip(): a for a in res_aliases.scalars().all()}

        skills_created = 0
        standards_created = 0
        aliases_created = 0

        for item in tech_skills:
            code = item["id"]
            name = item["name"].strip()
            uri = item.get("infoUrl") or f"https://skills.emsidata.com/skills/{code}"
            name_key = name.lower()

            core_domains, domain_tags = infer_domains(name)

            # Check if skill exists
            skill = skill_by_name.get(name_key)
            if not skill:
                skill = SkillModel(
                    skill_id=uuid4(),
                    name=name,
                    nature=SkillNature.TECH.value,
                    core_domains=core_domains,
                    domain_tags=domain_tags,
                    weight=1.0,
                )
                session.add(skill)
                skill_by_name[name_key] = skill
                skills_created += 1
            else:
                # Merge domain tags
                merged_tags = set(skill.domain_tags or []) | set(domain_tags)
                skill.domain_tags = sorted(merged_tags)
                merged_core = set(skill.core_domains or []) | set(core_domains)
                skill.core_domains = sorted(merged_core)

            # Link standard if not already linked
            if code not in existing_standards:
                standard = SkillStandardModel(
                    id=uuid4(),
                    skill_id=skill.skill_id,
                    standard_name="Lightcast",
                    standard_uri=uri,
                    standard_code=code,
                )
                session.add(standard)
                existing_standards[code] = standard
                standards_created += 1

            # Check for parenthetical aliases (e.g. ".NET MAUI (Multi-Platform App UI)" -> ".NET MAUI")
            paren_match = re.match(r"^(.*?)\s*\((.*?)\)$", name)
            if paren_match:
                main_part = paren_match.group(1).strip()
                sub_part = paren_match.group(2).strip()

                for variant in [main_part, sub_part]:
                    variant_key = variant.lower()
                    if (
                        len(variant) >= 2
                        and variant_key != name_key
                        and variant_key not in existing_aliases
                        and variant_key not in skill_by_name
                    ):
                        alias = SkillAliasModel(
                            alias_id=uuid4(),
                            alias_name=variant,
                            skill_id=skill.skill_id,
                        )
                        session.add(alias)
                        existing_aliases[variant_key] = alias
                        aliases_created += 1

        await session.commit()
        logger.info(
            "Lightcast seeding complete!",
            skills_created=skills_created,
            standards_created=standards_created,
            aliases_created=aliases_created,
            total_skills_now=len(skill_by_name),
            total_standards_now=len(existing_standards),
        )


if __name__ == "__main__":
    asyncio.run(seed())
