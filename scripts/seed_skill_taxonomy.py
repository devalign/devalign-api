"""Seed script: populate the skill_relations (knowledge graph) table.

This script is idempotent — it can be run multiple times without creating
duplicate edges.  Run it once after the skills catalog has been populated
(e.g., after running seed_demo_data.py or the normalize pipeline).

Usage:
    python scripts/seed_skill_taxonomy.py

The taxonomy defines two types of upward-inference edges:
- BELONGS_TO:  concrete implementation of a concept (PostgreSQL → SQL).
- REQUIRES:    technology presupposes another (Spring Boot → Java).

Only ALTERNATIVE_TO edges are defined here where they help the UI show
contextually relevant suggestions (React ↔ Vue), but they are NOT used by
the upward inference engine — they are informational only.
"""

import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_engine.domain.entities import SkillRelationType
from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository
from src.shared.database import AsyncSessionLocal

# ---------------------------------------------------------------------------
# Taxonomy definition
#
# Format: (child_name, parent_name, relation_type)
#
# "child" is the concrete/specific skill found in a CV.
# "parent" is the abstract/generic concept that the child implies.
#
# Child → parent direction is important for BELONGS_TO and REQUIRES:
# if we see the child in a CV, we infer the parent.
# ---------------------------------------------------------------------------
TAXONOMY: list[tuple[str, str, SkillRelationType]] = [
    # ── SQL Databases ───────────────────────────────────────────────────────
    ("PostgreSQL", "SQL", SkillRelationType.BELONGS_TO),
    ("MySQL", "SQL", SkillRelationType.BELONGS_TO),
    ("SQL Server", "SQL", SkillRelationType.BELONGS_TO),
    ("Oracle DB", "SQL", SkillRelationType.BELONGS_TO),
    ("MariaDB", "SQL", SkillRelationType.BELONGS_TO),
    ("SQLite", "SQL", SkillRelationType.BELONGS_TO),
    ("CockroachDB", "SQL", SkillRelationType.BELONGS_TO),
    ("Aurora", "SQL", SkillRelationType.BELONGS_TO),
    ("Snowflake", "SQL", SkillRelationType.BELONGS_TO),
    ("Redshift", "SQL", SkillRelationType.BELONGS_TO),
    ("BigQuery", "SQL", SkillRelationType.BELONGS_TO),
    # ── Java ecosystem ──────────────────────────────────────────────────────
    ("Spring Boot", "Java", SkillRelationType.REQUIRES),
    ("Spring Framework", "Java", SkillRelationType.REQUIRES),
    ("Spring Data JPA", "Java", SkillRelationType.REQUIRES),
    ("Spring Security", "Java", SkillRelationType.REQUIRES),
    ("Hibernate", "Java", SkillRelationType.REQUIRES),
    ("Quarkus", "Java", SkillRelationType.REQUIRES),
    ("Micronaut", "Java", SkillRelationType.REQUIRES),
    ("Maven", "Java", SkillRelationType.REQUIRES),
    ("Gradle", "Java", SkillRelationType.REQUIRES),
    # ── Kotlin ecosystem ────────────────────────────────────────────────────
    ("Kotlin", "Java", SkillRelationType.BELONGS_TO),  # Runs on JVM — implies JVM knowledge
    # ── Python ecosystem ────────────────────────────────────────────────────
    ("Django", "Python", SkillRelationType.REQUIRES),
    ("FastAPI", "Python", SkillRelationType.REQUIRES),
    ("Flask", "Python", SkillRelationType.REQUIRES),
    ("SQLAlchemy", "Python", SkillRelationType.REQUIRES),
    ("Pandas", "Python", SkillRelationType.REQUIRES),
    ("NumPy", "Python", SkillRelationType.REQUIRES),
    ("Pytest", "Python", SkillRelationType.REQUIRES),
    ("Celery", "Python", SkillRelationType.REQUIRES),
    ("Pydantic", "Python", SkillRelationType.REQUIRES),
    ("Alembic", "Python", SkillRelationType.REQUIRES),
    ("scikit-learn", "Python", SkillRelationType.REQUIRES),
    ("TensorFlow", "Python", SkillRelationType.REQUIRES),
    ("PyTorch", "Python", SkillRelationType.REQUIRES),
    # ── JavaScript / TypeScript ecosystem ───────────────────────────────────
    ("React", "JavaScript", SkillRelationType.REQUIRES),
    ("Vue.js", "JavaScript", SkillRelationType.REQUIRES),
    ("Next.js", "JavaScript", SkillRelationType.REQUIRES),
    ("Nuxt.js", "JavaScript", SkillRelationType.REQUIRES),
    ("Svelte", "JavaScript", SkillRelationType.REQUIRES),
    ("Express.js", "JavaScript", SkillRelationType.REQUIRES),
    ("Node.js", "JavaScript", SkillRelationType.REQUIRES),
    ("NestJS", "JavaScript", SkillRelationType.REQUIRES),
    ("Jest", "JavaScript", SkillRelationType.REQUIRES),
    ("Vite", "JavaScript", SkillRelationType.REQUIRES),
    ("Webpack", "JavaScript", SkillRelationType.REQUIRES),
    # TypeScript implies JavaScript (TS is a superset)
    ("TypeScript", "JavaScript", SkillRelationType.BELONGS_TO),
    # TS-specific frameworks
    ("Angular", "TypeScript", SkillRelationType.REQUIRES),
    ("Angular", "JavaScript", SkillRelationType.REQUIRES),
    ("Next.js", "TypeScript", SkillRelationType.REQUIRES),
    ("NestJS", "TypeScript", SkillRelationType.REQUIRES),
    # ── PHP ecosystem ───────────────────────────────────────────────────────
    ("Laravel", "PHP", SkillRelationType.REQUIRES),
    ("Symfony", "PHP", SkillRelationType.REQUIRES),
    ("WordPress", "PHP", SkillRelationType.REQUIRES),
    # ── Ruby ecosystem ──────────────────────────────────────────────────────
    ("Ruby on Rails", "Ruby", SkillRelationType.REQUIRES),
    # ── Go ecosystem ────────────────────────────────────────────────────────
    ("Gin", "Go", SkillRelationType.REQUIRES),
    ("Echo", "Go", SkillRelationType.REQUIRES),
    ("Fiber", "Go", SkillRelationType.REQUIRES),
    # ── .NET / C# ecosystem ─────────────────────────────────────────────────
    ("ASP.NET Core", "C#", SkillRelationType.REQUIRES),
    ("Entity Framework", "C#", SkillRelationType.REQUIRES),
    ("Blazor", "C#", SkillRelationType.REQUIRES),
    ("MAUI", "C#", SkillRelationType.REQUIRES),
    # ── Cloud / Infrastructure ──────────────────────────────────────────────
    ("EKS", "Kubernetes", SkillRelationType.BELONGS_TO),
    ("GKE", "Kubernetes", SkillRelationType.BELONGS_TO),
    ("AKS", "Kubernetes", SkillRelationType.BELONGS_TO),
    ("Helm", "Kubernetes", SkillRelationType.REQUIRES),
    ("Istio", "Kubernetes", SkillRelationType.REQUIRES),
    ("Docker Swarm", "Docker", SkillRelationType.REQUIRES),
    # ── Horizontal alternatives (informational, NOT used for inference) ─────
    ("React", "Vue.js", SkillRelationType.ALTERNATIVE_TO),
    ("Vue.js", "React", SkillRelationType.ALTERNATIVE_TO),
    ("React", "Angular", SkillRelationType.ALTERNATIVE_TO),
    ("Angular", "React", SkillRelationType.ALTERNATIVE_TO),
    ("PostgreSQL", "MySQL", SkillRelationType.ALTERNATIVE_TO),
    ("MySQL", "PostgreSQL", SkillRelationType.ALTERNATIVE_TO),
    ("Django", "FastAPI", SkillRelationType.ALTERNATIVE_TO),
    ("FastAPI", "Django", SkillRelationType.ALTERNATIVE_TO),
]


async def seed_taxonomy() -> None:
    """Resolve skill names to IDs and insert the taxonomy edges."""
    print("Starting skill taxonomy seed...")

    async with AsyncSessionLocal() as session:
        repo = SQLSkillRepository(session)

        # Load all skills and build a case-insensitive name → id index
        all_skills = await repo.get_all_skills()
        if not all_skills:
            print(
                "[WARNING] No skills found in the database. "
                "Run seed_demo_data.py or the normalisation pipeline first."
            )
            return

        name_to_id = {s.name.lower(): s.id for s in all_skills if s.id}
        print(f"  Loaded {len(all_skills)} skills from catalog.")

        # Resolve taxonomy definitions to (source_id, target_id, type) triples
        resolved: list[tuple] = []
        skipped: list[str] = []

        for child_name, parent_name, rel_type in TAXONOMY:
            child_id = name_to_id.get(child_name.lower())
            parent_id = name_to_id.get(parent_name.lower())

            if not child_id or not parent_id:
                missing = []
                if not child_id:
                    missing.append(f"child='{child_name}'")
                if not parent_id:
                    missing.append(f"parent='{parent_name}'")
                skipped.append(f"  SKIP ({rel_type.value}): {', '.join(missing)} not in catalog")
                continue

            resolved.append((child_id, parent_id, rel_type))

        if skipped:
            print(f"\n  [WARNING] Skipped {len(skipped)} edges (skills not in catalog):")
            for msg in skipped:
                print(msg)

        if not resolved:
            print("\n  No edges could be resolved. Aborting.")
            return

        print(f"\n  Inserting {len(resolved)} edges (skipping existing ones)...")
        await repo.add_relations(resolved)
        await session.commit()
        print(f"Taxonomy seed complete. {len(resolved)} edges processed.")


if __name__ == "__main__":
    asyncio.run(seed_taxonomy())
