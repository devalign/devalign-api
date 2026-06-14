"""Development seed script for Devalign tech clusters and skills with weights and frequencies."""

import asyncio
import os
import random
import sys
from uuid import uuid4

# Append src/ to path so we can import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_engine.infrastructure.models import (
    EMBEDDING_DIM,
    ClusterModel,
    ClusterSkillModel,
    SkillModel,
)
from src.shared.database import AsyncSessionLocal


async def seed() -> None:
    print("Starting development database seeding with weights and frequencies...")
    session = AsyncSessionLocal()

    # Generate a random unit vector for centroids (since pgvector needs it for compatibility)
    def make_mock_vector():
        vec = [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec]

    # Cluster definition list
    clusters_data = [
        {
            "name": "Backend Java",
            "description": "Desarrollo de lógica de servidor, APIs y arquitectura de microservicios robusta en ecosistema Java/Spring.",
            "job_offer_count": 145,
            "skills": [
                {"name": "Java", "category": "hard_skill", "weight": 3.0, "frequency": 0.92},
                {"name": "Spring Boot", "category": "hard_skill", "weight": 3.0, "frequency": 0.88},
                {"name": "REST APIs", "category": "hard_skill", "weight": 2.0, "frequency": 0.80},
                {
                    "name": "Microservicios",
                    "category": "hard_skill",
                    "weight": 2.0,
                    "frequency": 0.82,
                },
                {"name": "PostgreSQL", "category": "hard_skill", "weight": 2.0, "frequency": 0.74},
                {"name": "AWS", "category": "hard_skill", "weight": 2.0, "frequency": 0.62},
                {"name": "Docker", "category": "tool", "weight": 2.0, "frequency": 0.70},
                {"name": "Kubernetes", "category": "tool", "weight": 2.0, "frequency": 0.55},
                {"name": "Git", "category": "tool", "weight": 1.0, "frequency": 0.90},
                {"name": "Liderazgo", "category": "soft_skill", "weight": 1.0, "frequency": 0.85},
                {
                    "name": "Comunicación",
                    "category": "soft_skill",
                    "weight": 1.0,
                    "frequency": 0.80,
                },
                {"name": "Scrum", "category": "methodology", "weight": 1.0, "frequency": 0.82},
            ],
        },
        {
            "name": "Backend Python",
            "description": "Desarrollo ágil de microservicios y APIs utilizando Python, FastAPI y bases de datos modernas.",
            "job_offer_count": 110,
            "skills": [
                {"name": "Python", "category": "hard_skill", "weight": 3.0, "frequency": 0.95},
                {"name": "FastAPI", "category": "hard_skill", "weight": 3.0, "frequency": 0.82},
                {"name": "PostgreSQL", "category": "hard_skill", "weight": 2.0, "frequency": 0.78},
                {"name": "MongoDB", "category": "hard_skill", "weight": 2.0, "frequency": 0.60},
                {"name": "REST APIs", "category": "hard_skill", "weight": 2.0, "frequency": 0.85},
                {"name": "AWS", "category": "hard_skill", "weight": 2.0, "frequency": 0.68},
                {"name": "Docker", "category": "tool", "weight": 2.0, "frequency": 0.75},
                {"name": "Redis", "category": "hard_skill", "weight": 1.5, "frequency": 0.55},
                {"name": "Git", "category": "tool", "weight": 1.0, "frequency": 0.92},
                {
                    "name": "Trabajo en equipo",
                    "category": "soft_skill",
                    "weight": 1.0,
                    "frequency": 0.88,
                },
                {
                    "name": "Resolución de problemas",
                    "category": "soft_skill",
                    "weight": 1.0,
                    "frequency": 0.90,
                },
                {"name": "Scrum", "category": "methodology", "weight": 1.0, "frequency": 0.80},
            ],
        },
        {
            "name": "Frontend React",
            "description": "Construcción de interfaces de usuario interactivas y SPA modernas usando React, Next.js y CSS moderno.",
            "job_offer_count": 190,
            "skills": [
                {"name": "React", "category": "hard_skill", "weight": 3.0, "frequency": 0.96},
                {"name": "Next.js", "category": "hard_skill", "weight": 3.0, "frequency": 0.85},
                {"name": "JavaScript", "category": "hard_skill", "weight": 2.5, "frequency": 0.94},
                {"name": "TypeScript", "category": "hard_skill", "weight": 2.5, "frequency": 0.88},
                {"name": "HTML5", "category": "hard_skill", "weight": 2.0, "frequency": 0.90},
                {"name": "CSS3", "category": "hard_skill", "weight": 2.0, "frequency": 0.90},
                {"name": "Tailwind CSS", "category": "tool", "weight": 1.5, "frequency": 0.82},
                {"name": "Git", "category": "tool", "weight": 1.0, "frequency": 0.92},
                {
                    "name": "Adaptabilidad",
                    "category": "soft_skill",
                    "weight": 1.0,
                    "frequency": 0.85,
                },
                {
                    "name": "Comunicación",
                    "category": "soft_skill",
                    "weight": 1.0,
                    "frequency": 0.88,
                },
                {"name": "Scrum", "category": "methodology", "weight": 1.0, "frequency": 0.85},
            ],
        },
        {
            "name": "DevOps Cloud",
            "description": "Automatización, integración continua (CI/CD) e infraestructura como código sobre nubes líderes.",
            "job_offer_count": 85,
            "skills": [
                {"name": "Docker", "category": "tool", "weight": 3.0, "frequency": 0.94},
                {"name": "Kubernetes", "category": "tool", "weight": 3.0, "frequency": 0.88},
                {"name": "Terraform", "category": "tool", "weight": 3.0, "frequency": 0.80},
                {"name": "AWS", "category": "hard_skill", "weight": 2.5, "frequency": 0.85},
                {"name": "Linux", "category": "hard_skill", "weight": 2.5, "frequency": 0.78},
                {"name": "CI/CD", "category": "methodology", "weight": 2.5, "frequency": 0.90},
                {"name": "Prometheus", "category": "tool", "weight": 2.0, "frequency": 0.65},
                {"name": "Grafana", "category": "tool", "weight": 2.0, "frequency": 0.68},
                {"name": "Git", "category": "tool", "weight": 1.0, "frequency": 0.95},
                {
                    "name": "Resolución de problemas",
                    "category": "soft_skill",
                    "weight": 1.0,
                    "frequency": 0.92,
                },
                {"name": "Agile", "category": "methodology", "weight": 1.0, "frequency": 0.80},
            ],
        },
        {
            "name": "Data Engineering",
            "description": "Procesamiento de datos masivos (Big Data), diseño de pipelines ETL y bodegas analíticas en la nube.",
            "job_offer_count": 95,
            "skills": [
                {"name": "Python", "category": "hard_skill", "weight": 3.0, "frequency": 0.92},
                {"name": "Spark", "category": "hard_skill", "weight": 3.0, "frequency": 0.88},
                {"name": "SQL", "category": "hard_skill", "weight": 2.5, "frequency": 0.90},
                {"name": "NoSQL", "category": "hard_skill", "weight": 2.0, "frequency": 0.70},
                {"name": "Kafka", "category": "tool", "weight": 2.5, "frequency": 0.75},
                {"name": "Airflow", "category": "tool", "weight": 2.5, "frequency": 0.80},
                {"name": "Snowflake", "category": "tool", "weight": 2.5, "frequency": 0.65},
                {"name": "Docker", "category": "tool", "weight": 2.0, "frequency": 0.60},
                {"name": "Git", "category": "tool", "weight": 1.0, "frequency": 0.85},
                {
                    "name": "Pensamiento crítico",
                    "category": "soft_skill",
                    "weight": 1.0,
                    "frequency": 0.88,
                },
                {"name": "Scrum", "category": "methodology", "weight": 1.0, "frequency": 0.80},
            ],
        },
    ]

    try:
        from sqlalchemy import delete, select

        from src.ml_engine.infrastructure.models import DiagnosticModel, DiagnosticSkillModel

        # Clean diagnostics to avoid foreign key constraint violations
        print("Cleaning existing diagnostics and diagnostic skills...")
        await session.execute(delete(DiagnosticSkillModel))
        await session.execute(delete(DiagnosticModel))
        await session.flush()

        # Clean existing clusters and cluster_skills (cascade will delete junction entries)
        print("Cleaning existing clusters...")
        await session.execute(delete(ClusterModel))
        await session.flush()

        # Cache existing skills to avoid duplicates and update their weights
        existing_skills_res = await session.execute(select(SkillModel))
        skill_cache = {s.name.lower(): s for s in existing_skills_res.scalars().all()}

        for cdata in clusters_data:
            # Create cluster
            cluster = ClusterModel(
                cluster_id=uuid4(),
                name=cdata["name"],
                description=cdata["description"],
                job_offer_count=cdata["job_offer_count"],
                centroid_vec=make_mock_vector(),
            )
            session.add(cluster)
            await session.flush()

            # Insert/load skills and link to cluster
            for sdata in cdata["skills"]:
                sname_lower = sdata["name"].lower()
                if sname_lower in skill_cache:
                    skill = skill_cache[sname_lower]
                    # Update category and weight to the latest definition
                    skill.category = sdata["category"]
                    skill.weight = sdata["weight"]
                else:
                    skill = SkillModel(
                        skill_id=uuid4(),
                        name=sdata["name"],
                        category=sdata["category"],
                        weight=sdata["weight"],
                    )
                    session.add(skill)
                    await session.flush()
                    skill_cache[sname_lower] = skill

                # Link skill to cluster with frequency
                cluster_skill = ClusterSkillModel(
                    cluster_skill_id=uuid4(),
                    cluster_id=cluster.cluster_id,
                    skill_id=skill.skill_id,
                    importance_score=sdata["frequency"],
                )
                session.add(cluster_skill)

        await session.commit()
        print("Database seeded successfully with 5 clusters, weights, and frequencies!")
    except Exception as e:
        await session.rollback()
        print(f"Error seeding database: {e}")
        raise e
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(seed())
