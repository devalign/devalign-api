"""Development seed script for Devalign tech clusters and skills."""

import asyncio
import random
from uuid import uuid4
import sys
import os

# Append src/ to path so we can import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.database import AsyncSessionLocal
from src.ml_engine.infrastructure.models import (
    ClusterModel,
    ClusterSkillModel,
    SkillModel,
    EMBEDDING_DIM,
)


async def seed() -> None:
    print("Starting development database seeding...")
    session = AsyncSessionLocal()

    # Generate a random unit vector for centroids
    def make_mock_vector():
        vec = [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec]

    # Cluster definition list
    clusters_data = [
        {
            "name": "Backend",
            "description": "Desarrollo de lógica de servidor, APIs, bases de datos y arquitectura de microservicios.",
            "skills": [
                {"name": "Java", "category": "hard_skill"},
                {"name": "Spring Boot", "category": "hard_skill"},
                {"name": "REST APIs", "category": "hard_skill"},
                {"name": "Microservicios", "category": "hard_skill"},
                {"name": "Node.js", "category": "hard_skill"},
                {"name": "PostgreSQL", "category": "hard_skill"},
                {"name": "MongoDB", "category": "hard_skill"},
                {"name": "AWS", "category": "hard_skill"},
                {"name": "Docker", "category": "tool"},
                {"name": "Kubernetes", "category": "tool"},
                {"name": "CI/CD", "category": "methodology"},
            ],
        },
        {
            "name": "Cloud",
            "description": "Diseño y administración de infraestructura en la nube, escalabilidad y alta disponibilidad.",
            "skills": [
                {"name": "AWS", "category": "hard_skill"},
                {"name": "Azure", "category": "hard_skill"},
                {"name": "Google Cloud", "category": "hard_skill"},
                {"name": "Terraform", "category": "tool"},
                {"name": "Docker", "category": "tool"},
                {"name": "Kubernetes", "category": "tool"},
                {"name": "Linux", "category": "hard_skill"},
                {"name": "CI/CD", "category": "methodology"},
            ],
        },
        {
            "name": "DevOps",
            "description": "Automatización de despliegues, integración continua y observabilidad de sistemas.",
            "skills": [
                {"name": "Docker", "category": "tool"},
                {"name": "Kubernetes", "category": "tool"},
                {"name": "CI/CD", "category": "methodology"},
                {"name": "Terraform", "category": "tool"},
                {"name": "Linux", "category": "hard_skill"},
                {"name": "Prometheus", "category": "tool"},
                {"name": "Grafana", "category": "tool"},
                {"name": "AWS", "category": "hard_skill"},
            ],
        },
        {
            "name": "Frontend",
            "description": "Construcción de interfaces de usuario interactivas, accesibles y de alta fidelidad visual.",
            "skills": [
                {"name": "React", "category": "hard_skill"},
                {"name": "Next.js", "category": "hard_skill"},
                {"name": "HTML5", "category": "hard_skill"},
                {"name": "CSS3", "category": "hard_skill"},
                {"name": "JavaScript", "category": "hard_skill"},
                {"name": "TypeScript", "category": "hard_skill"},
                {"name": "Tailwind CSS", "category": "tool"},
                {"name": "Vue", "category": "hard_skill"},
            ],
        },
        {
            "name": "Data Engineering",
            "description": "Procesamiento de datos a gran escala, tuberías ETL/ELT y almacenamiento analítico.",
            "skills": [
                {"name": "Python", "category": "hard_skill"},
                {"name": "Spark", "category": "hard_skill"},
                {"name": "Hadoop", "category": "hard_skill"},
                {"name": "SQL", "category": "hard_skill"},
                {"name": "NoSQL", "category": "hard_skill"},
                {"name": "Kafka", "category": "tool"},
                {"name": "Airflow", "category": "tool"},
                {"name": "Snowflake", "category": "tool"},
            ],
        },
    ]

    try:
        # Cache existing skills to avoid duplicates
        from sqlalchemy import select
        existing_skills_res = await session.execute(select(SkillModel))
        skill_cache = {s.name.lower(): s for s in existing_skills_res.scalars().all()}

        for cdata in clusters_data:
            # Create cluster
            cluster = ClusterModel(
                cluster_id=uuid4(),
                name=cdata["name"],
                description=cdata["description"],
                centroid_vec=make_mock_vector(),
            )
            session.add(cluster)
            await session.flush()

            # Insert/load skills and link to cluster
            for sdata in cdata["skills"]:
                sname_lower = sdata["name"].lower()
                if sname_lower in skill_cache:
                    skill = skill_cache[sname_lower]
                else:
                    skill = SkillModel(
                        skill_id=uuid4(),
                        name=sdata["name"],
                        category=sdata["category"],
                    )
                    session.add(skill)
                    await session.flush()
                    skill_cache[sname_lower] = skill

                # Link skill to cluster
                cluster_skill = ClusterSkillModel(
                    cluster_skill_id=uuid4(),
                    cluster_id=cluster.cluster_id,
                    skill_id=skill.skill_id,
                    importance_score=0.9,
                )
                session.add(cluster_skill)

        await session.commit()
        print("Database seeded successfully with 5 clusters and centroid skills!")
    except Exception as e:
        await session.rollback()
        print(f"Error seeding database: {e}")
        raise e
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(seed())
