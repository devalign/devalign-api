"""Development seed script for Devalign tech clusters and skills with weights, frequencies, domain_tags, and core_domains."""

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


def get_nature_from_category(category: str) -> str:
    if category in ("hard_skill", "tool"):
        return "tech"
    elif category == "soft_skill":
        return "soft"
    elif category == "methodology":
        return "concept"
    return "tech"


async def seed() -> None:
    print("Starting development database seeding with weights, frequencies, domain_tags, and core_domains...")
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
                {"name": "Java", "category": "hard_skill", "weight": 3.0, "frequency": 0.92, "domain_tags": ["java", "backend"], "core_domains": ["Backend"]},
                {"name": "Spring Boot", "category": "hard_skill", "weight": 3.0, "frequency": 0.88, "domain_tags": ["spring", "backend", "java"], "core_domains": ["Backend"]},
                {"name": "REST APIs", "category": "hard_skill", "weight": 2.0, "frequency": 0.80, "domain_tags": ["api", "rest"], "core_domains": ["Backend"]},
                {"name": "Microservicios", "category": "hard_skill", "weight": 2.0, "frequency": 0.82, "domain_tags": ["microservices", "architecture"], "core_domains": ["Backend"]},
                {"name": "PostgreSQL", "category": "hard_skill", "weight": 2.0, "frequency": 0.74, "domain_tags": ["database", "relational", "postgresql"], "core_domains": ["Data", "Backend"]},
                {"name": "AWS", "category": "hard_skill", "weight": 2.0, "frequency": 0.62, "domain_tags": ["cloud", "aws"], "core_domains": ["Cloud", "DevOps"]},
                {"name": "Docker", "category": "tool", "weight": 2.0, "frequency": 0.70, "domain_tags": ["containers", "docker", "devops"], "core_domains": ["DevOps"]},
                {"name": "Kubernetes", "category": "tool", "weight": 2.0, "frequency": 0.55, "domain_tags": ["devops", "cloud", "kubernetes", "orchestration"], "core_domains": ["DevOps", "Cloud"]},
                {"name": "Git", "category": "tool", "weight": 1.0, "frequency": 0.90, "domain_tags": ["git", "vcs"], "core_domains": ["DevOps"]},
                {"name": "Liderazgo", "category": "soft_skill", "weight": 1.0, "frequency": 0.85, "domain_tags": ["leadership"], "core_domains": []},
                {"name": "Comunicación", "category": "soft_skill", "weight": 1.0, "frequency": 0.80, "domain_tags": ["communication"], "core_domains": []},
                {"name": "Scrum", "category": "methodology", "weight": 1.0, "frequency": 0.82, "domain_tags": ["scrum", "agile", "methodology"], "core_domains": []},
            ],
        },
        {
            "name": "Backend Python",
            "description": "Desarrollo ágil de microservicios y APIs utilizando Python, FastAPI y bases de datos modernas.",
            "job_offer_count": 110,
            "skills": [
                {"name": "Python", "category": "hard_skill", "weight": 3.0, "frequency": 0.95, "domain_tags": ["python", "backend"], "core_domains": ["Backend", "Data"]},
                {"name": "FastAPI", "category": "hard_skill", "weight": 3.0, "frequency": 0.82, "domain_tags": ["fastapi", "backend", "python", "api"], "core_domains": ["Backend"]},
                {"name": "PostgreSQL", "category": "hard_skill", "weight": 2.0, "frequency": 0.78, "domain_tags": ["database", "relational", "postgresql"], "core_domains": ["Data", "Backend"]},
                {"name": "MongoDB", "category": "hard_skill", "weight": 2.0, "frequency": 0.60, "domain_tags": ["database", "nosql", "mongodb"], "core_domains": ["Data", "Backend"]},
                {"name": "REST APIs", "category": "hard_skill", "weight": 2.0, "frequency": 0.85, "domain_tags": ["api", "rest"], "core_domains": ["Backend"]},
                {"name": "AWS", "category": "hard_skill", "weight": 2.0, "frequency": 0.68, "domain_tags": ["cloud", "aws"], "core_domains": ["Cloud", "DevOps"]},
                {"name": "Docker", "category": "tool", "weight": 2.0, "frequency": 0.75, "domain_tags": ["containers", "docker", "devops"], "core_domains": ["DevOps"]},
                {"name": "Redis", "category": "hard_skill", "weight": 1.5, "frequency": 0.55, "domain_tags": ["database", "cache", "redis"], "core_domains": ["Backend"]},
                {"name": "Git", "category": "tool", "weight": 1.0, "frequency": 0.92, "domain_tags": ["git", "vcs"], "core_domains": ["DevOps"]},
                {"name": "Trabajo en equipo", "category": "soft_skill", "weight": 1.0, "frequency": 0.88, "domain_tags": ["teamwork"], "core_domains": []},
                {"name": "Resolución de problemas", "category": "soft_skill", "weight": 1.0, "frequency": 0.90, "domain_tags": ["problem-solving"], "core_domains": []},
                {"name": "Scrum", "category": "methodology", "weight": 1.0, "frequency": 0.80, "domain_tags": ["scrum", "agile", "methodology"], "core_domains": []},
            ],
        },
        {
            "name": "Frontend React",
            "description": "Construcción de interfaces de usuario interactivas y SPA modernas usando React, Next.js y CSS moderno.",
            "job_offer_count": 190,
            "skills": [
                {"name": "React", "category": "hard_skill", "weight": 3.0, "frequency": 0.96, "domain_tags": ["react", "frontend", "javascript", "spa"], "core_domains": ["Frontend"]},
                {"name": "Next.js", "category": "hard_skill", "weight": 3.0, "frequency": 0.85, "domain_tags": ["nextjs", "frontend", "react", "framework", "web"], "core_domains": ["Frontend"]},
                {"name": "JavaScript", "category": "hard_skill", "weight": 2.5, "frequency": 0.94, "domain_tags": ["javascript", "frontend", "web"], "core_domains": ["Frontend"]},
                {"name": "TypeScript", "category": "hard_skill", "weight": 2.5, "frequency": 0.88, "domain_tags": ["typescript", "frontend", "programming-language"], "core_domains": ["Frontend"]},
                {"name": "HTML5", "category": "hard_skill", "weight": 2.0, "frequency": 0.90, "domain_tags": ["html", "frontend", "web"], "core_domains": ["Frontend"]},
                {"name": "CSS3", "category": "hard_skill", "weight": 2.0, "frequency": 0.90, "domain_tags": ["css", "frontend", "web"], "core_domains": ["Frontend"]},
                {"name": "Tailwind CSS", "category": "tool", "weight": 1.5, "frequency": 0.82, "domain_tags": ["tailwindcss", "css", "frontend", "design"], "core_domains": ["Frontend"]},
                {"name": "Git", "category": "tool", "weight": 1.0, "frequency": 0.92, "domain_tags": ["git", "vcs"], "core_domains": ["DevOps"]},
                {"name": "Adaptabilidad", "category": "soft_skill", "weight": 1.0, "frequency": 0.85, "domain_tags": ["adaptability"], "core_domains": []},
                {"name": "Comunicación", "category": "soft_skill", "weight": 1.0, "frequency": 0.88, "domain_tags": ["communication"], "core_domains": []},
                {"name": "Scrum", "category": "methodology", "weight": 1.0, "frequency": 0.85, "domain_tags": ["scrum", "agile", "methodology"], "core_domains": []},
            ],
        },
        {
            "name": "DevOps Cloud",
            "description": "Automatización, integración continua (CI/CD) e infraestructura como código sobre nubes líderes.",
            "job_offer_count": 85,
            "skills": [
                {"name": "Docker", "category": "tool", "weight": 3.0, "frequency": 0.94, "domain_tags": ["containers", "docker", "devops"], "core_domains": ["DevOps"]},
                {"name": "Kubernetes", "category": "tool", "weight": 3.0, "frequency": 0.88, "domain_tags": ["devops", "cloud", "kubernetes", "orchestration"], "core_domains": ["DevOps", "Cloud"]},
                {"name": "Terraform", "category": "tool", "weight": 3.0, "frequency": 0.80, "domain_tags": ["terraform", "devops", "cloud", "iac"], "core_domains": ["DevOps", "Cloud"]},
                {"name": "AWS", "category": "hard_skill", "weight": 2.5, "frequency": 0.85, "domain_tags": ["cloud", "aws"], "core_domains": ["Cloud", "DevOps"]},
                {"name": "Linux", "category": "hard_skill", "weight": 2.5, "frequency": 0.78, "domain_tags": ["linux", "os"], "core_domains": ["DevOps", "Backend"]},
                {"name": "CI/CD", "category": "methodology", "weight": 2.5, "frequency": 0.90, "domain_tags": ["cicd", "devops", "automation"], "core_domains": ["DevOps"]},
                {"name": "Prometheus", "category": "tool", "weight": 2.0, "frequency": 0.65, "domain_tags": ["prometheus", "monitoring", "devops"], "core_domains": ["DevOps"]},
                {"name": "Grafana", "category": "tool", "weight": 2.0, "frequency": 0.68, "domain_tags": ["grafana", "monitoring", "visualization", "devops"], "core_domains": ["DevOps"]},
                {"name": "Git", "category": "tool", "weight": 1.0, "frequency": 0.95, "domain_tags": ["git", "vcs"], "core_domains": ["DevOps"]},
                {"name": "Resolución de problemas", "category": "soft_skill", "weight": 1.0, "frequency": 0.92, "domain_tags": ["problem-solving"], "core_domains": []},
                {"name": "Agile", "category": "methodology", "weight": 1.0, "frequency": 0.80, "domain_tags": ["agile", "methodology"], "core_domains": []},
            ],
        },
        {
            "name": "Data Engineering",
            "description": "Procesamiento de datos masivos (Big Data), diseño de pipelines ETL y bodegas analíticas en la nube.",
            "job_offer_count": 95,
            "skills": [
                {"name": "Python", "category": "hard_skill", "weight": 3.0, "frequency": 0.92, "domain_tags": ["python", "backend"], "core_domains": ["Backend", "Data"]},
                {"name": "Spark", "category": "hard_skill", "weight": 3.0, "frequency": 0.88, "domain_tags": ["spark", "bigdata", "data-engineering"], "core_domains": ["Data"]},
                {"name": "SQL", "category": "hard_skill", "weight": 2.5, "frequency": 0.90, "domain_tags": ["sql", "database", "relational"], "core_domains": ["Data", "Backend"]},
                {"name": "NoSQL", "category": "hard_skill", "weight": 2.0, "frequency": 0.70, "domain_tags": ["database", "nosql"], "core_domains": ["Data"]},
                {"name": "Kafka", "category": "tool", "weight": 2.5, "frequency": 0.75, "domain_tags": ["kafka", "streaming", "pubsub", "messaging"], "core_domains": ["Data", "Backend"]},
                {"name": "Airflow", "category": "tool", "weight": 2.5, "frequency": 0.80, "domain_tags": ["airflow", "orchestration", "data-engineering", "workflow"], "core_domains": ["Data", "DevOps"]},
                {"name": "Snowflake", "category": "tool", "weight": 2.5, "frequency": 0.65, "domain_tags": ["snowflake", "cloud", "data-warehouse"], "core_domains": ["Data", "Cloud"]},
                {"name": "Docker", "category": "tool", "weight": 2.0, "frequency": 0.60, "domain_tags": ["containers", "docker", "devops"], "core_domains": ["DevOps"]},
                {"name": "Git", "category": "tool", "weight": 1.0, "frequency": 0.85, "domain_tags": ["git", "vcs"], "core_domains": ["DevOps"]},
                {"name": "Pensamiento crítico", "category": "soft_skill", "weight": 1.0, "frequency": 0.88, "domain_tags": ["critical-thinking"], "core_domains": []},
                {"name": "Scrum", "category": "methodology", "weight": 1.0, "frequency": 0.80, "domain_tags": ["scrum", "agile", "methodology"], "core_domains": []},
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
                nature_val = get_nature_from_category(sdata["category"])
                
                if sname_lower in skill_cache:
                    skill = skill_cache[sname_lower]
                    # Update properties
                    skill.nature = nature_val
                    skill.weight = sdata["weight"]
                    skill.domain_tags = sdata["domain_tags"]
                    skill.core_domains = sdata["core_domains"]
                else:
                    skill = SkillModel(
                        skill_id=uuid4(),
                        name=sdata["name"],
                        nature=nature_val,
                        weight=sdata["weight"],
                        domain_tags=sdata["domain_tags"],
                        core_domains=sdata["core_domains"],
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
        print("Database seeded successfully with 5 clusters, weights, frequencies, and core domains!")
    except Exception as e:
        await session.rollback()
        print(f"Error seeding database: {e}")
        raise e
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(seed())
