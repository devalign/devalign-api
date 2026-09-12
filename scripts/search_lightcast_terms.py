"""Search Lightcast skills for specific translations and candidate concepts."""

import asyncio
from sqlalchemy import text
from src.shared.database import AsyncSessionLocal


async def main() -> None:
    async with AsyncSessionLocal() as session:
        search_terms = [
            "configuration management",
            "relational database",
            "cybersecurity",
            "quality assurance",
            "data visualization",
            "mobile development",
            "data model",
            "open source",
            "load test",
            "performance test",
            "api gateway",
            "cloud native",
            "excel",
            "sharepoint",
            "soap",
            "sftp",
            "agile",
            "documentation",
            "power bi",
            "postman",
            "jira",
            "scrum",
            "office",
            "jwt",
            "json",
            "kafka",
            "tdd",
            "owasp",
            "airflow",
            "jmeter",
            "kubernetes",
            "cloudformation",
            "aws lambda",
            "azure",
            "css",
            "html",
        ]
        for term in search_terms:
            res = await session.execute(
                text("""
                SELECT s.skill_id, s.name, ss.standard_code
                FROM skills s
                JOIN skill_standards ss ON ss.skill_id = s.skill_id
                WHERE LOWER(s.name) LIKE :term
                ORDER BY s.name ASC
                LIMIT 6;
            """),
                {"term": f"%{term}%"},
            )
            matches = res.fetchall()
            print(f"TERM: '{term}' -> {[m[1] for m in matches]}")


if __name__ == "__main__":
    asyncio.run(main())
