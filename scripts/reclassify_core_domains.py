"""
Reclassify and standardize core_domains and domain_tags across all skills.

Taxonomy:
- Frontend
- Backend
- DevOps
- Data
- QA
- Security
- Mobile
- Architecture
- Management
- Engineering (clean fallback for general software/systems)

Ensures:
- All values are TitleCase.
- Index 0 is the primary domain (e.g. JavaScript -> ['Frontend', 'Backend']).
- No empty core_domains.
- software_engineering is replaced with precise domain or Engineering.
"""

import asyncio
import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sqlalchemy import select, update, bindparam
from src.shared.database import AsyncSessionLocal
from src.ml_engine.infrastructure.models import SkillModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Heuristics for domain classification
FRONTEND_KEYWORDS = [
    "react", "angular", "vue", "svelte", "frontend", "front-end", "html", "css", "scss", "sass",
    "tailwind", "bootstrap", "ui", "ux", "javascript", "typescript", "dom", "canvas", "svg",
    "webpack", "vite", "next.js", "nuxt", "redux", "zustand", "remix", "styled-components",
    "chakra", "shadcn", "webgl", "three.js", "electron", "responsive", "web design", "figma",
    "adobe xd", "sketch", "user interface", "user experience", "astro", "jquery", "ajax",
    "material-ui", "ant design", "storybook", "after effects", "photoshop", "illustrator",
    "graphic design", "web accessibility", "a11y", "client-side"
]

BACKEND_KEYWORDS = [
    "backend", "back-end", "api", "rest", "graphql", "grpc", "soap", "websocket", "microservice",
    "spring", "django", "fastapi", "flask", "express", "nestjs", "rails", "laravel", "asp.net",
    ".net", "node.js", "nodejs", "python", "golang", "go (programming", "java (", "c#", "c++",
    "php", "ruby", "rust", "scala", "elixir", "clojure", "perl", "r (programming",
    "celery", "rabbitmq", "apache kafka", "zeromq", "serverless", "socket.io", "middleware",
    "rpc", "orm", "hibernate", "prisma", "sqlalchemy", "entity framework", "activemq", "pulsar",
    "apache thrift", "apache camel", "apache cxf", "apache openjpa", "message broker", "messaging"
]

MOBILE_KEYWORDS = [
    "mobile", "ios", "android", "react native", "flutter", "swift", "kotlin", "dart",
    "objective-c", "expo", "xcode", "android studio", "swiftui", "jetpack compose",
    "cross-platform mobile", "mobile app", "ionic", "cordova", "xamarin", "app store", "google play"
]

DATA_KEYWORDS = [
    "database", "sql", "postgres", "mysql", "mongodb", "redis", "cassandra", "dynamodb",
    "elasticsearch", "sqlite", "oracle", "mariadb", "neo4j", "couchdb", "data", "etl",
    "machine learning", "deep learning", "ai", "artificial intelligence", "bigquery", "snowflake",
    "spark", "hadoop", "airflow", "dbt", "databricks", "pandas", "numpy", "pytorch", "tensorflow",
    "scikit-learn", "keras", "computer vision", "nlp", "natural language", "llm", "large language",
    "generative ai", "neural network", "analytics", "bi", "business intelligence", "tableau",
    "power bi", "looker", "data warehouse", "data lake", "data pipeline", "vector database",
    "pinecone", "milvus", "chromadb", "data science", "statistics", "data engineering", "mining",
    "apache beam", "apache derby", "apache drill", "apache lucene", "apache parquet", "apache pig",
    "apache nifi", "apache solr", "apache flink", "apache hive", "apache avro", "mapreduce",
    "olap", "oltp", "kudu", "presto", "trino", "apache mxnet", "apache mahout"
]

DEVOPS_KEYWORDS = [
    "cloud", "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
    "kubernetes", "k8s", "devops", "ci/cd", "continuous integration", "continuous delivery",
    "continuous deployment", "terraform", "ansible", "jenkins", "gitlab ci", "github actions",
    "argo", "helm", "linux", "unix", "bash", "shell", "powershell", "nginx", "apache http",
    "traefik", "prometheus", "grafana", "datadog", "new relic", "splunk", "elk stack",
    "infrastructure as code", "serverless framework", "sre", "site reliability", "networking",
    "dns", "tcp/ip", "load balancing", "vpn", "ssh", "ssl/tls", "open telemetry", "opentelemetry",
    "subversion", "svn", "apache ant", "apache maven", "maven", "gradle", "apache mesos", "apache ambari",
    "network infrastructure", "system administration", "sysadmin"
]

QA_KEYWORDS = [
    "qa", "quality assurance", "test", "testing", "selenium", "cypress", "playwright", "jest",
    "junit", "pytest", "mocha", "chai", "cucumber", "postman", "jmeter", "k6", "tdd", "bdd",
    "test automation", "unit test", "integration test", "end-to-end", "e2e", "load test",
    "stress test", "manual testing", "qa automation", "performance test", "sonarqube"
]

SECURITY_KEYWORDS = [
    "security", "cyber", "cybersecurity", "oauth", "openid", "jwt", "keycloak", "cryptography",
    "penetration test", "pen test", "ethical hacking", "vulnerability", "owasp", "firewall",
    "siem", "soc", "iam", "identity access", "encryption", "auth0", "zero trust", "infosec",
    "malware", "incident response", "compliance", "iso 27001", "gdpr", "hipaa", "pci-dss",
    "antivirus", "comptia"
]

ARCHITECTURE_KEYWORDS = [
    "architecture", "design patterns", "microservices", "clean architecture", "domain-driven",
    "ddd", "solid", "dry", "system design", "event-driven", "cqrs", "hexagonal", "software design",
    "enterprise architecture", "solution architecture", "software architecture", "scalability",
    "high availability", "distributed systems", "cloud architecture"
]

MANAGEMENT_KEYWORDS = [
    "agile", "scrum", "kanban", "jira", "confluence", "trello", "asana", "project management",
    "product management", "product owner", "scrum master", "sprint", "sdlc", "waterfall",
    "lean", "okr", "kpi", "team leadership", "mentoring", "technical writing", "code review",
    "documentation", "git", "github", "gitlab", "bitbucket", "version control", "sbok",
    "pmbok", "pmp", "engineering management"
]

# Primary priority overrides for specific multi-domain skills
EXPLICIT_PRIMARY = {
    # Frontend primary
    "javascript (programming language)": ["Frontend", "Backend"],
    "typescript": ["Frontend", "Backend"],
    "javascript": ["Frontend", "Backend"],
    "html": ["Frontend"],
    "css": ["Frontend"],
    "react.js": ["Frontend"],
    "react (web framework)": ["Frontend"],
    "vue.js": ["Frontend"],
    "angular (web framework)": ["Frontend"],
    "svelte": ["Frontend"],
    "next.js": ["Frontend", "Backend"],
    "nuxt.js": ["Frontend"],
    "tailwind css": ["Frontend"],
    "web development": ["Frontend", "Backend"],

    # Mobile primary
    "react native": ["Mobile", "Frontend"],
    "flutter (software)": ["Mobile", "Frontend"],
    "flutter": ["Mobile", "Frontend"],
    "swift (programming language)": ["Mobile"],
    "kotlin (programming language)": ["Mobile", "Backend"],
    "dart (programming language)": ["Mobile", "Frontend"],
    "android (operating system)": ["Mobile"],
    "ios": ["Mobile"],

    # Backend primary
    "python (programming language)": ["Backend", "Data"],
    "go (programming language)": ["Backend", "DevOps"],
    "java (programming language)": ["Backend"],
    "c# (programming language)": ["Backend"],
    "php (programming language)": ["Backend"],
    "ruby (programming language)": ["Backend"],
    "c++ (programming language)": ["Backend", "Engineering"],
    "c (programming language)": ["Backend", "Engineering"],
    "rust (programming language)": ["Backend", "Engineering"],
    "node.js": ["Backend", "Frontend"],
    "spring framework": ["Backend"],
    "spring boot": ["Backend"],
    "fastapi": ["Backend"],
    "django (web framework)": ["Backend"],
    "express.js": ["Backend"],
    "graphql (technique)": ["Backend", "Frontend"],
    "restful api": ["Backend"],
    "application programming interface (api)": ["Backend"],

    # DevOps primary
    "docker (software)": ["DevOps"],
    "kubernetes": ["DevOps"],
    "terraform (software)": ["DevOps"],
    "amazon web services": ["DevOps", "Backend"],
    "microsoft azure": ["DevOps", "Backend"],
    "google cloud platform": ["DevOps", "Backend"],
    "git": ["DevOps", "Management"],
    "linux": ["DevOps"],
    "jenkins (software)": ["DevOps"],
    "ci/cd": ["DevOps"],

    # Data primary
    "postgresql": ["Data", "Backend"],
    "mysql": ["Data", "Backend"],
    "mongodb": ["Data", "Backend"],
    "redis": ["Data", "Backend"],
    "apache kafka": ["Data", "Backend", "DevOps"],
    "apache spark": ["Data"],
    "sql (programming language)": ["Data"],
    "machine learning": ["Data"],
    "artificial intelligence": ["Data"],
    "deep learning": ["Data"],

    # QA primary
    "cypress (software)": ["QA", "Frontend"],
    "selenium (software)": ["QA"],
    "jest (javascript testing framework)": ["QA", "Frontend"],
    "junit": ["QA", "Backend"],
    "pytest": ["QA", "Backend"],

    # Architecture primary
    "microservices": ["Architecture", "Backend"],
    "software architecture": ["Architecture"],
    "system design": ["Architecture"],
    "design patterns": ["Architecture"],
    "domain-driven design (ddd)": ["Architecture"],
    "solid": ["Architecture"],
    "clean architecture": ["Architecture"],

    # Management primary
    "agile methodology": ["Management"],
    "scrum (software development)": ["Management"],
    "kanban (software development)": ["Management"],
    "jira (software)": ["Management"],
    "code review": ["Management", "Engineering"],
    "software documentation": ["Management", "Engineering"]
}


def classify_skill(name: str, current_domains: list[str]) -> tuple[list[str], list[str]]:
    """Determine prioritized core_domains and domain_tags for a skill."""
    nl = name.strip().lower()

    # 1. Check explicit primary overrides
    if nl in EXPLICIT_PRIMARY:
        domains = list(EXPLICIT_PRIMARY[nl])
        tags = [f"{d} Development" if d in ["Frontend", "Backend"] else d for d in domains]
        return domains, tags

    detected = set()

    # Heuristic matchers
    def matches_any(keywords):
        return any(kw in nl for kw in keywords)

    # Specific domains
    if matches_any(MOBILE_KEYWORDS):
        detected.add("Mobile")
    if matches_any(FRONTEND_KEYWORDS):
        detected.add("Frontend")
    if matches_any(BACKEND_KEYWORDS):
        detected.add("Backend")
    if matches_any(DATA_KEYWORDS):
        detected.add("Data")
    if matches_any(DEVOPS_KEYWORDS):
        detected.add("DevOps")
    if matches_any(QA_KEYWORDS):
        detected.add("QA")
    if matches_any(SECURITY_KEYWORDS):
        detected.add("Security")
    if matches_any(ARCHITECTURE_KEYWORDS):
        detected.add("Architecture")
    if matches_any(MANAGEMENT_KEYWORDS):
        detected.add("Management")

    # If current_domains has useful clues
    if not detected and current_domains:
        for cd in current_domains:
            cd_l = cd.lower()
            if cd_l in ["frontend", "front-end"]:
                detected.add("Frontend")
            elif cd_l in ["backend", "back-end"]:
                detected.add("Backend")
            elif cd_l in ["cloud_devops", "devops", "cloud"]:
                detected.add("DevOps")
            elif cd_l in ["data", "database", "ai/ml"]:
                detected.add("Data")
            elif cd_l in ["qa", "testing"]:
                detected.add("QA")
            elif cd_l in ["security", "cybersecurity"]:
                detected.add("Security")
            elif cd_l in ["mobile"]:
                detected.add("Mobile")

    # Fallback for general skills
    if not detected:
        detected.add("Engineering")

    # Priority ordering rule:
    # If Mobile and Frontend: Mobile is first
    ordered = []
    priority_order = ["Mobile", "QA", "Security", "DevOps", "Data", "Frontend", "Backend", "Architecture", "Management", "Engineering"]

    # If both Frontend and Backend are present:
    if "Frontend" in detected and "Backend" in detected:
        # Check which keyword came first or is more prominent
        frontend_hit = any(kw in nl for kw in ["react", "angular", "vue", "html", "css", "ui", "ux", "front"])
        backend_hit = any(kw in nl for kw in ["spring", "django", "fastapi", "flask", "express", "sql", "back", "api"])
        if frontend_hit and not backend_hit:
            ordered.append("Frontend")
            ordered.append("Backend")
        else:
            ordered.append("Backend")
            ordered.append("Frontend")
        for d in priority_order:
            if d in detected and d not in ordered:
                ordered.append(d)
    else:
        for d in priority_order:
            if d in detected:
                ordered.append(d)

    tags = [f"{d} Development" if d in ["Frontend", "Backend"] else d for d in ordered]
    return ordered, tags


async def run_reclassification(dry_run: bool = False):
    """Execute core_domains reclassification."""
    logger.info(f"Starting core_domains reclassification (dry_run={dry_run})...")

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(SkillModel))
        skills = result.scalars().all()
        logger.info(f"Total skills to evaluate: {len(skills)}")

        updated_count = 0
        batch_payload = []
        domain_distribution = {}
        for skill in skills:
            curr_domains = skill.core_domains or []
            new_domains, new_tags = classify_skill(skill.name, curr_domains)

            primary = new_domains[0]
            domain_distribution[primary] = domain_distribution.get(primary, 0) + 1

            if curr_domains != new_domains or skill.domain_tags != new_tags:
                updated_count += 1
                batch_payload.append({
                    "b_id": skill.skill_id,
                    "b_domains": new_domains,
                    "b_tags": new_tags
                })

        if not dry_run and batch_payload:
            stmt = (
                SkillModel.__table__.update()
                .where(SkillModel.__table__.c.skill_id == bindparam("b_id"))
                .values(core_domains=bindparam("b_domains"), domain_tags=bindparam("b_tags"))
            )
            for i in range(0, len(batch_payload), 250):
                chunk = batch_payload[i:i+250]
                await session.execute(stmt, chunk)
            await session.commit()
            logger.info(f"Successfully committed {len(batch_payload)} updated core_domains and domain_tags!")
        else:
            await session.rollback()
            logger.info(f"Dry-run complete (rolled back {len(batch_payload)} updates).")

        logger.info(f"Total skills modified: {updated_count}/{len(skills)}")
        logger.info("Primary Domain Distribution:")
        for dom, cnt in sorted(domain_distribution.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {dom}: {cnt} skills")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_reclassification(dry_run=args.dry_run))
