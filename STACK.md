# Devalign API — Stack

## Runtime

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.12+ |
| Framework | FastAPI | 0.115+ |
| Server | Uvicorn (ASGI) | 0.30+ |
| Validation | Pydantic v2 | 2.9+ |

## Data & Persistence

| Component | Technology | Notes |
|-----------|-----------|-------|
| Database | Supabase PostgreSQL | Managed |
| ORM | SQLAlchemy 2.0 (async) | asyncpg driver |
| Migrations | Alembic | Autogenerate support |
| Vector Store | pgvector | Built into Supabase |
| Storage | Supabase Storage | CVs, documents |

## ML & AI

| Component | Technology | Notes |
|-----------|-----------|-------|
| ML Core | Scikit-learn | Pipelines, metrics |
| Clustering | kmodes | K-Prototypes |
| Embeddings (dev) | sentence-transformers | Local, free |
| Embeddings (prod) | OpenAI text-embedding-3-small | Optional upgrade |
| LLM (dev) | Groq llama-3.1-70b | Fast, free tier |
| LLM (prod) | OpenAI gpt-4o-mini | Higher quality |
| LLM Orchestration | LangChain | RAG chains |
| Document Parsing | pypdf + python-docx | PDF, DOCX |

## DevOps & Quality

| Component | Technology | Notes |
|-----------|-----------|-------|
| Package Manager | uv | 10-100x faster than pip |
| Linter/Formatter | Ruff | Replaces flake8+black+isort |
| Type Checker | mypy (strict) | Enforced in CI |
| Testing | pytest + pytest-asyncio | Async test support |
| Pre-commit | pre-commit | Ruff + mypy hooks |
| CI/CD | GitHub Actions | Quality + test pipeline |
| Containerization | Docker (multi-stage) | uv-based build |
| Deployment | Koyeb (free tier) | Primary target |
