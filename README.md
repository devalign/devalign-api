# Devalign API

> ML-powered API for tech skills gap analysis and personalized learning roadmap generation for Peruvian developers.

## Overview

Devalign analyzes job market demand (via scraping) and individual developer profiles (via CV analysis) to:

1. Identify a developer's primary technical specialty using ML clustering (K-Modes)
2. Detect skill gaps against the market standard for that specialty
3. Generate a personalized, standards-backed learning roadmap via LLM (Groq/OpenAI)

## Quick Start

```bash
# Requires: Python 3.12+ and uv
pip install uv

# Install dependencies
make install

# Setup environment
cp .env.example .env
# Edit .env with your Supabase, Groq/OpenAI, and Voyage AI credentials

# Start development server
make dev
# → http://localhost:8000/api/v1/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/users/me` | Current user profile |
| POST | `/api/v1/users/me/cv` | Upload CV |
| GET | `/api/v1/users/me/cvs` | List user CVs |
| POST | `/api/v1/users/me/cvs/{cv_id}/reanalyze` | Re-analyze a CV |
| DELETE | `/api/v1/users/me/cvs/{cv_id}` | Delete a CV |
| POST | `/api/v1/profile/analyze` | Analyze CV → generate profile |
| GET | `/api/v1/profile/me` | Get my analyzed profile |
| PATCH | `/api/v1/profile/me` | Update my profile |
| PUT | `/api/v1/profile/skills` | Update my skills |
| GET | `/api/v1/profile/clusters` | List tech specialties |
| POST | `/api/v1/profile/normalize-skills` | Normalize skill names |
| GET | `/api/v1/scraper/status` | Scraper pipeline status |

## Architecture

```
src/
├── delivery/       # Auth, users, CV upload & management
├── ml_engine/      # CV parsing, embeddings (Voyage/OpenAI), clustering (K-Modes), LLM analysis
├── scraper/        # Job offer acquisition (stub — external repo)
└── shared/         # Database, logging, middleware, security
```

Clean Architecture pattern: `domain → application → infrastructure → interface`

## Tech Stack

- **FastAPI** + Pydantic v2 + pydantic-settings
- **Supabase** (PostgreSQL + pgvector + Auth + Storage)
- **Voyage AI / OpenAI** (embeddings via HTTP API)
- **Groq / OpenAI** (LLM via HTTP API — no LangChain)
- **scikit-learn** + **kmodes** (K-Modes clustering)
- **pypdf** + **python-docx** (CV parsing)
- **SQLAlchemy 2.0** (async) + **asyncpg** + **pgvector**
- **structlog** (structured logging)
- **tenacity** (retry logic)
- **python-jose** (JWT decoding)
- **uv** + Ruff + mypy + pytest

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Stack](STACK.md)
- [Decisions](DECISIONS.md)
- [Progress](PROGRESS.md)
