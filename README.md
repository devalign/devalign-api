# Devalign API

> ML-powered API for tech skills gap analysis and personalized learning roadmap generation for Peruvian developers.

## Overview

Devalign analyzes job market demand (via scraping) and individual developer profiles (via CV analysis) to:

1. Identify a developer's primary technical specialty using ML clustering
2. Detect skill gaps against the market standard for that specialty
3. Generate a personalized, standards-backed learning roadmap using RAG + LLM

## Quick Start

```bash
# Requires: Python 3.12+ and uv
pip install uv

# Install dependencies
make install

# Setup environment
cp .env.example .env
# Edit .env with your Supabase and Groq credentials

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
| POST | `/api/v1/profile/analyze` | Analyze CV → generate profile |
| GET | `/api/v1/profile/clusters` | List tech specialties |
| POST | `/api/v1/roadmap/generate` | Generate personalized roadmap |
| GET | `/api/v1/scraper/status` | Scraper pipeline status |

## Architecture

```
src/
├── delivery/     # Auth, users, CV upload
├── ml_engine/    # CV parsing, embeddings, clustering, gap detection
├── genai/        # LangChain RAG pipeline, LLM roadmap generation
├── scraper/      # Job offer acquisition (stub — external repo)
└── shared/       # Database, logging, middleware, security
```

Clean Architecture pattern: `domain → application → infrastructure → interface`

## Tech Stack

- **FastAPI** + Pydantic v2
- **Supabase** (PostgreSQL + pgvector + Auth + Storage)
- **LangChain** (RAG pipeline, Groq/OpenAI LLM)
- **sentence-transformers** (local embeddings)
- **Scikit-learn** + kmodes (K-Prototypes clustering)
- **uv** + Ruff + mypy + pytest

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Stack](STACK.md)
- [Decisions](DECISIONS.md)
- [Progress](PROGRESS.md)
