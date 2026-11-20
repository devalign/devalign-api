# devalign-api — Agent Rules

Backend service providing CV parsing, deterministic skill extraction, gap/affinity scoring, and roadmap persistence for Devalign.

## Stack & Architecture

- **Runtime:** Python 3.12+, managed with `uv` (`pyproject.toml`, `uv.lock`).
- **Frameworks:** FastAPI, Pydantic v2, pydantic-settings, SQLAlchemy Async (`asyncpg`), Alembic, `pgvector`, Supabase, `structlog`, `tenacity`.
- **Architecture:** Clean Architecture under `src/`. Keep domain entities, DTOs, ports, and use cases isolated from FastAPI routers, database repositories, and external HTTP/LLM clients.

## Quality Gates

Run inside `c:\Projects\Devalign\devalign-api` before finishing code changes:

- Lint: `uv run ruff check src/ tests/`
- Format check: `uv run ruff format --check src/ tests/`
- Type check (`mypy` strict mode): `uv run mypy src/`
- Unit tests: `uv run pytest -m "not integration and not slow"`

## Local Repo Skills

Domain skills under `devalign-api/.agents/skills/` include:
`architecture`, `implementation`, `planning`, `review`, `testing`, `software-architecture-planner`, `supabase`, and `supabase-postgres-best-practices`.

## Safety Boundaries

- **Never** run `uv run alembic upgrade head`, `uv run python scripts/seed_lightcast_skills.py`, or any write/seed command against a remote/production Supabase database without explicit user confirmation.
- **Never** read, print, or overwrite `.env`; modify only `.env.example` when adding configuration keys.
- Write unit tests for new use cases/endpoints and regression tests for bug fixes.
