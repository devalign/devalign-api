# Devalign API Agent Notes

## Scope

Applies to the backend in this folder. For full context, start with the docs:
- [README](README.md)
- [Progress tracking](PROGRESS.md)
- [Architecture decisions](../docs/decisions.md)
- [Stack](../docs/stack.md)

For work orchestration, follow the repo harness:
- [.harness overview](../.harness/HARNESS.md)

## Architecture

- Modular monolith with 4 bounded contexts under `src/`: `delivery`, `ml_engine`, `genai`, `scraper`.
- Clean Architecture per module (domain/application/infrastructure/interface).

## Conventions

- Ruff formatting, line length 100.
- mypy strict, pytest + pytest-asyncio.
- Async SQLAlchemy + asyncpg.
- App settings live in `src/config.py`.
- Dependency injection lives in `src/dependencies.py`.

## Workflow

- Plan first for non-trivial changes.
- Do not use `PROGRESS.md` for tracking work status.
- Use .harness workflows, state, and handoffs for tracking and transitions.
- Update `docs/decisions.md` when making a new architectural or dependency choice.