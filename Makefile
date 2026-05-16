# =============================================================
# Devalign API — Makefile
# =============================================================

.PHONY: help install dev test lint type-check format check migrate seed

# Default target
help:
	@echo ""
	@echo "Devalign API — Available commands:"
	@echo ""
	@echo "  make install      Install dependencies (requires uv)"
	@echo "  make dev          Start development server with hot reload"
	@echo "  make test         Run test suite with coverage"
	@echo "  make lint         Run Ruff linter"
	@echo "  make format       Run Ruff formatter"
	@echo "  make type-check   Run mypy strict type checking"
	@echo "  make check        Run lint + type-check + tests (full CI)"
	@echo "  make migrate      Run pending Alembic migrations"
	@echo "  make seed         Seed pgvector with SFIA9/SWECOM documents"
	@echo ""

install:
	uv sync --all-extras

dev:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest -m "not integration and not slow"

test-all:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

type-check:
	uv run mypy src/

check: lint type-check test

migrate:
	uv run alembic upgrade head

migrate-create:
	@read -p "Migration message: " msg; uv run alembic revision --autogenerate -m "$$msg"

seed:
	uv run python scripts/seed_vectors.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

pre-commit-install:
	uv run pre-commit install
