# Project Overview

**Devalign API** — Backend inteligente para el análisis de competencias técnicas demandadas por el sector IT del Perú.

El sistema combina:
- **Web Scraping** de ofertas laborales (GetOnBoard, Computrabajo)
- **Machine Learning** (K-Prototypes clustering + embeddings semánticos)
- **RAG con LangChain** para generación de roadmaps personalizados
- **FastAPI** como framework de entrega con autenticación Supabase

---

# Engineering Workflow

Always follow this workflow:

1. Clarify requirements
2. Generate architecture
3. Define specs
4. Create implementation plan
5. Generate tests
6. Implement incrementally
7. Validate continuously
8. Run automated checks
9. Review/refactor
10. Commit/document

Never skip planning.

---

# Engineering Principles

Prioritize:
- maintainability
- scalability
- modularity
- readability
- explicit technical decisions

Avoid:
- overengineering
- hidden coupling
- premature abstractions

---

# Architecture Rules

- Modular Monolith with Clean Architecture (4 bounded contexts)
- Prefer explicit boundaries between modules
- Keep modules cohesive (domain/application/infrastructure/interface per module)
- Maintain low coupling via ports/adapters pattern
- Design for future AI integrations
- See ARCHITECTURE.md for full architecture documentation

---

# Module Map

| Module | Prefix | Responsibility |
|--------|--------|----------------|
| `delivery` | `/api/v1/users` | Auth, CV upload, user profile |
| `ml_engine` | `/api/v1/profile` | CV parsing, embeddings, clustering, gap detection |
| `genai` | `/api/v1/roadmap` | RAG pipeline, LLM roadmap generation |
| `scraper` | `/api/v1/scraper` | Job offer acquisition (stub — external repo) |

---

# Development Standards

- Use strict typing (mypy strict)
- Follow Ruff linting and formatting (line-length: 100)
- Write tests for critical logic (pytest + pytest-asyncio)
- Validate edge cases
- Update documentation continuously

---

# Quick Start

```bash
# Install dependencies
make install

# Copy env template
cp .env.example .env
# Fill in your .env values

# Start development server
make dev

# Run tests
make test

# Lint + type-check + test (full CI)
make check
```

---

# Documentation Rules

As an AI agent working on this project, you MUST proactively follow this protocol without being asked:
- **PROGRESS.md**: Automatically update it AT THE END of any task, feature implementation, or bug fix. Mark tasks as completed before finishing your response.
- **DECISIONS.md**: Update it WHENEVER an architectural decision is made, a key dependency is added, or a design pattern is defined.
- **TASKS.md**: Keep it strictly synchronized with pending tasks.

---

# Behavior Rules

- Think like a Staff Engineer
- Justify important decisions
- Analyze tradeoffs
- Ask questions if requirements are ambiguous
- Never implement large features without planning first