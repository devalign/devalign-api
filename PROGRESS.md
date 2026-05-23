# Devalign API — Progress

## Phase 0: Foundation ✅

- [x] Architecture plan created (ARCHITECTURE.md)
- [x] Technology stack defined (STACK.md)
- [x] ADRs documented (DECISIONS.md)
- [x] `pyproject.toml` — uv + ruff + mypy + pytest
- [x] `.env.example` — environment variable template
- [x] `.pre-commit-config.yaml` — quality hooks
- [x] `Makefile` — developer shortcuts
- [x] `Dockerfile` — multi-stage with uv
- [x] `docker-compose.yml` — development environment
- [x] `.github/workflows/ci.yml` — CI pipeline
- [x] Full `src/` directory structure created
- [x] `src/config.py` — pydantic-settings
- [x] `src/main.py` — FastAPI app factory
- [x] `src/dependencies.py` — global DI
- [x] `src/shared/` — database, exceptions, middleware, logging, security, supabase_client

## Phase 1: Module delivery ✅

- [x] `delivery/domain/entities.py` — User, CVDocument
- [x] `delivery/domain/ports.py` — UserRepository, CVRepository, StorageService
- [x] `delivery/application/dtos.py`
- [x] `delivery/application/use_cases.py` — GetCurrentUser, UploadCV, ListUserCVs
- [x] `delivery/infrastructure/models.py` — SQLAlchemy ORM models
- [x] `delivery/infrastructure/repository.py` — SQLAlchemy implementations
- [x] `delivery/infrastructure/supabase_storage.py` — Storage adapter
- [x] `delivery/interface/router.py` — /users/me, /users/me/cv endpoints

## Phase 2: Module ml_engine ✅

- [x] `ml_engine/domain/entities.py` — TechCluster, UserProfile, SkillGap
- [x] `ml_engine/domain/ports.py` — EmbeddingService, CVParserService, ClusterRepository
- [x] `ml_engine/application/dtos.py`
- [x] `ml_engine/application/use_cases.py` — ProfileUserFromCV, ListClusters
- [x] `ml_engine/infrastructure/cv_parser.py` — PDF + DOCX parser
- [x] `ml_engine/infrastructure/embeddings.py` — Local + OpenAI embedding services
- [x] `ml_engine/interface/router.py` — /profile/analyze, /profile/clusters

## Phase 3: Module genai ✅

- [x] `genai/domain/entities.py` — Roadmap, RoadmapPhase
- [x] `genai/domain/ports.py` — LLMService, VectorStorePort, RoadmapRepository
- [x] `genai/application/dtos.py`
- [x] `genai/application/use_cases.py` — GenerateRoadmap (RAG pipeline)
- [x] `genai/infrastructure/langchain_chain.py` — Groq + OpenAI LLM services
- [x] `genai/infrastructure/vector_store.py` — pgvector implementation
- [x] `genai/interface/router.py` — /roadmap/generate

## Phase 4: Scraper (Stub) ✅

- [x] `scraper/domain/entities.py` — JobOffer, RawSkillMention
- [x] `scraper/domain/ports.py` — ScraperPort, JobOfferRepository
- [x] `scraper/interface/router.py` — /scraper/status (stub endpoint)
- [ ] Integration with external scraper repository (deferred)

## Phase 5: Database (Alembic) ✅

- [x] `alembic.ini` configured (incl. logging)
- [x] `alembic/env.py` — async migrations support
- [x] ORM models for ML Engine (clusters, profiles, skills)
- [x] ORM models for GenAI (roadmaps)
- [x] ORM models for Scraper (job_offers, offer_skills)
- [x] Initial migration applied (`001_create_all_tables.py`) with `pgvector`

## Phase 6: Tests ✅

- [x] `tests/conftest.py` — async test client fixture
- [x] `tests/unit/test_health.py` — health check smoke tests
- [x] `tests/unit/test_exceptions.py` — exception hierarchy tests
- [ ] ML Engine unit tests
- [ ] GenAI unit tests (mock LLM)
- [ ] Delivery use case tests

## Phase 7: Frontend Auth Integration ✅

- [x] Extract full JWT payload in FastAPI security layer (`CurrentUserPayloadDep`)
- [x] JIT User Provisioning implemented in `GetCurrentUserUseCase`
- [x] Implicitly provision Google / Email profile metadata (name, email, avatar) on `GET /users/me`
- [x] Fixed pre-existing SQLAlchemy ORM typing bug in `SQLAlchemyUserRepository` (`UserModel.user_id`)

## Pending (Next Steps)

- [x] Develop ML Pipeline for skill normalization (populate `skills` from `raw_hard_skills`)
- [ ] Seed pgvector with SFIA9/SWECOM documents (`scripts/seed_vectors.py`)
- [x] Integration testing with `devalign-scraping` (Supabase DB schema aligned & skills normalized)
- [ ] ML Engine unit tests
- [ ] GenAI unit tests (mock LLM)
- [ ] Delivery use case tests

