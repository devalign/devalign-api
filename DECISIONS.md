# Architecture Decision Records

## ADR-001: Modular Monolith over Microservices

**Date:** 2026-05-15
**Status:** Accepted

**Context:**
The PRD described the system as "microservices based". However, the project has a single development team, academic context, and MVP requirements.

**Decision:**
Implement as a **modular monolith** with 4 bounded contexts:
- `scraper` — data acquisition
- `ml_engine` — ML inference pipeline
- `genai` — RAG + LLM orchestration
- `delivery` — REST API, auth, user management

Each module has clean architecture internally (domain/application/infrastructure/interface).

**Consequences:**
- Simpler deployment (single container)
- Easier development and debugging
- Modules can be extracted to services later if needed
- No inter-service network overhead

---

## ADR-002: LLM Provider Strategy

**Date:** 2026-05-15
**Status:** Accepted

**Context:**
Need a cost-effective LLM strategy for development + production.

**Decision:**
- **Development:** Groq API (`llama-3.1-70b-versatile`) — fast, free tier
- **Production:** OpenAI (`gpt-4o-mini`) — higher quality
- **Abstraction:** LangChain `LLMService` port — provider swappable via config

**Consequences:**
- Zero cost during development
- No code changes needed to switch providers
- LangChain adds a dependency layer

---

## ADR-003: Embedding Strategy

**Date:** 2026-05-15
**Status:** Accepted

**Context:**
Embeddings are needed for CV vectorization and pgvector search.

**Decision:**
- **Default:** `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Optional:** OpenAI `text-embedding-3-small` (higher quality, paid)
- **Abstraction:** `EmbeddingService` port — swappable via `EMBEDDING_PROVIDER` env var

**Consequences:**
- Zero API cost for development and academic project
- Local inference slightly slower but acceptable
- Can upgrade to OpenAI without code changes

---

## ADR-004: PostgreSQL + pgvector (Supabase)

**Date:** 2026-05-15
**Status:** Accepted

**Context:**
Need a relational database + vector store. Could use separate services (e.g., Pinecone for vectors).

**Decision:**
Use **Supabase PostgreSQL with pgvector extension** for both relational data and vector similarity search.

**Consequences:**
- Single database engine to manage
- Supabase provides managed auth + storage + DB
- pgvector has slightly lower performance than dedicated vector DBs at very large scale
- Acceptable for MVP (5K offers, <500 CVs)

---

## ADR-005: Scraper as External Repository

**Date:** 2026-05-15
**Status:** Accepted

**Context:**
The scraper was already implemented in a separate repository. The user confirmed it exists.

**Decision:**
- Define domain contracts (entities + ports) in this repo
- Stub the scraper module interface
- Defer integration to Phase 4
- Integration options: shared DB, package import, or CLI script

**Consequences:**
- Clean separation of concerns
- No blocking dependency on scraper for MVP
- Integration requires defining the data contract clearly

---

## ADR-006: Platforms for Scraping

**Date:** 2026-05-15
**Status:** Accepted

**Context:**
Original PRD mentioned LinkedIn, GetOnBoard, Computrabajo.

**Decision:**
Limit scraping to **GetOnBoard** and **Computrabajo** only.

**Reason:**
LinkedIn has strict anti-scraping ToS with legal risk. GetOnBoard has a developer-friendly API. Computrabajo is Peru-specific and high-relevance.

---

## ADR-007: Deployment Target

**Date:** 2026-05-15
**Status:** Under Consideration

**Options:**
| Platform | Cost | Complexity |
|----------|------|-----------|
| Koyeb | Free tier | Low |
| Railway | $5/mo | Low |
| Render | Free tier (with sleep) | Low |
| Fly.io | $5/mo | Low |

**Decision:**
Primary: **Koyeb** (free tier, no sleep). Secondary: **Railway** if Koyeb has limitations.

**Status:** To be validated when deploying.
