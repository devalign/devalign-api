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

---

## ADR-008: Database Connection Strategy (Supabase)

**Date:** 2026-05-16
**Status:** Accepted

**Context:**
Alembic migrations initially failed with `WinError 121` (timeout) and `socket.gaierror` when connecting to the direct Postgres instance (`db.[project-id].supabase.co`). Supabase recently moved direct connections to IPv6-only in some regions, which causes DNS/routing failures on local networks that only support IPv4.

**Decision:**
Use the **Supabase Connection Pooler** (`aws-0-[region].pooler.supabase.com`) in **Session Mode** (port 5432) as the standard `DATABASE_URL` for both local development (Alembic) and production.

**Consequences:**
- Guarantees IPv4 compatibility and bypasses local DNS/IPv6 routing issues.
- Provides connection pooling out-of-the-box, essential if deployed to Serverless environments.
- Requires appending `?ssl=require` and formatting the username as `postgres.[project-id]`.

---

## ADR-009: Skill Normalization State Management and Pipeline

**Date:** 2026-05-17
**Status:** Accepted

**Context:**
The scraping module populates `job_offers` continually with raw skill text. We need a robust mechanism in the ML Engine to process and normalize these skills into `skills` (canonical catalog) and `offer_skills` without redundant execution or data duplication.

**Decision:**
1. **Control of State:** Add `is_normalized` (Boolean, default False) directly to `job_offers` schema. The pipeline queries only `is_normalized = False` and sets them to `True` after successful mapping.
2. **Deduplication Strategy:** Combine exact match lookup (fast) with semantic cosine similarity using `all-MiniLM-L6-v2` local embeddings.
3. **Threshold:** Set similarity threshold to `0.88` to map similar terms (e.g. "react.js" and "react") to a single canonical skill without grouping unrelated skills.
4. **Endpoint Exposure:** Expose the pipeline as `POST /api/v1/profile/normalize-skills` for programmatic control.

**Consequences:**
- Optimal computing efficiency (only new job offers are processed).
- Eliminates manual DB cleanup.
- Clean canonical catalog containing unique high-quality skills.
- Low-latency inference after the embedding model is loaded.

---

## ADR-010: Just-In-Time (JIT) User Provisioning for Frontend Authentication

**Date:** 2026-05-17
**Status:** Accepted

**Context:**
The frontend utilizes Supabase Auth for managing user login and signup (via email/password or Google OAuth). Once authenticated, the frontend calls the FastAPI backend. However, because user accounts are created inside Supabase's internal `auth.users` schema, they do not automatically exist in our application's local `public.users` table, which is required for relational integrity (such as uploading CVs). We need a simple, lightweight way to synchronize these records.

**Decision:**
Implement **Just-In-Time (JIT) Provisioning** within the `GET /users/me` endpoint:
1. Decode the Supabase JWT using the `SUPABASE_ANON_KEY` to extract identity information (user ID, email, name, and avatar URL).
2. Query the local `public.users` table to see if the user profile exists.
3. If the user does not exist (meaning it's their first time accessing the API after registration), automatically create and persist their user entity locally *on-the-fly*.
4. Return the user profile model to the frontend seamlessly.

**Reasoning:**
This approach avoids complex, brittle database triggers or Webhooks that listen to Supabase Auth events, which are difficult to manage locally and lack deployment portability. It encapsulates the synchronization logic in the application layer, guaranteeing a smooth and decoupled "One-Click" sign-in integration for the frontend.

**Consequences:**
- The frontend doesn't need to make an explicit "Register Profile" call.
- Eliminates overhead and latency during signup.
- Decouples local database constraints from external authentication actions.

