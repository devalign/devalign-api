"""ML Engine use cases."""

import math
from uuid import UUID

import structlog

from src.ml_engine.application.dtos import ClusterAffinityDTO, ClusterDTO, SkillDTO, UserProfileDTO
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    SeniorityLevel,
    UserProfile,
    Skill,
    SkillType,
)
from src.ml_engine.domain.ports import (
    ClusterRepository,
    CVParserService,
    EmbeddingService,
    UserProfileRepository,
)
from src.shared.exceptions import MLPipelineError

logger = structlog.get_logger(__name__)


class ProfileUserFromCVUseCase:
    """
    Core use case: vectorize a CV, match against clusters, detect gaps.

    Steps:
    1. Extract text from CV (PDF/DOCX)
    2. Generate embedding vector
    3. Compute cosine similarity against all cluster centroids
    4. Detect skill gaps vs target cluster
    5. Estimate seniority (heuristic on skill count + cluster)
    6. Persist and return profile
    """

    def __init__(
        self,
        cv_parser: CVParserService,
        embedding_service: EmbeddingService,
        cluster_repository: ClusterRepository,
        profile_repository: UserProfileRepository,
    ) -> None:
        self._cv_parser = cv_parser
        self._embedder = embedding_service
        self._clusters = cluster_repository
        self._profiles = profile_repository

    async def execute(
        self,
        user_id: UUID,
        cv_id: UUID,
        cv_content: bytes,
        content_type: str,
    ) -> UserProfileDTO:
        try:
            # Step 1: Extract text
            logger.info("Extracting CV text", user_id=str(user_id))
            cv_text = await self._cv_parser.extract_text(cv_content, content_type)

            if not cv_text.strip():
                raise MLPipelineError("CV text extraction returned empty content")

            # Step 2: Embed CV text
            logger.info("Generating CV embedding")
            cv_embedding = await self._embedder.embed_text(cv_text)

            # Step 3: Load clusters and compute similarities
            clusters = await self._clusters.get_all_active()
            if not clusters:
                raise MLPipelineError("No tech clusters available — run clustering first")

            # Step 4: Compute cosine similarity per cluster
            affinities = []
            for cluster in clusters:
                if not cluster.centroid_skills:
                    continue
                # Use cluster name + skills as centroid text representation
                centroid_text = f"{cluster.name}: " + ", ".join(
                    s.normalized_name for s in cluster.centroid_skills
                )
                centroid_embedding = await self._embedder.embed_text(centroid_text)
                score = _cosine_similarity(cv_embedding, centroid_embedding)
                affinities.append(
                    ClusterAffinity(
                        cluster_id=cluster.id,
                        cluster_name=cluster.name,
                        affinity_score=score,
                        is_primary=False,
                    )
                )

            # Sort and mark primary
            affinities.sort(key=lambda a: a.affinity_score, reverse=True)
            primary = affinities[0]
            primary = ClusterAffinity(
                cluster_id=primary.cluster_id,
                cluster_name=primary.cluster_name,
                affinity_score=primary.affinity_score,
                is_primary=True,
            )
            secondaries = affinities[1:3]  # Top 2 secondary affinities

            # Step 5: Seniority heuristic (placeholder — will improve with real data)
            seniority = _estimate_seniority(cv_text)

            # Step 6: Gap detection (simplified — full impl in next phase)
            primary_cluster = next((c for c in clusters if c.id == primary.cluster_id), None)
            skill_gaps = []
            if primary_cluster:
                detected_skill_names = {word.lower() for word in cv_text.split()}
                for skill in primary_cluster.centroid_skills:
                    if skill.normalized_name not in detected_skill_names:
                        skill_gaps.append(
                            SkillDTO(
                                name=skill.name,
                                skill_type=skill.skill_type.value,
                                market_importance="high",
                            )
                        )

            # Persist profile
            profile = UserProfile(
                user_id=user_id,
                cv_id=cv_id,
                embedding=cv_embedding,
                detected_skills=[],  # Full NER extraction in future phase
                seniority=seniority,
                primary_affinity=primary,
                secondary_affinities=secondaries,
                skill_gaps=[],
            )
            await self._profiles.save(profile)

            logger.info(
                "Profile generated",
                user_id=str(user_id),
                specialty=primary.cluster_name,
                score=primary.affinity_score,
            )

            return UserProfileDTO(
                user_id=user_id,
                cv_id=cv_id,
                seniority=seniority.value,
                primary_specialty=primary.cluster_name,
                alignment_score=primary.affinity_score,
                secondary_affinities=[
                    ClusterAffinityDTO(
                        cluster_id=a.cluster_id,
                        cluster_name=a.cluster_name,
                        affinity_score=a.affinity_score,
                        is_primary=False,
                    )
                    for a in secondaries
                ],
                skill_gaps=skill_gaps[:10],  # Return top 10 gaps
            )

        except MLPipelineError:
            raise
        except Exception as exc:
            logger.exception("ML pipeline failed", error=str(exc))
            raise MLPipelineError("Profile generation failed unexpectedly") from exc


class ListClustersUseCase:
    """Return all available tech clusters (market specialties)."""

    def __init__(self, cluster_repository: ClusterRepository) -> None:
        self._clusters = cluster_repository

    async def execute(self) -> list[ClusterDTO]:
        clusters = await self._clusters.get_all_active()
        return [
            ClusterDTO(
                id=c.id,
                name=c.name,
                description=c.description,
                top_skills=[s.name for s in c.centroid_skills[:8]],
                job_offer_count=c.job_offer_count,
            )
            for c in clusters
        ]


# === Helpers ===


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm1 = math.sqrt(sum(a**2 for a in v1))
    norm2 = math.sqrt(sum(b**2 for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _estimate_seniority(cv_text: str) -> SeniorityLevel:
    """Heuristic seniority estimation based on keyword presence in CV text."""
    text_lower = cv_text.lower()
    senior_keywords = {"architect", "lead", "principal", "staff", "senior", "tech lead"}
    mid_keywords = {"mid", "intermediate", "semi-senior"}

    if any(kw in text_lower for kw in senior_keywords):
        return SeniorityLevel.SENIOR
    if any(kw in text_lower for kw in mid_keywords):
        return SeniorityLevel.MID
    return SeniorityLevel.JUNIOR


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


class NormalizeSkillsUseCase:
    """
    ML Pipeline step: Normalizes raw string skills from job_offers into
    canonical Skill entities, and links them via offer_skills.
    
    Uses Word Embeddings (Cosine Similarity) to deduplicate skills
    (e.g., "react.js" vs "React" vs "reactjs").
    """

    def __init__(
        self,
        job_offer_repo,
        skill_repo,
        embedding_service: EmbeddingService,
    ) -> None:
        self._job_offers = job_offer_repo
        self._skills = skill_repo
        self._embedder = embedding_service
        self.SIMILARITY_THRESHOLD = 0.88  # Tunable threshold

    async def execute(self) -> dict:
        """Run the normalization pipeline for unnormalized job offers."""
        from src.ml_engine.domain.entities import Skill, SkillType

        logger.info("Starting Skill Normalization Pipeline")
        
        # 1. Fetch unnormalized offers
        offers = await self._job_offers.get_unnormalized_offers(limit=100)
        if not offers:
            logger.info("No unnormalized offers found.")
            return {"processed_offers": 0, "new_skills": 0}

        # 2. Fetch existing skills
        existing_skills = await self._skills.get_all_skills()
        skill_map = {s.normalized_name: s for s in existing_skills}
        
        # We also need embeddings for existing skills to do semantic matching
        # In a real production system, you'd cache these embeddings.
        # For this prototype, we'll embed the names of existing skills on the fly
        # if the exact match fails.
        existing_skill_names = [s.name for s in existing_skills]
        if existing_skill_names:
            existing_embeddings = await self._embedder.embed_batch(existing_skill_names)
        else:
            existing_embeddings = []

        new_skills_to_create = {}
        offer_skills_to_insert = []
        processed_offer_ids = []

        for offer in offers:
            processed_offer_ids.append(offer["id"])
            
            raw_skills = offer.get("raw_hard_skills", [])
            
            for raw_skill in raw_skills:
                clean_name = raw_skill.strip()
                norm_name = clean_name.lower().replace(" ", "").replace(".", "")
                
                if not norm_name:
                    continue
                    
                matched_skill = None
                
                # 2.1 Exact match
                if norm_name in skill_map:
                    matched_skill = skill_map[norm_name]
                elif norm_name in new_skills_to_create:
                    matched_skill = new_skills_to_create[norm_name]
                else:
                    # 2.2 Semantic match (Cosine similarity)
                    raw_emb = await self._embedder.embed_text(clean_name)
                    best_score = 0.0
                    best_idx = -1
                    
                    for i, ext_emb in enumerate(existing_embeddings):
                        score = _cosine_similarity(raw_emb, ext_emb)
                        if score > best_score:
                            best_score = score
                            best_idx = i
                            
                    if best_score >= self.SIMILARITY_THRESHOLD:
                        matched_skill = existing_skills[best_idx]
                    else:
                        # 2.3 Create new skill
                        matched_skill = Skill(
                            name=clean_name,
                            skill_type=SkillType.HARD_SKILL,
                            normalized_name=norm_name
                        )
                        new_skills_to_create[norm_name] = matched_skill
                        # Update our local memory to match future raw skills in this batch
                        existing_skills.append(matched_skill)
                        existing_embeddings.append(raw_emb)
                        skill_map[norm_name] = matched_skill

                # We don't have the UUID for the newly created skill yet.
                # We will assign them after bulk inserting new skills.
                offer_skills_to_insert.append({
                    "job_offer_id": offer["id"],
                    "skill_norm_name": matched_skill.normalized_name,
                    "skill_type": "hard_skill"
                })

        # 3. Save new skills to database
        if new_skills_to_create:
            logger.info(f"Creating {len(new_skills_to_create)} new canonical skills.")
            saved_skills = await self._skills.save_skills(list(new_skills_to_create.values()))
            # Update skill_map with the newly generated UUIDs (mocked as returning them)
            # Since the actual ORM models get their IDs generated via default=uuid4,
            # we need to make sure save_skills fetches them back.
            # For simplicity, we just fetch all skills again
            all_skills = await self._skills.get_all_skills()
            skill_id_map = {s.normalized_name: s.id for s in all_skills if hasattr(s, 'id')}
        else:
            all_skills = await self._skills.get_all_skills()
            skill_id_map = {s.normalized_name: s.id for s in all_skills if hasattr(s, 'id')}

        # 4. Insert offer_skills relations
        final_offer_skills = []
        for os in offer_skills_to_insert:
            skill_id = skill_id_map.get(os["skill_norm_name"])
            if skill_id:
                final_offer_skills.append({
                    "job_offer_id": os["job_offer_id"],
                    "skill_id": skill_id,
                    "skill_type": os["skill_type"]
                })

        if final_offer_skills:
            logger.info(f"Saving {len(final_offer_skills)} offer_skills relations.")
            await self._job_offers.save_offer_skills(final_offer_skills)

        # 5. Mark offers as normalized
        if processed_offer_ids:
            logger.info(f"Marking {len(processed_offer_ids)} offers as normalized.")
            await self._job_offers.mark_as_normalized(processed_offer_ids)

        return {
            "processed_offers": len(processed_offer_ids),
            "new_skills": len(new_skills_to_create),
            "offer_skills_linked": len(final_offer_skills)
        }
