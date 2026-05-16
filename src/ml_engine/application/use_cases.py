"""ML Engine use cases."""

import math
from uuid import UUID

import structlog

from src.ml_engine.application.dtos import ClusterAffinityDTO, ClusterDTO, SkillDTO, UserProfileDTO
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    SeniorityLevel,
    UserProfile,
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
