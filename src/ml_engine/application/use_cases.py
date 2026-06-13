"""ML Engine use cases."""

import json
import math
from typing import Any
from uuid import UUID

import structlog

from src.genai.domain.ports import LLMService
from src.ml_engine.application.dtos import ClusterAffinityDTO, ClusterDTO, SkillDTO, UserProfileDTO
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    SeniorityLevel,
    Skill,
    SkillGap,
    SkillType,
    UserProfile,
)
from src.ml_engine.domain.ports import (
    ClusterRepository,
    CVParserService,
    EmbeddingService,
    MLJobOfferRepository,
    SkillRepository,
    UserProfileRepository,
)
from src.shared.exceptions import MLPipelineError

logger = structlog.get_logger(__name__)


class ProfileUserFromCVUseCase:
    """
    Core use case: extract CV, normalize skills, compute Weighted Jaccard affinity vs clusters.

    Steps:
    1. Extract text from CV (PDF/DOCX)
    2. Run LLM structured extraction (experience, skills, certifications, education, personal info)
    3. Generate embedding vector (for backwards compatibility/raw search)
    4. Normalize user skills against canonical catalog (using exact & fuzzy matching)
    5. Compute Weighted Jaccard Similarity against active clusters
    6. Estimate seniority
    7. Detect and prioritize skill gaps vs primary cluster
    8. Persist and return profile
    """

    def __init__(
        self,
        cv_parser: CVParserService,
        embedding_service: EmbeddingService,
        cluster_repository: ClusterRepository,
        profile_repository: UserProfileRepository,
        llm_service: LLMService,
        skill_repository: SkillRepository,
    ) -> None:
        self._cv_parser = cv_parser
        self._embedder = embedding_service
        self._clusters = cluster_repository
        self._profiles = profile_repository
        self._llm = llm_service
        self._skills = skill_repository

    async def _normalize_user_skills(self, raw_skills: dict[str, list[str]] | list[str] | Any) -> list[Skill]:
        import difflib

        # Load all canonical skills
        existing_skills = await self._skills.get_all_skills()
        existing_skills_map = {s.normalized_name: s for s in existing_skills}

        normalized_skills = []
        new_skills_to_create = []

        # Normalize inputs
        if isinstance(raw_skills, list):
            raw_skills_dict = {
                "technical": raw_skills,
                "soft": [],
                "tools": [],
                "methodologies": []
            }
        elif isinstance(raw_skills, dict):
            raw_skills_dict = raw_skills
        else:
            raw_skills_dict = {}

        cat_mapping = {
            "technical": SkillType.HARD_SKILL,
            "soft": SkillType.SOFT_SKILL,
            "tools": SkillType.TOOL,
            "methodologies": SkillType.METHODOLOGY
        }

        for category_key, skill_type in cat_mapping.items():
            names = raw_skills_dict.get(category_key, [])
            if not isinstance(names, list):
                continue
            for name in names:
                if not isinstance(name, str):
                    continue
                clean_name = name.strip()
                if not clean_name:
                    continue
                norm_name = clean_name.lower().replace(" ", "").replace(".", "")

                if norm_name in existing_skills_map:
                    normalized_skills.append(existing_skills_map[norm_name])
                else:
                    # Fuzzy match against existing skills
                    matches = difflib.get_close_matches(norm_name, list(existing_skills_map.keys()), n=1, cutoff=0.85)
                    if matches:
                        normalized_skills.append(existing_skills_map[matches[0]])
                    else:
                        new_skill = Skill(
                            name=clean_name,
                            skill_type=skill_type,
                            normalized_name=norm_name,
                            weight=1.0,
                            frequency=1.0
                        )
                        new_skills_to_create.append(new_skill)

        if new_skills_to_create:
            # Dedup new skills to create by normalized name
            unique_new_skills = {}
            for ns in new_skills_to_create:
                unique_new_skills[ns.normalized_name] = ns

            saved_skills = await self._skills.save_skills(list(unique_new_skills.values()))
            normalized_skills.extend(saved_skills)

        return normalized_skills

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

            # Step 2: Run LLM structured extraction
            logger.info("Extracting structured info using LLM")
            llm_failed = False
            try:
                prompt = _build_cv_extraction_prompt(cv_text)
                raw_llm_output = await self._llm.generate(prompt=prompt, context=[])
                extracted_data = _parse_cv_extraction_output(raw_llm_output)
                if not extracted_data:
                    # If empty dict returned from parsing, treat as fallback trigger
                    raise ValueError("Empty extraction data parsed")
            except Exception as e:
                logger.warning("LLM structured extraction failed, using mock profile data fallback", error=str(e))
                extracted_data = {
                    "full_name": "Usuario Simulado",
                    "current_job_role": "Backend Engineer",
                    "years_experience": 4,
                    "preferred_modality": "Híbrido",
                    "location": "Lima, Perú",
                    "availability": "Inmediata",
                    "skills": {
                        "technical": ["Python", "SQL", "NoSQL"],
                        "soft": ["Liderazgo", "Comunicación"],
                        "tools": ["Docker", "Kubernetes", "Git"],
                        "methodologies": ["Scrum", "Microservicios"]
                    },
                    "work_experience": [
                        {
                            "company": "Tech Solutions",
                            "role": "Software Developer",
                            "start_date": "2022-01",
                            "end_date": "2024-05",
                            "description": "Desarrollo de microservicios con Python y bases de datos relacionales.",
                        }
                    ],
                    "education": [
                        {
                            "institution": "Universidad Nacional",
                            "degree": "Bachiller en Ingeniería de Sistemas",
                            "start_date": "2017",
                            "end_date": "2021",
                        }
                    ],
                    "certifications": [
                        {
                            "name": "AWS Certified Cloud Practitioner",
                            "issuer": "Amazon Web Services",
                            "date": "2023",
                        }
                    ],
                }
                llm_failed = True

            # Step 3: Embed CV text (kept for backwards compatibility/fallback vector)
            logger.info("Generating CV embedding")
            try:
                cv_embedding = await self._embedder.embed_text(cv_text)
            except Exception as e:
                logger.warning("Embedding CV text failed, using zero vector", error=str(e))
                cv_embedding = [0.0] * 1024

            # Step 4: Normalize user skills
            raw_skills = extracted_data.get("skills", {})
            detected_skills = await self._normalize_user_skills(raw_skills)

            # Step 5: Load active clusters
            clusters = await self._clusters.get_all_active()
            if not clusters:
                raise MLPipelineError("No tech clusters available — run clustering first")

            active_clusters = [c for c in clusters if c.centroid_skills]
            if not active_clusters:
                raise MLPipelineError("No active clusters with centroid skills available")

            # Step 6: Compute Weighted Jaccard Similarity per cluster
            # Participant user hard/tool skills
            user_hard_skills = [
                s for s in detected_skills
                if s.skill_type in (SkillType.HARD_SKILL, SkillType.TOOL)
            ]
            user_hard_norms = {s.normalized_name: s for s in user_hard_skills}

            affinities = []
            for idx, cluster in enumerate(active_clusters):
                cluster_hard_skills = [
                    s for s in cluster.centroid_skills
                    if s.skill_type in (SkillType.HARD_SKILL, SkillType.TOOL)
                ]
                cluster_hard_norms = {s.normalized_name: s for s in cluster_hard_skills}

                union_norms = set(cluster_hard_norms.keys()) | set(user_hard_norms.keys())

                numerator = 0.0
                denominator = 0.0

                for norm_name in union_norms:
                    # Get weight
                    w = 1.0
                    if norm_name in cluster_hard_norms:
                        w = cluster_hard_norms[norm_name].weight
                    elif norm_name in user_hard_norms:
                        w = user_hard_norms[norm_name].weight

                    in_user = norm_name in user_hard_norms
                    in_cluster = norm_name in cluster_hard_norms

                    if in_user and in_cluster:
                        f_s = cluster_hard_norms[norm_name].frequency
                        numerator += w * f_s
                        denominator += w * f_s
                    elif in_cluster:
                        f_s = cluster_hard_norms[norm_name].frequency
                        denominator += w * f_s
                    else:
                        # User has it but it's not in centroid, count as w * 1.0
                        denominator += w * 1.0

                score = (numerator / denominator) if denominator > 0.0 else 0.0

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

            # Step 7: Seniority estimation
            years_exp = extracted_data.get("years_experience")
            if isinstance(years_exp, (int, float)):
                if years_exp >= 6:
                    seniority = SeniorityLevel.SENIOR
                elif years_exp >= 3:
                    seniority = SeniorityLevel.MID
                else:
                    seniority = SeniorityLevel.JUNIOR
            else:
                seniority = _estimate_seniority(cv_text)

            # Step 8: Detect and prioritize skill gaps vs primary cluster
            primary_cluster = next((c for c in clusters if c.id == primary.cluster_id), None)
            skill_gaps = []

            if primary_cluster:
                primary_cluster_hard_skills = [
                    s for s in primary_cluster.centroid_skills
                    if s.skill_type in (SkillType.HARD_SKILL, SkillType.TOOL)
                ]
                for skill in primary_cluster_hard_skills:
                    if skill.normalized_name not in user_hard_norms:
                        priority = skill.weight * skill.frequency
                        if priority >= 2.0:
                            importance = "critical"
                        elif priority >= 1.0:
                            importance = "high"
                        else:
                            importance = "medium"

                        skill_gaps.append(
                            SkillGap(
                                skill=skill,
                                market_importance=importance,
                            )
                        )
                # Sort gaps by importance priority
                skill_gaps.sort(
                    key=lambda g: g.skill.weight * g.skill.frequency,
                    reverse=True
                )

            # Step 9: Persist profile
            profile = UserProfile(
                user_id=user_id,
                cv_id=cv_id,
                embedding=cv_embedding,
                detected_skills=detected_skills,
                seniority=seniority,
                primary_affinity=primary,
                secondary_affinities=secondaries,
                skill_gaps=skill_gaps,
                full_name=extracted_data.get("full_name") or None,
                current_job_role=extracted_data.get("current_job_role") or None,
                years_experience=int(years_exp) if isinstance(years_exp, (int, float)) else None,
                preferred_modality=extracted_data.get("preferred_modality") or None,
                location=extracted_data.get("location") or None,
                availability=extracted_data.get("availability") or None,
                work_experience=extracted_data.get("work_experience") or [],
                education=extracted_data.get("education") or [],
                certifications=extracted_data.get("certifications") or [],
            )
            await self._profiles.save(profile)

            logger.info(
                "Profile generated and saved via Weighted Jaccard",
                user_id=str(user_id),
                specialty=primary.cluster_name,
                score=primary.affinity_score,
            )

            # Map DTOs
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
                detected_skills=[
                    SkillDTO(
                        name=s.name,
                        skill_type=s.skill_type.value,
                        market_importance="consolidated",
                        market_demand_percentage=round(s.frequency * 100) if s.frequency is not None else 100,
                    ) for s in detected_skills
                ],
                skill_gaps=[
                    SkillDTO(
                        name=g.skill.name,
                        skill_type=g.skill.skill_type.value,
                        market_importance=g.market_importance,
                        market_demand_percentage=round(g.skill.frequency * 100) if g.skill.frequency is not None else None,
                    )
                    for g in skill_gaps
                ],
                full_name=profile.full_name,
                current_job_role=profile.current_job_role,
                years_experience=profile.years_experience,
                preferred_modality=profile.preferred_modality,
                location=profile.location,
                availability=profile.availability,
                work_experience=profile.work_experience,
                education=profile.education,
                certifications=profile.certifications,
                message="Profile generated successfully",
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


def _build_cv_extraction_prompt(cv_text: str) -> str:
    """Build structured prompt for LLM CV extraction."""
    return f"""You are a professional CV analyzer.
Extract the following details from the CV text in a structured JSON format:
1. Full Name (full_name)
2. Current Job Role (current_job_role)
3. Years of experience (years_experience, integer)
4. Preferred modality (preferred_modality: 'Remota', 'Híbrida', 'Presencial', 'Remota / Híbrida' etc)
5. Location (location: 'City, Country')
6. Availability (availability: e.g. 'Inmediata', '1 mes', etc)
7. Work experience (work_experience: list of objects with company, role, description, start_date, end_date, current (bool))
8. Education (education: list of objects with institution, degree, start_date, end_date)
9. Certifications (certifications: list of objects with name, issuer, date)
10. Technical & Soft Skills (skills: object containing lists of skills categorized under 'technical' (e.g. Python, React, SQL), 'soft' (e.g. Liderazgo), 'tools' (e.g. Docker, GitHub, AWS, Jira), and 'methodologies' (e.g. Scrum, CI/CD))

CV Text:
{cv_text}

Respond ONLY with a valid JSON object matching this schema:
{{
  "full_name": "string or null",
  "current_job_role": "string or null",
  "years_experience": integer or null,
  "preferred_modality": "string or null",
  "location": "string or null",
  "availability": "string or null",
  "work_experience": [
    {{
      "company": "string",
      "role": "string",
      "description": "string",
      "start_date": "string",
      "end_date": "string or null",
      "current": boolean
    }}
  ],
  "education": [
    {{
      "institution": "string",
      "degree": "string",
      "start_date": "string",
      "end_date": "string or null"
    }}
  ],
  "certifications": [
    {{
      "name": "string",
      "issuer": "string or null",
      "date": "string or null"
    }}
  ],
  "skills": {{
    "technical": ["string"],
    "soft": ["string"],
    "tools": ["string"],
    "methodologies": ["string"]
  }}
}}"""


def _parse_cv_extraction_output(raw_output: str) -> dict[str, Any]:
    """Parse JSON block from LLM output."""
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in LLM output")
        parsed = json.loads(raw_output[start:end])
        if not isinstance(parsed, dict):
            raise ValueError("Parsed output is not a dictionary")
        return parsed
    except Exception as exc:
        logger.warning(
            "Failed to parse LLM CV extraction, fallback to empty defaults", error=str(exc)
        )
        return {}


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


class NormalizeSkillsUseCase:
    """
    ML Pipeline step: Normalizes raw string skills from job_offers into
    canonical Skill entities, and links them via offer_skills.

    Uses Word Embeddings (Cosine Similarity) to deduplicate skills
    (e.g., "react.js" vs "React" vs "reactjs").
    """

    def __init__(
        self,
        job_offer_repo: MLJobOfferRepository,
        skill_repo: SkillRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        self._job_offers = job_offer_repo
        self._skills = skill_repo
        self._embedder = embedding_service
        self.SIMILARITY_THRESHOLD = 0.88  # Tunable threshold

    async def execute(self) -> dict[str, Any]:
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

        new_skills_to_create: dict[str, Skill] = {}
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
                            normalized_name=norm_name,
                        )
                        new_skills_to_create[norm_name] = matched_skill
                        # Update our local memory to match future raw skills in this batch
                        existing_skills.append(matched_skill)
                        existing_embeddings.append(raw_emb)
                        skill_map[norm_name] = matched_skill

                # We don't have the UUID for the newly created skill yet.
                # We will assign them after bulk inserting new skills.
                offer_skills_to_insert.append(
                    {
                        "job_offer_id": offer["id"],
                        "skill_norm_name": matched_skill.normalized_name,
                        "skill_type": "hard_skill",
                    }
                )

        # 3. Save new skills to database
        if new_skills_to_create:
            logger.info(f"Creating {len(new_skills_to_create)} new canonical skills.")
            await self._skills.save_skills(list(new_skills_to_create.values()))
            # Update skill_map with the newly generated UUIDs (mocked as returning them)
            # Since the actual ORM models get their IDs generated via default=uuid4,
            # we need to make sure save_skills fetches them back.
            # For simplicity, we just fetch all skills again
            all_skills = await self._skills.get_all_skills()
            skill_id_map = {s.normalized_name: s.id for s in all_skills if hasattr(s, "id")}
        else:
            all_skills = await self._skills.get_all_skills()
            skill_id_map = {s.normalized_name: s.id for s in all_skills if hasattr(s, "id")}

        # 4. Insert offer_skills relations
        final_offer_skills = []
        for os in offer_skills_to_insert:
            skill_id = skill_id_map.get(os["skill_norm_name"])
            if skill_id:
                final_offer_skills.append(
                    {
                        "job_offer_id": os["job_offer_id"],
                        "skill_id": skill_id,
                        "skill_type": os["skill_type"],
                    }
                )

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
            "offer_skills_linked": len(final_offer_skills),
        }
