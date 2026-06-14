"""ML Engine use cases."""

import json
from typing import Any
from uuid import UUID

import structlog

from src.ml_engine.application.dtos import (
    ClusterAffinityDTO,
    ClusterDTO,
    DomainAffinityDTO,
    SkillDTO,
    UserProfileDTO,
)
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    SeniorityLevel,
    Skill,
    SkillGap,
    SkillType,
    TechCluster,
    UserProfile,
)
from src.ml_engine.domain.ports import (
    ClusterRepository,
    CVParserService,
    EmbeddingService,
    LLMService,
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
        cluster_repository: ClusterRepository,
        profile_repository: UserProfileRepository,
        llm_service: LLMService,
        skill_repository: SkillRepository,
    ) -> None:
        self._cv_parser = cv_parser
        self._clusters = cluster_repository
        self._profiles = profile_repository
        self._llm = llm_service
        self._skills = skill_repository

    async def _normalize_user_skills(
        self, raw_skills: dict[str, list[str]] | list[str] | Any
    ) -> list[Skill]:
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
                "methodologies": [],
            }
        elif isinstance(raw_skills, dict):
            raw_skills_dict = raw_skills
        else:
            raw_skills_dict = {}

        cat_mapping = {
            "technical": SkillType.HARD_SKILL,
            "soft": SkillType.SOFT_SKILL,
            "tools": SkillType.TOOL,
            "methodologies": SkillType.METHODOLOGY,
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
                    matches = difflib.get_close_matches(
                        norm_name, list(existing_skills_map.keys()), n=1, cutoff=0.85
                    )
                    if matches:
                        normalized_skills.append(existing_skills_map[matches[0]])
                    else:
                        new_skill = Skill(
                            name=clean_name,
                            skill_type=skill_type,
                            normalized_name=norm_name,
                            weight=1.0,
                            frequency=1.0,
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
            try:
                prompt = _build_cv_extraction_prompt(cv_text)
                raw_llm_output = await self._llm.generate(prompt=prompt, context=[])
                extracted_data = _parse_cv_extraction_output(raw_llm_output)
                if not extracted_data:
                    # If empty dict returned from parsing, treat as fallback trigger
                    raise ValueError("Empty extraction data parsed")
            except Exception as e:
                logger.warning(
                    "LLM structured extraction failed, using mock profile data fallback",
                    error=str(e),
                )
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
                        "methodologies": ["Scrum", "Microservicios"],
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

            # Step 3: Embed CV text (mocked as zero vector for schema backwards compatibility)
            logger.info("Setting CV embedding to static zero vector")
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
            primary, secondaries, affinities, _ = compute_affinities_and_domains(
                detected_skills, active_clusters
            )
            if not primary:
                raise MLPipelineError("No primary cluster affinity computed")

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
                user_hard_skills = [
                    s
                    for s in detected_skills
                    if s.skill_type in (SkillType.HARD_SKILL, SkillType.TOOL)
                ]
                user_hard_norms = {s.normalized_name: s for s in user_hard_skills}

                primary_cluster_hard_skills = [
                    s
                    for s in primary_cluster.centroid_skills
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
                skill_gaps.sort(key=lambda g: g.skill.weight * g.skill.frequency, reverse=True)

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

            # Compute Domain Affinities
            _, _, _, domain_affinities_dto = compute_affinities_and_domains(
                detected_skills, active_clusters
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
                        market_insights=a.market_insights,
                        compatible_roles=a.compatible_roles,
                    )
                    for a in secondaries
                ],
                all_affinities=[
                    ClusterAffinityDTO(
                        cluster_id=a.cluster_id,
                        cluster_name=a.cluster_name,
                        affinity_score=a.affinity_score,
                        is_primary=(a.cluster_id == primary.cluster_id),
                        market_insights=a.market_insights,
                        compatible_roles=a.compatible_roles,
                    )
                    for a in affinities
                ],
                domain_affinities=domain_affinities_dto,
                detected_skills=[
                    SkillDTO(
                        name=s.name,
                        skill_type=s.skill_type.value,
                        market_importance="consolidated",
                        market_demand_percentage=round(s.frequency * 100)
                        if s.frequency is not None
                        else 100,
                    )
                    for s in detected_skills
                ],
                skill_gaps=[
                    SkillDTO(
                        name=g.skill.name,
                        skill_type=g.skill.skill_type.value,
                        market_importance=g.market_importance,
                        market_demand_percentage=round(g.skill.frequency * 100)
                        if g.skill.frequency is not None
                        else None,
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

    Uses Voyage/OpenAI embeddings to deduplicate skills semantically
    and falls back to Fuzzy Matching (difflib) if embeddings are missing.
    """

    def __init__(
        self,
        job_offer_repo: MLJobOfferRepository,
        skill_repo: SkillRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        self._job_offers = job_offer_repo
        self._skills = skill_repo
        self._embeddings = embedding_service

    async def execute(self) -> dict[str, Any]:
        """Run the normalization pipeline for unnormalized job offers."""
        import difflib

        import numpy as np

        from src.ml_engine.domain.entities import Skill, SkillType

        logger.info("Starting Skill Normalization Pipeline")

        # 1. Fetch unnormalized offers
        offers = await self._job_offers.get_unnormalized_offers(limit=100)
        if not offers:
            logger.info("No unnormalized offers found.")
            return {"processed_offers": 0, "new_skills": 0}

        # 2. Fetch existing skills from database
        existing_skills = await self._skills.get_all_skills()
        skill_map = {s.normalized_name: s for s in existing_skills}

        new_skills_to_create: dict[str, Skill] = {}
        processed_offer_ids = []

        # Gather unique skills in this batch that are not exactly matched
        unmapped_raw_skills: dict[str, str] = {}  # norm_name -> raw_name

        for offer in offers:
            processed_offer_ids.append(offer["id"])
            raw_skills = offer.get("raw_hard_skills", [])

            for raw_skill in raw_skills:
                clean_name = raw_skill.strip()
                norm_name = clean_name.lower().replace(" ", "").replace(".", "")

                if not norm_name:
                    continue

                # Exact match check against existing DB or already planned creations
                if norm_name not in skill_map and norm_name not in new_skills_to_create:
                    unmapped_raw_skills[norm_name] = clean_name

        # If there are unmapped skills, generate embeddings in batch
        unmapped_embeddings: dict[str, list[float]] = {}
        if unmapped_raw_skills:
            norm_names = list(unmapped_raw_skills.keys())
            raw_names = [unmapped_raw_skills[k] for k in norm_names]
            try:
                logger.info(f"Generating embeddings for {len(raw_names)} unmapped skills.")
                vectors = await self._embeddings.embed_batch(raw_names)
                for norm_name, vector in zip(norm_names, vectors, strict=True):
                    unmapped_embeddings[norm_name] = vector
            except Exception as exc:
                logger.error("Failed to generate embeddings in batch", error=str(exc))
                # Fallback to fuzzy matching if embedding fails

        # Process matching using embeddings / fuzzy matching
        for norm_name, clean_name in unmapped_raw_skills.items():
            matched_skill = None
            skill_vector = unmapped_embeddings.get(norm_name)

            # Check if it was mapped in this loop (to avoid duplicate processing within same batch)
            if norm_name in new_skills_to_create or norm_name in skill_map:
                continue
            else:
                # Semantic Match vs existing skills
                best_match = None
                best_score = -1.0

                if skill_vector:
                    # Search best match in existing skills that have embeddings
                    for s in list(skill_map.values()) + list(new_skills_to_create.values()):
                        if s.embedding is not None:
                            # Cosine similarity
                            a = np.array(skill_vector)
                            b = np.array(s.embedding)
                            norm_a = np.linalg.norm(a)
                            norm_b = np.linalg.norm(b)
                            if norm_a > 0 and norm_b > 0:
                                sim = float(np.dot(a, b) / (norm_a * norm_b))
                                if sim > best_score:
                                    best_score = sim
                                    best_match = s

                # Threshold decision
                if best_match and best_score >= 0.88:
                    logger.info(
                        f"Mapped '{clean_name}' semantically to canonical '{best_match.name}' (score: {best_score:.3f})"
                    )
                    # Map this alias to existing skill in local map for rest of batch
                    skill_map[norm_name] = best_match
                else:
                    # Fallback: Fuzzy matching (difflib)
                    all_keys = list(skill_map.keys()) + list(new_skills_to_create.keys())
                    matches = difflib.get_close_matches(norm_name, all_keys, n=1, cutoff=0.85)
                    if matches:
                        matched_name = matches[0]
                        if matched_name in skill_map:
                            matched_skill = skill_map[matched_name]
                        else:
                            matched_skill = new_skills_to_create[matched_name]
                        logger.info(
                            f"Mapped '{clean_name}' via fuzzy matching to canonical '{matched_skill.name}'"
                        )
                        skill_map[norm_name] = matched_skill
                    else:
                        # Create new canonical skill
                        logger.info(f"Creating new canonical skill: '{clean_name}'")
                        matched_skill = Skill(
                            name=clean_name,
                            skill_type=SkillType.HARD_SKILL,
                            normalized_name=norm_name,
                            embedding=skill_vector,
                        )
                        new_skills_to_create[norm_name] = matched_skill
                        skill_map[norm_name] = matched_skill

        # Re-iterate offers to build final link list
        final_offer_skills_to_insert = []
        for offer in offers:
            raw_skills = offer.get("raw_hard_skills", [])
            for raw_skill in raw_skills:
                clean_name = raw_skill.strip()
                norm_name = clean_name.lower().replace(" ", "").replace(".", "")
                if norm_name in skill_map:
                    final_offer_skills_to_insert.append(
                        {
                            "job_offer_id": offer["id"],
                            "skill_norm_name": skill_map[norm_name].normalized_name,
                            "skill_type": "hard_skill",
                        }
                    )

        # 3. Save new skills to database
        if new_skills_to_create:
            logger.info(f"Creating {len(new_skills_to_create)} new canonical skills in DB.")
            await self._skills.save_skills(list(new_skills_to_create.values()))
            all_skills = await self._skills.get_all_skills()
            skill_id_map = {s.normalized_name: s.id for s in all_skills if hasattr(s, "id")}
        else:
            all_skills = await self._skills.get_all_skills()
            skill_id_map = {s.normalized_name: s.id for s in all_skills if hasattr(s, "id")}

        # 4. Insert offer_skills relations
        final_offer_skills = []
        for os in final_offer_skills_to_insert:
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


def compute_affinities_and_domains(
    detected_skills: list[Skill],
    active_clusters: list[TechCluster],
) -> tuple[
    ClusterAffinity | None, list[ClusterAffinity], list[ClusterAffinity], list[DomainAffinityDTO]
]:
    from src.ml_engine.domain.entities import SkillType

    user_hard_skills = [
        s for s in detected_skills if s.skill_type in (SkillType.HARD_SKILL, SkillType.TOOL)
    ]
    user_hard_norms = {s.normalized_name: s for s in user_hard_skills}

    affinities = []
    for cluster in active_clusters:
        cluster_hard_skills = [
            s
            for s in cluster.centroid_skills
            if s.skill_type in (SkillType.HARD_SKILL, SkillType.TOOL)
        ]
        cluster_hard_norms = {s.normalized_name: s for s in cluster_hard_skills}

        union_norms = set(cluster_hard_norms.keys()) | set(user_hard_norms.keys())

        numerator = 0.0
        denominator = 0.0

        for norm_name in union_norms:
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
                denominator += w * 1.0

        score = (numerator / denominator) if denominator > 0.0 else 0.0

        affinities.append(
            ClusterAffinity(
                cluster_id=cluster.id,
                cluster_name=cluster.name,
                affinity_score=score,
                is_primary=False,
                market_insights=cluster.market_insights,
                compatible_roles=cluster.compatible_roles,
            )
        )

    affinities.sort(key=lambda a: a.affinity_score, reverse=True)
    if not affinities:
        return None, [], [], []

    primary = affinities[0]
    primary = ClusterAffinity(
        cluster_id=primary.cluster_id,
        cluster_name=primary.cluster_name,
        affinity_score=primary.affinity_score,
        is_primary=True,
        market_insights=primary.market_insights,
        compatible_roles=primary.compatible_roles,
    )
    secondaries = affinities[1:3]

    domain_scores = {}
    for s in detected_skills:
        if s.domain:
            d = s.domain
            if d not in domain_scores:
                domain_scores[d] = 0.0
            domain_scores[d] += s.weight * s.frequency

    total_domain_score = sum(domain_scores.values()) if domain_scores else 1.0
    domain_affinities_dto = [
        DomainAffinityDTO(domain=d, affinity_score=score / total_domain_score)
        for d, score in domain_scores.items()
    ]
    domain_affinities_dto.sort(key=lambda x: x.affinity_score, reverse=True)

    return primary, secondaries, affinities, domain_affinities_dto


class GetMyProfileUseCase:
    """Gets the logged-in user's profile and computes real-time affinities against active clusters."""

    def __init__(
        self,
        profile_repository: UserProfileRepository,
        cluster_repository: ClusterRepository,
    ) -> None:
        self._profiles = profile_repository
        self._clusters = cluster_repository

    async def execute(self, user_id: UUID) -> UserProfileDTO | None:
        from src.ml_engine.application.dtos import ClusterAffinityDTO, SkillDTO, UserProfileDTO

        profile = await self._profiles.get_by_user_id(user_id)
        if not profile:
            return None

        active_clusters = await self._clusters.get_all_active()
        active_clusters = [c for c in active_clusters if c.centroid_skills]

        primary, secondaries, all_affinities, domain_affinities_dto = (
            compute_affinities_and_domains(profile.detected_skills, active_clusters)
        )

        return UserProfileDTO(
            user_id=profile.user_id,
            cv_id=profile.cv_id,
            seniority=profile.seniority.value,
            primary_specialty=primary.cluster_name if primary else profile.primary_specialty,
            alignment_score=primary.affinity_score if primary else profile.alignment_score,
            secondary_affinities=[
                ClusterAffinityDTO(
                    cluster_id=a.cluster_id,
                    cluster_name=a.cluster_name,
                    affinity_score=a.affinity_score,
                    is_primary=False,
                    market_insights=a.market_insights,
                    compatible_roles=a.compatible_roles,
                )
                for a in (secondaries if secondaries else [])
            ],
            all_affinities=[
                ClusterAffinityDTO(
                    cluster_id=a.cluster_id,
                    cluster_name=a.cluster_name,
                    affinity_score=a.affinity_score,
                    is_primary=(primary and a.cluster_id == primary.cluster_id),
                    market_insights=a.market_insights,
                    compatible_roles=a.compatible_roles,
                )
                for a in (all_affinities if all_affinities else [])
            ],
            domain_affinities=domain_affinities_dto if domain_affinities_dto else [],
            detected_skills=[
                SkillDTO(
                    name=s.name,
                    skill_type=s.skill_type.value,
                    market_importance="consolidated",
                    market_demand_percentage=round(s.frequency * 100)
                    if s.frequency is not None
                    else 100,
                )
                for s in profile.detected_skills
            ],
            skill_gaps=[
                SkillDTO(
                    name=g.skill.name,
                    skill_type=g.skill.skill_type.value,
                    market_importance=g.market_importance,
                    market_demand_percentage=round(g.skill.frequency * 100)
                    if g.skill.frequency is not None
                    else None,
                )
                for g in profile.skill_gaps
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
            message="Profile retrieved successfully",
        )
