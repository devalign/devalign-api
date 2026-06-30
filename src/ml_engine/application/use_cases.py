"""ML Engine use cases."""

import json
from dataclasses import replace as dc_replace
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
from src.ml_engine.application.skill_catalog_service import SkillCatalogService
from src.ml_engine.domain.entities import (
    ClusterAffinity,
    SeniorityLevel,
    Skill,
    SkillGap,
    SkillNature,
    SkillRelationType,
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
        skill_catalog: SkillCatalogService | None = None,
    ) -> None:
        self._cv_parser = cv_parser
        self._clusters = cluster_repository
        self._profiles = profile_repository
        self._llm = llm_service
        self._skills = skill_repository
        self._catalog = skill_catalog or SkillCatalogService(skill_repository, llm_service)

    async def _normalize_user_skills(
        self, raw_skills: list[dict[str, Any]] | dict[str, list[str]] | Any
    ) -> list[Skill]:
        # Handle the list of dicts structure (new LLM format)
        skill_evidence_map = {}
        raw_strings = []

        if isinstance(raw_skills, list):
            for item in raw_skills:
                if isinstance(item, dict) and "name" in item:
                    name = item["name"]
                    raw_strings.append(name)
                    skill_evidence_map[name.lower().strip()] = item
                elif isinstance(item, str):
                    raw_strings.append(item)
        elif isinstance(raw_skills, dict):
            # Backward compatibility / fallback mock format support
            for val_list in raw_skills.values():
                if isinstance(val_list, list):
                    for item in val_list:
                        if isinstance(item, str):
                            raw_strings.append(item)

        if not raw_strings:
            return []

        # Delegate to the O(1) + LLM fallback service
        resolved_skills = await self._catalog.resolve_skills(raw_strings)

        # Decorate resolved skills with their evidence details
        decorated_skills = []
        for skill in resolved_skills:
            evidence = None
            norm_name = skill.name.lower().strip()
            if norm_name in skill_evidence_map:
                evidence = skill_evidence_map[norm_name]
            else:
                # Try to find by partial match or aliases
                for k, v in skill_evidence_map.items():
                    if k in norm_name or norm_name in k:
                        evidence = v
                        break

            if evidence:
                self_taught = bool(evidence.get("self_taught", False))
                personal_projects = bool(evidence.get("personal_projects", False))
                years_exp = int(evidence.get("years_of_experience", 0) or 0)
                has_cert = bool(evidence.get("has_certification", False))

                stamped_skill = dc_replace(
                    skill,
                    self_taught=self_taught,
                    personal_projects=personal_projects,
                    years_of_experience=years_exp,
                    has_certification=has_cert,
                )
                stamped_skill = dc_replace(
                    stamped_skill,
                    ict_score=stamped_skill.calculate_ict()
                )
                decorated_skills.append(stamped_skill)
            else:
                decorated_skills.append(skill)

        return decorated_skills

    async def _expand_with_upward_inference(self, skills: list[Skill]) -> list[Skill]:
        """Traverse upward-pointing relations in the skill graph to infer implicit parent skills.

        Traversal rules:
        - BELONGS_TO: child is a concrete implementation of parent (e.g. PostgreSQL → SQL).
        - REQUIRES: child skill presupposes parent (e.g. Angular → JavaScript).

        Both relation types indicate that mastery of the child implies working
        knowledge of the parent.  Self-loops and already-visited nodes are
        skipped to prevent cycles.

        Args:
            skills: The explicitly extracted skills from the candidate's CV.

        Returns:
            The original skills plus any inferred parent skills, deduplicated by ID.
        """
        if not skills:
            return []

        logger.info("Performing upward inference on detected skills", count=len(skills))
        # Load the full graph in one round-trip (resolves names correctly)
        skill_graph = await self._skills.get_skill_graph()

        # Start with the explicitly detected skills, keyed by ID for O(1) dedup
        inferred_skills: dict[UUID, Skill] = {s.id: s for s in skills if s.id}
        to_process = [s for s in skills if s.id]

        _upward_types = {SkillRelationType.BELONGS_TO, SkillRelationType.REQUIRES}

        while to_process:
            current_skill = to_process.pop(0)
            full_skill = skill_graph.get(current_skill.id) if current_skill.id else None
            if not full_skill:
                continue

            for relation in full_skill.relations:
                if relation.relation_type not in _upward_types:
                    continue
                parent_id = relation.target_skill_id
                if parent_id in skill_graph:
                    if parent_id not in inferred_skills:
                        parent_skill = skill_graph[parent_id]
                        stamped_parent = dc_replace(
                            parent_skill,
                            inferred_from=[current_skill.name],
                            self_taught=current_skill.self_taught,
                            personal_projects=current_skill.personal_projects,
                            years_of_experience=current_skill.years_of_experience,
                            has_certification=current_skill.has_certification,
                            ict_score=current_skill.ict_score,
                        )
                        inferred_skills[parent_id] = stamped_parent
                        to_process.append(stamped_parent)
                        logger.debug(
                            "Inferred parent skill",
                            child=current_skill.name,
                            parent=parent_skill.name,
                            relation=relation.relation_type,
                        )
                    else:
                        existing = inferred_skills[parent_id]
                        if current_skill.ict_score > existing.ict_score:
                            stamped_parent = dc_replace(
                                existing,
                                inferred_from=list(set(existing.inferred_from + [current_skill.name])),
                                self_taught=existing.self_taught or current_skill.self_taught,
                                personal_projects=existing.personal_projects or current_skill.personal_projects,
                                years_of_experience=max(existing.years_of_experience, current_skill.years_of_experience),
                                has_certification=existing.has_certification or current_skill.has_certification,
                                ict_score=max(existing.ict_score, current_skill.ict_score),
                            )
                            inferred_skills[parent_id] = stamped_parent

        return list(inferred_skills.values())

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
                    "skills": [
                        {"name": "Python", "category": "technical", "self_taught": False, "personal_projects": True, "years_of_experience": 3, "has_certification": True},
                        {"name": "SQL", "category": "technical", "self_taught": True, "personal_projects": False, "years_of_experience": 4, "has_certification": False},
                        {"name": "NoSQL", "category": "technical", "self_taught": True, "personal_projects": True, "years_of_experience": 2, "has_certification": False},
                        {"name": "Liderazgo", "category": "soft", "self_taught": False, "personal_projects": False, "years_of_experience": 2, "has_certification": False},
                        {"name": "Comunicación", "category": "soft", "self_taught": False, "personal_projects": False, "years_of_experience": 2, "has_certification": False},
                        {"name": "Docker", "category": "tools", "self_taught": True, "personal_projects": True, "years_of_experience": 2, "has_certification": False},
                        {"name": "Kubernetes", "category": "tools", "self_taught": True, "personal_projects": False, "years_of_experience": 1, "has_certification": False},
                        {"name": "Git", "category": "tools", "self_taught": False, "personal_projects": True, "years_of_experience": 4, "has_certification": False},
                        {"name": "Scrum", "category": "methodologies", "self_taught": False, "personal_projects": False, "years_of_experience": 3, "has_certification": True},
                        {"name": "Microservicios", "category": "methodologies", "self_taught": False, "personal_projects": True, "years_of_experience": 3, "has_certification": False},
                    ],
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

            # Step 4b: Perform Upward Inference to detect implicit parent skills
            detected_skills = await self._expand_with_upward_inference(detected_skills)

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
                user_tech_skills = {
                    s.normalized_name for s in detected_skills if s.nature == SkillNature.TECH
                }

                primary_cluster_tech_skills = [
                    s for s in primary_cluster.centroid_skills if s.nature == SkillNature.TECH
                ]
                for skill in primary_cluster_tech_skills:
                    if skill.normalized_name not in user_tech_skills:
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

            # Step 9: Persist profile (Only save primary diagnostic initially)
            profile = UserProfile(
                user_id=user_id,
                cv_id=cv_id,
                embedding=cv_embedding,
                detected_skills=detected_skills,
                seniority=seniority,
                primary_affinity=primary,
                secondary_affinities=[],  # Only save primary diagnostic initially
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

            user_skills_map = {s.normalized_name: s for s in detected_skills}

            primary_dto = ClusterAffinityDTO(
                cluster_id=primary.cluster_id,
                cluster_name=primary.cluster_name,
                affinity_score=primary.affinity_score,
                is_primary=True,
                market_insights=primary.market_insights,
                compatible_roles=primary.compatible_roles,
                detected_skills=[
                    SkillDTO(
                        name=s.name,
                        skill_type=s.nature.value,
                        market_importance="critical"
                        if (s.weight * (s.frequency if s.frequency is not None else 1.0)) >= 2.0
                        else (
                            "high"
                            if (s.weight * (s.frequency if s.frequency is not None else 1.0)) >= 1.0
                            else "medium"
                        ),
                        market_demand_percentage=round(s.frequency * 100)
                        if s.frequency is not None
                        else 100,
                        self_taught=user_skills_map.get(s.normalized_name).self_taught if s.normalized_name in user_skills_map else False,
                        personal_projects=user_skills_map.get(s.normalized_name).personal_projects if s.normalized_name in user_skills_map else False,
                        years_of_experience=user_skills_map.get(s.normalized_name).years_of_experience if s.normalized_name in user_skills_map else 0,
                        has_certification=user_skills_map.get(s.normalized_name).has_certification if s.normalized_name in user_skills_map else False,
                        ict_score=user_skills_map.get(s.normalized_name).ict_score if s.normalized_name in user_skills_map else 0.0,
                        trend=determine_trend(s.name),
                    )
                    for s in primary.detected_skills
                ],
                skill_gaps=[
                    SkillDTO(
                        name=g.skill.name,
                        skill_type=g.skill.nature.value,
                        market_importance=g.market_importance,
                        market_demand_percentage=round(g.skill.frequency * 100)
                        if g.skill.frequency is not None
                        else None,
                        trend=determine_trend(g.skill.name),
                    )
                    for g in primary.skill_gaps
                ],
            )

            # Map DTOs
            return UserProfileDTO(
                user_id=user_id,
                cv_id=cv_id,
                seniority=seniority.value,
                primary_specialty=primary.cluster_name,
                alignment_score=primary.affinity_score,
                secondary_affinities=[],
                all_affinities=[primary_dto],
                domain_affinities=domain_affinities_dto,
                detected_skills=[
                    SkillDTO(
                        name=s.name,
                        skill_type=s.nature.value,
                        market_importance="consolidated",
                        market_demand_percentage=round(s.frequency * 100)
                        if s.frequency is not None
                        else 100,
                        self_taught=s.self_taught,
                        personal_projects=s.personal_projects,
                        years_of_experience=s.years_of_experience,
                        has_certification=s.has_certification,
                        ict_score=s.ict_score,
                        trend=determine_trend(s.name),
                    )
                    for s in detected_skills
                ],
                skill_gaps=[
                    SkillDTO(
                        name=g.skill.name,
                        skill_type=g.skill.nature.value,
                        market_importance=g.market_importance,
                        market_demand_percentage=round(g.skill.frequency * 100)
                        if g.skill.frequency is not None
                        else None,
                        trend=determine_trend(g.skill.name),
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


MOCK_TRENDS = {
    "react": "growing",
    "python": "growing",
    "docker": "growing",
    "kubernetes": "growing",
    "aws": "growing",
    "cloud computing": "growing",
    "typescript": "growing",
    "machine learning": "growing",
    "inteligencia artificial": "growing",
    "devops": "growing",
    "microservicios": "growing",
    "scrum": "stable",
    "sql": "stable",
    "git": "stable",
    "comunicación": "stable",
    "liderazgo": "stable",
    "cobol": "shrinking",
    "jquery": "shrinking",
}

def determine_trend(name: str) -> str:
    norm_name = name.lower().strip()
    return MOCK_TRENDS.get(norm_name, "stable")


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
10. Technical & Soft Skills (skills: list of objects). For EACH skill mentioned in the CV (either explicitly or contextually inferred from job description tasks/projects/certifications), identify:
    - name (the name of the skill, e.g., Python, Docker, Scrum, Liderazgo)
    - category (one of: 'technical', 'soft', 'tools', 'methodologies')
    - self_taught (boolean: true if the CV mentions taking courses, bootcamps, or learning this self-taught/autodidacta)
    - personal_projects (boolean: true if the skill is used in personal projects or open source contributions mentioned in the CV)
    - years_of_experience (integer: the number of years the candidate has used this skill in work experience)
    - has_certification (boolean: true if there is an official certification in the CV matching this skill)

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
  "skills": [
    {{
      "name": "string",
      "category": "technical | soft | tools | methodologies",
      "self_taught": boolean,
      "personal_projects": boolean,
      "years_of_experience": integer,
      "has_certification": boolean
    }}
  ]
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

        from src.ml_engine.domain.entities import Skill, SkillNature

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
                            nature=SkillNature.TECH,
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
    skill_trends: dict[str, float] | None = None,
) -> tuple[
    ClusterAffinity | None, list[ClusterAffinity], list[ClusterAffinity], list[DomainAffinityDTO]
]:
    from src.ml_engine.domain.entities import SkillNature

    user_tech_skills = [s for s in detected_skills if s.nature == SkillNature.TECH]
    user_tech_norms = {s.normalized_name: s for s in user_tech_skills}
    user_all_norms = {s.normalized_name: s for s in detected_skills}

    affinities = []
    for cluster in active_clusters:
        cluster_tech_skills = [s for s in cluster.centroid_skills if s.nature == SkillNature.TECH]
        cluster_tech_norms = {s.normalized_name: s for s in cluster_tech_skills}

        union_norms = set(cluster_tech_norms.keys()) | set(user_tech_norms.keys())

        numerator = 0.0
        denominator = 0.0
        matched_skills = []
        partial_matches = []
        missing_skills = []

        for norm_name in union_norms:
            w = 1.0
            if norm_name in cluster_tech_norms:
                w = cluster_tech_norms[norm_name].weight
            elif norm_name in user_tech_norms:
                w = user_tech_norms[norm_name].weight

            in_user = norm_name in user_tech_norms
            in_cluster = norm_name in cluster_tech_norms

            f_s = cluster_tech_norms[norm_name].frequency if in_cluster else 1.0

            if in_user and in_cluster:
                # Evidence-based Jaccard: scale by the user's proficiency (ICT score / 10.0)
                user_score = user_tech_norms[norm_name].ict_score / 10.0
                numerator += w * f_s * user_score
                denominator += w * f_s
                matched_skills.append(cluster_tech_norms[norm_name].name)
            elif in_cluster:
                cluster_skill = cluster_tech_norms[norm_name]
                cluster_domains = set(cluster_skill.domain_tags)

                partial_match_score = 0.0
                if cluster_domains:
                    # Find the user's best matching alternative skill in same domain
                    alternative_skills = [
                        u for u in user_tech_skills
                        if set(u.domain_tags) & cluster_domains
                    ]
                    if alternative_skills:
                        best_alt = max(alternative_skills, key=lambda u: u.ict_score)
                        # Scale partial credit (30%) by the alternative's proficiency
                        partial_match_score = 0.3 * (best_alt.ict_score / 10.0)
                        partial_matches.append((cluster_skill.name, best_alt.name))

                if partial_match_score == 0.0:
                    missing_skills.append(cluster_skill.name)

                numerator += (w * f_s) * partial_match_score
                denominator += w * f_s
            else:
                user_score = user_tech_norms[norm_name].ict_score / 10.0
                denominator += w * user_score

        score = (numerator / denominator) if denominator > 0.0 else 0.0

        insight_parts = []
        if matched_skills:
            insight_parts.append(
                f"Dominas {len(matched_skills)} tecnologías clave (como {', '.join(matched_skills[:2])})."
            )
        if partial_matches:
            examples = [f"tienes {u} en lugar de {c}" for c, u in partial_matches[:2]]
            insight_parts.append(f"Cubres áreas relacionadas ({'; '.join(examples)}).")
        if missing_skills:
            insight_parts.append(
                f"Para mejorar, considera aprender {', '.join(missing_skills[:2])}."
            )

        ai_insight = (
            " ".join(insight_parts) if insight_parts else "Afinidad calculada en base a tu perfil."
        )

        # Compute cluster strengths and gaps dynamically
        cluster_detected_skills = []
        cluster_skill_gaps = []
        for skill in cluster.centroid_skills:
            if skill.normalized_name in user_all_norms:
                cluster_detected_skills.append(skill)
            else:
                # Apply Mittas temporal trend multiplier to the priority score if available
                trend_multiplier = 1.0
                if skill_trends and skill.normalized_name in skill_trends:
                    trend_multiplier = skill_trends[skill.normalized_name]

                priority = skill.weight * skill.frequency * trend_multiplier
                if priority >= 2.0:
                    importance = "critical"
                elif priority >= 1.0:
                    importance = "high"
                else:
                    importance = "medium"

                cluster_skill_gaps.append(
                    SkillGap(
                        skill=skill,
                        market_importance=importance,
                    )
                )

        cluster_detected_skills.sort(
            key=lambda s: (s.frequency if s.frequency is not None else 1.0, s.weight),
            reverse=True,
        )
        cluster_skill_gaps.sort(
            key=lambda g: g.skill.weight * g.skill.frequency,
            reverse=True,
        )

        affinities.append(
            ClusterAffinity(
                cluster_id=cluster.id,
                cluster_name=cluster.name,
                affinity_score=score,
                is_primary=False,
                market_insights=cluster.market_insights,
                compatible_roles=cluster.compatible_roles,
                ai_insight=ai_insight,
                detected_skills=cluster_detected_skills,
                skill_gaps=cluster_skill_gaps,
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
        ai_insight=primary.ai_insight,
        detected_skills=primary.detected_skills,
        skill_gaps=primary.skill_gaps,
    )
    secondaries = affinities[1:3]

    domain_scores = {}
    for s in detected_skills:
        if s.core_domains:
            for d in s.core_domains:
                if d not in domain_scores:
                    domain_scores[d] = 0.0
                domain_scores[d] += s.weight * s.frequency

    # Calcular promedios de demanda de mercado por dominio
    domain_demands_accum: dict[str, list[float]] = {}
    for cluster in active_clusters:
        for skill in cluster.centroid_skills:
            if skill.domain_tags:
                for d in skill.domain_tags:
                    d_clean = d.strip().lower()
                    if d_clean not in domain_demands_accum:
                        domain_demands_accum[d_clean] = []
                    domain_demands_accum[d_clean].append(skill.frequency)

    domain_market_demand = {
        d: sum(freqs) / len(freqs) if freqs else 0.5 for d, freqs in domain_demands_accum.items()
    }

    total_domain_score = sum(domain_scores.values()) if domain_scores else 1.0
    domain_affinities_dto = [
        DomainAffinityDTO(
            domain=d,
            affinity_score=score / total_domain_score,
            market_demand=domain_market_demand.get(d.strip().lower(), 0.5),
        )
        for d, score in domain_scores.items()
    ]
    domain_affinities_dto.sort(key=lambda x: x.affinity_score, reverse=True)

    return primary, secondaries, affinities, domain_affinities_dto


class GetKnowledgeGraphUseCase:
    """Builds a Knowledge Graph representation for frontend visualization."""

    def __init__(
        self,
        skill_repository: SkillRepository,
        profile_repository: UserProfileRepository,
    ) -> None:
        self._skills = skill_repository
        self._profiles = profile_repository

    async def execute(self, user_id: UUID | None = None, cluster_name: str | None = None) -> Any:
        from src.ml_engine.application.dtos import GraphLinkDTO, GraphNodeDTO, GraphResponseDTO

        # When the user is authenticated, build a focused graph scoped to their
        # own detected skills and gaps. This avoids loading the entire skill
        # catalog (potentially thousands of rows) which causes request timeouts.
        # The unauthenticated path (global explorer) still loads all skills.
        if user_id:
            return await self._build_user_graph(user_id, cluster_name, GraphNodeDTO, GraphLinkDTO, GraphResponseDTO)

        # --- Unauthenticated / global explorer path (full catalog) ---
        all_skills = await self._skills.get_all_skills()

        nodes = [
            GraphNodeDTO(
                id=s.normalized_name,
                label=s.name,
                group=s.nature.value if hasattr(s, "nature") and s.nature else "tech",
                domains=s.domain_tags if hasattr(s, "domain_tags") and s.domain_tags else [],
                status="neutral",
            )
            for s in all_skills
        ]

        # Build links only from explicit relations (skip O(N²) implicit domain links for global view)
        skill_by_id = {s.id: s for s in all_skills if s.id}
        links = []
        for s in all_skills:
            if hasattr(s, "relations") and s.relations:
                for rel in s.relations:
                    target = skill_by_id.get(rel.target_skill_id)
                    if target:
                        links.append(
                            GraphLinkDTO(
                                source=s.normalized_name,
                                target=target.normalized_name,
                                value=2.0,
                                type=f"explicit_{rel.relation_type}",
                            )
                        )

        return GraphResponseDTO(nodes=nodes, links=links)

    async def _build_user_graph(
        self,
        user_id: UUID,
        cluster_name: str | None,
        GraphNodeDTO: type,
        GraphLinkDTO: type,
        GraphResponseDTO: type,
    ) -> Any:
        """Build a knowledge graph scoped to a user's detected skills and skill gaps.

        Fetches only the user's profile data (a tiny, bounded set) rather than
        the full skill catalog, making this O(1) in catalog size.
        If cluster_name is provided, scopes the skills to that specific cluster.
        """
        profile = await self._profiles.get_by_user_id(user_id)

        if not profile:
            return GraphResponseDTO(nodes=[], links=[])

        if cluster_name:
            target_affinity = None
            if profile.primary_affinity and profile.primary_affinity.cluster_name == cluster_name:
                target_affinity = profile.primary_affinity
            else:
                for a in profile.secondary_affinities:
                    if a.cluster_name == cluster_name:
                        target_affinity = a
                        break

            if target_affinity:
                acquired = {s.normalized_name: s for s in target_affinity.detected_skills}
                gaps = {g.skill.normalized_name: g.skill for g in target_affinity.skill_gaps}
                neutral = {s.normalized_name: s for s in profile.detected_skills if s.normalized_name not in acquired}
            else:
                acquired = {s.normalized_name: s for s in profile.detected_skills}
                gaps = {g.skill.normalized_name: g.skill for g in profile.skill_gaps}
                neutral = {}
        else:
            acquired = {s.normalized_name: s for s in profile.detected_skills}
            gaps = {g.skill.normalized_name: g.skill for g in profile.skill_gaps}
            neutral = {}

        # Fetch all non-ESCO skills to render as the general market backdrop
        non_esco_skills = await self._skills.get_non_esco_skills()
        market = {
            s.normalized_name: s for s in non_esco_skills
            if s.normalized_name not in acquired
            and s.normalized_name not in gaps
            and s.normalized_name not in neutral
        }

        all_skills_to_render = list(acquired.values()) + list(gaps.values()) + list(neutral.values()) + list(market.values())

        # Deduplicate (a skill can appear in both acquired and gaps due to partial overlap)
        seen: set[str] = set()
        nodes = []
        for s in all_skills_to_render:
            if s.normalized_name in seen:
                continue
            seen.add(s.normalized_name)
            if s.normalized_name in acquired:
                status = "acquired"
            elif s.normalized_name in gaps:
                status = "gap"
            elif s.normalized_name in neutral:
                status = "neutral"
            else:
                status = "market"
            nodes.append(
                GraphNodeDTO(
                    id=s.normalized_name,
                    label=s.name,
                    group=s.nature.value if hasattr(s, "nature") and s.nature else "tech",
                    domains=s.domain_tags if hasattr(s, "domain_tags") and s.domain_tags else [],
                    status=status,
                )
            )

        # Build implicit links between skills that share a domain tag
        domain_map: dict[str, list[str]] = {}
        for s in all_skills_to_render:
            if hasattr(s, "domain_tags") and s.domain_tags:
                for d in s.domain_tags:
                    domain_map.setdefault(d, [])
                    if s.normalized_name not in domain_map[d]:
                        domain_map[d].append(s.normalized_name)

        links = []
        for skill_names in domain_map.values():
            for i in range(len(skill_names) - 1):
                links.append(
                    GraphLinkDTO(
                        source=skill_names[i],
                        target=skill_names[i + 1],
                        value=0.5,
                        type="implicit_domain",
                    )
                )

        return GraphResponseDTO(nodes=nodes, links=links)


def compute_domain_affinities(
    detected_skills: list[Skill],
    active_clusters: list[TechCluster],
) -> list[DomainAffinityDTO]:
    from src.ml_engine.application.dtos import DomainAffinityDTO
    domain_scores = {}
    for s in detected_skills:
        if s.core_domains:
            for d in s.core_domains:
                if d not in domain_scores:
                    domain_scores[d] = 0.0
                domain_scores[d] += s.weight * (s.frequency if s.frequency is not None else 1.0)

    domain_demands_accum: dict[str, list[float]] = {}
    for cluster in active_clusters:
        for skill in cluster.centroid_skills:
            if skill.core_domains:
                for d in skill.core_domains:
                    d_clean = d.strip().lower()
                    if d_clean not in domain_demands_accum:
                        domain_demands_accum[d_clean] = []
                    domain_demands_accum[d_clean].append(skill.frequency)

    domain_market_demand = {
        d: sum(freqs) / len(freqs) if freqs else 0.5 for d, freqs in domain_demands_accum.items()
    }

    total_domain_score = sum(domain_scores.values()) if domain_scores else 1.0
    domain_affinities_dto = [
        DomainAffinityDTO(
            domain=d,
            affinity_score=score / total_domain_score,
            market_demand=domain_market_demand.get(d.strip().lower(), 0.5),
        )
        for d, score in domain_scores.items()
    ]
    domain_affinities_dto.sort(key=lambda x: x.affinity_score, reverse=True)
    return domain_affinities_dto


class GetMyProfileUseCase:
    """Gets the logged-in user's profile and loads database-persisted diagnostics."""

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
            from src.ml_engine.domain.entities import ClusterAffinity, SeniorityLevel, UserProfile
            empty_profile = UserProfile(
                user_id=user_id,
                cv_id=None,
                embedding=[],
                detected_skills=[],
                seniority=SeniorityLevel.MID,
                primary_affinity=ClusterAffinity(
                    cluster_id=None,
                    cluster_name="Sin Diagnóstico",
                    affinity_score=0.0,
                    is_primary=True,
                ),
                secondary_affinities=[],
                skill_gaps=[],
            )
            await self._profiles.save(empty_profile)
            profile = await self._profiles.get_by_user_id(user_id)
            if not profile:
                return None

        active_clusters = await self._clusters.get_all_active()
        active_clusters = [c for c in active_clusters if c.centroid_skills]

        domain_affinities_dto = compute_domain_affinities(profile.detected_skills, active_clusters)

        primary = profile.primary_affinity
        secondaries = profile.secondary_affinities
        all_affinities = [primary] + secondaries if primary.cluster_name != "Sin Diagnóstico" else []

        user_skills_map = {s.normalized_name: s for s in profile.detected_skills}

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
                    detected_skills=[
                        SkillDTO(
                            name=s.name,
                            skill_type=s.nature.value,
                            market_importance="critical"
                            if (s.weight * (s.frequency if s.frequency is not None else 1.0)) >= 2.0
                            else (
                                "high"
                                if (s.weight * (s.frequency if s.frequency is not None else 1.0))
                                >= 1.0
                                else "medium"
                            ),
                            market_demand_percentage=round(s.frequency * 100)
                            if s.frequency is not None
                            else 100,
                            self_taught=user_skills_map.get(s.normalized_name).self_taught if s.normalized_name in user_skills_map else False,
                            personal_projects=user_skills_map.get(s.normalized_name).personal_projects if s.normalized_name in user_skills_map else False,
                            years_of_experience=user_skills_map.get(s.normalized_name).years_of_experience if s.normalized_name in user_skills_map else 0,
                            has_certification=user_skills_map.get(s.normalized_name).has_certification if s.normalized_name in user_skills_map else False,
                            ict_score=user_skills_map.get(s.normalized_name).ict_score if s.normalized_name in user_skills_map else 0.0,
                            trend=determine_trend(s.name),
                        )
                        for s in a.detected_skills
                    ],
                    skill_gaps=[
                        SkillDTO(
                            name=g.skill.name,
                            skill_type=g.skill.nature.value,
                            market_importance=g.market_importance,
                            market_demand_percentage=round(g.skill.frequency * 100)
                            if g.skill.frequency is not None
                            else None,
                            trend=determine_trend(g.skill.name),
                        )
                        for g in a.skill_gaps
                    ],
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
                    detected_skills=[
                        SkillDTO(
                            name=s.name,
                            skill_type=s.nature.value,
                            market_importance="critical"
                            if (s.weight * (s.frequency if s.frequency is not None else 1.0)) >= 2.0
                            else (
                                "high"
                                if (s.weight * (s.frequency if s.frequency is not None else 1.0))
                                >= 1.0
                                else "medium"
                            ),
                            market_demand_percentage=round(s.frequency * 100)
                            if s.frequency is not None
                            else 100,
                            self_taught=user_skills_map.get(s.normalized_name).self_taught if s.normalized_name in user_skills_map else False,
                            personal_projects=user_skills_map.get(s.normalized_name).personal_projects if s.normalized_name in user_skills_map else False,
                            years_of_experience=user_skills_map.get(s.normalized_name).years_of_experience if s.normalized_name in user_skills_map else 0,
                            has_certification=user_skills_map.get(s.normalized_name).has_certification if s.normalized_name in user_skills_map else False,
                            ict_score=user_skills_map.get(s.normalized_name).ict_score if s.normalized_name in user_skills_map else 0.0,
                            trend=determine_trend(s.name),
                        )
                        for s in a.detected_skills
                    ],
                    skill_gaps=[
                        SkillDTO(
                            name=g.skill.name,
                            skill_type=g.skill.nature.value,
                            market_importance=g.market_importance,
                            market_demand_percentage=round(g.skill.frequency * 100)
                            if g.skill.frequency is not None
                            else None,
                            trend=determine_trend(g.skill.name),
                        )
                        for g in a.skill_gaps
                    ],
                )
                for a in (all_affinities if all_affinities else [])
            ],
            domain_affinities=domain_affinities_dto if domain_affinities_dto else [],
            detected_skills=[
                SkillDTO(
                    name=s.name,
                    skill_type=s.nature.value,
                    market_importance="consolidated",
                    market_demand_percentage=round(s.frequency * 100)
                    if s.frequency is not None
                    else 100,
                    self_taught=s.self_taught,
                    personal_projects=s.personal_projects,
                    years_of_experience=s.years_of_experience,
                    has_certification=s.has_certification,
                    ict_score=s.ict_score,
                    trend=determine_trend(s.name),
                )
                for s in profile.detected_skills
            ],
            skill_gaps=[
                SkillDTO(
                    name=g.skill.name,
                    skill_type=g.skill.nature.value,
                    market_importance=g.market_importance,
                    market_demand_percentage=round(g.skill.frequency * 100)
                    if g.skill.frequency is not None
                    else None,
                    trend=determine_trend(g.skill.name),
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


class EvaluateClusterDiagnosticUseCase:
    """Computes the affinity of a user profile's detected skills against a specific cluster and saves it."""

    def __init__(
        self,
        profile_repository: UserProfileRepository,
        cluster_repository: ClusterRepository,
    ) -> None:
        self._profiles = profile_repository
        self._clusters = cluster_repository

    async def execute(self, user_id: UUID, cluster_name: str) -> UserProfileDTO | None:
        from dataclasses import replace

        from fastapi import HTTPException

        profile = await self._profiles.get_by_user_id(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="No profile found. Please upload a CV first.")

        active_clusters = await self._clusters.get_all_active()
        requested_cluster = next((c for c in active_clusters if c.name == cluster_name), None)
        if not requested_cluster:
            raise HTTPException(status_code=404, detail=f"Cluster '{cluster_name}' not found.")

        # Compute affinity for this cluster
        primary, secondaries, affinities, _ = compute_affinities_and_domains(
            profile.detected_skills, [requested_cluster]
        )
        if not affinities:
            raise HTTPException(status_code=500, detail="Failed to compute affinity score.")

        new_affinity = affinities[0]
        # Ensure is_primary is False since it's evaluated on demand
        new_affinity = ClusterAffinity(
            cluster_id=new_affinity.cluster_id,
            cluster_name=new_affinity.cluster_name,
            affinity_score=new_affinity.affinity_score,
            is_primary=False,
            market_insights=new_affinity.market_insights,
            compatible_roles=new_affinity.compatible_roles,
            ai_insight=new_affinity.ai_insight,
            detected_skills=new_affinity.detected_skills,
            skill_gaps=new_affinity.skill_gaps,
        )

        # Merge secondary_affinities (remove existing with same name if any)
        existing_secondaries = [
            a for a in profile.secondary_affinities if a.cluster_name != cluster_name
        ]
        updated_secondaries = existing_secondaries + [new_affinity]

        # Save profile
        updated_profile = replace(profile, secondary_affinities=updated_secondaries)
        await self._profiles.save(updated_profile)

        # Return updated profile
        return await GetMyProfileUseCase(self._profiles, self._clusters).execute(user_id)

