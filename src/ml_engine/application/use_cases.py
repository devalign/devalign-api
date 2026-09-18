"""ML Engine use cases."""

import json
from collections import deque
from dataclasses import replace as dc_replace
from typing import Any
from uuid import UUID, uuid4

import structlog

from src.ml_engine.application.dtos import (
    ClusterAffinityDTO,
    ClusterDTO,
    DiagnosticDetailDTO,
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

# ---------------------------------------------------------------------------
# CV content validation — keyword-based heuristic to detect if a document
# actually looks like a professional CV/resume before sending to the LLM.
# ---------------------------------------------------------------------------
_CV_SECTION_KEYWORDS: frozenset[str] = frozenset(
    {
        "experience",
        "experiencia",
        "work history",
        "historial laboral",
        "education",
        "educación",
        "formación académica",
        "skills",
        "habilidades",
        "competencias",
        "professional summary",
        "resumen profesional",
        "perfil profesional",
        "employment",
        "empleo",
        "trayectoria",
        "projects",
        "proyectos",
        "certifications",
        "certificaciones",
        "languages",
        "idiomas",
        "references",
        "referencias",
        "objective",
        "objetivo profesional",
        "work experience",
        "laboral",
    }
)


CONCEPT_PATTERNS: tuple[str, ...] = (
    "back end",
    "front end",
    "full stack",
    "agile",
    "configuration management",
    "software engineering",
    "software development",
    "cloud computing",
    "web development",
    "object-oriented",
)


def is_concept_skill(skill: Any) -> bool:
    """Check if a skill is an abstract/umbrella conceptual skill rather than an actionable tool."""
    if not skill:
        return False
    if hasattr(skill, "nature") and skill.nature == SkillNature.CONCEPT:
        return True
    s_name = (getattr(skill, "name", "") or "").lower().strip()
    return any(pat in s_name for pat in CONCEPT_PATTERNS)


def _looks_like_a_cv(text: str) -> bool:
    """Quick heuristic: does the text contain CV-like section headers?"""
    text_lower = text.lower()
    section_matches = sum(1 for kw in _CV_SECTION_KEYWORDS if kw in text_lower)
    return section_matches >= 2


def _build_cv_classification_prompt(text: str) -> str:
    """Lightweight LLM prompt to classify whether text is a CV/resume."""
    return f"""You are a document classifier. Determine if the following text is a professional CV/resume (currículum vitae).

A CV/resume typically contains:
- Personal information (name, contact details)
- Work experience (companies, roles, dates)
- Education history
- Technical and soft skills
- Professional summary or objective

Respond with ONLY a valid JSON object with exactly two fields:
{{"is_cv": true/false, "confidence": 0.0-1.0}}

Text:
{text[:3000]}"""


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

    async def _classify_as_cv(self, text: str) -> tuple[bool, float]:
        """Determine if the extracted text is actually a CV/resume.

        Uses a fast heuristic only. The structured extraction step already has
        an explicit ``not_a_cv`` guard, so we avoid a second LLM round-trip on
        the hot path.

        Returns (is_cv, confidence).
        """
        if _looks_like_a_cv(text):
            logger.debug("CV heuristic passed — document looks like a CV")
            return True, 0.8

        logger.info("CV heuristic inconclusive, continuing with structured extraction")
        return True, 0.5

    async def _normalize_user_skills(
        self,
        raw_skills: list[dict[str, Any]] | dict[str, list[str]] | Any,
        use_llm_fallback: bool = True,
        existing_skills_cache: list[Skill] | None = None,
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

        # Delegate to the O(1) + optional LLM fallback service
        resolved_skills = await self._catalog.resolve_skills(
            raw_strings,
            use_llm_fallback=use_llm_fallback,
            existing_skills_cache=existing_skills_cache,
        )

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
                is_custom = bool(evidence.get("is_custom", getattr(skill, "is_custom", False)))

                stamped_skill = dc_replace(
                    skill,
                    self_taught=self_taught,
                    personal_projects=personal_projects,
                    years_of_experience=years_exp,
                    has_certification=has_cert,
                    is_custom=is_custom,
                )
                stamped_skill = dc_replace(stamped_skill, ict_score=stamped_skill.calculate_ict())
                decorated_skills.append(stamped_skill)
            else:
                decorated_skills.append(skill)

        return decorated_skills

    async def _expand_with_upward_inference(
        self,
        skills: list[Skill],
        skill_graph: dict[UUID, Skill] | None = None,
        all_skills: list[Skill] | None = None,
    ) -> list[Skill]:
        """Traverse upward-pointing relations in the skill graph to infer implicit parent skills.

        Traversal rules:
        - BELONGS_TO: child is a concrete implementation of parent (e.g. PostgreSQL → SQL).
        - REQUIRES: child skill presupposes parent (e.g. Angular → JavaScript).

        Both relation types indicate that mastery of the child implies working
        knowledge of the parent.  Self-loops and already-visited nodes are
        skipped to prevent cycles.

        Args:
            skills: The explicitly extracted skills from the candidate's CV.
            skill_graph: Optional pre-built graph (bypasses DB round-trip).
            all_skills: Optional pre-loaded skill list (builds graph from it,
                bypassing the DB round-trip).  Takes precedence over skill_graph.

        Returns:
            The original skills plus any inferred parent skills, deduplicated by ID.
        """
        if not skills:
            return []

        logger.info("Performing upward inference on detected skills", count=len(skills))
        if all_skills is not None:
            skill_graph = {s.id: s for s in all_skills if s.id}
        elif skill_graph is None:
            skill_graph = await self._skills.get_skill_graph()

        # build name_to_skill map for concept/domain mapping
        name_to_skill: dict[str, Skill] = {}
        for s in skill_graph.values():
            if s.name:
                name_to_skill[s.name.lower()] = s
            if s.normalized_name:
                name_to_skill[s.normalized_name.lower()] = s

        # Start with the explicitly detected skills, keyed by ID for O(1) dedup
        inferred_skills: dict[UUID, Skill] = {s.id: s for s in skills if s.id}
        to_process: deque[Skill] = deque(s for s in skills if s.id)

        _upward_types = {SkillRelationType.BELONGS_TO, SkillRelationType.REQUIRES}

        while to_process:
            current_skill = to_process.popleft()
            full_skill = skill_graph.get(current_skill.id) if current_skill.id else None
            if not full_skill:
                continue

            # 1. Standard upward relation inference
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
                                inferred_from=list(
                                    set([*existing.inferred_from, current_skill.name])
                                ),
                                self_taught=existing.self_taught or current_skill.self_taught,
                                personal_projects=existing.personal_projects
                                or current_skill.personal_projects,
                                years_of_experience=max(
                                    existing.years_of_experience, current_skill.years_of_experience
                                ),
                                has_certification=existing.has_certification
                                or current_skill.has_certification,
                                ict_score=max(existing.ict_score, current_skill.ict_score),
                            )
                            inferred_skills[parent_id] = stamped_parent

            # 2. Dynamic inference from core_domains and domain_tags
            # Only infer concepts to avoid over-matching tech skills
            domains_to_check: set[str] = set()
            if full_skill.core_domains:
                domains_to_check.update(d.lower() for d in full_skill.core_domains)
            if full_skill.domain_tags:
                domains_to_check.update(t.lower() for t in full_skill.domain_tags)

            for domain_name in domains_to_check:
                domain_parent_skill = name_to_skill.get(domain_name)
                if domain_parent_skill and domain_parent_skill.nature == SkillNature.CONCEPT:
                    domain_parent_id = domain_parent_skill.id
                    if domain_parent_id:
                        if domain_parent_id not in inferred_skills:
                            stamped_parent = dc_replace(
                                domain_parent_skill,
                                inferred_from=[current_skill.name],
                                self_taught=current_skill.self_taught,
                                personal_projects=current_skill.personal_projects,
                                years_of_experience=current_skill.years_of_experience,
                                has_certification=current_skill.has_certification,
                                ict_score=current_skill.ict_score,
                            )
                            inferred_skills[domain_parent_id] = stamped_parent
                            to_process.append(stamped_parent)
                            logger.debug(
                                "Inferred concept from core_domains/domain_tags",
                                child=current_skill.name,
                                parent=domain_parent_skill.name,
                            )
                        else:
                            existing = inferred_skills[domain_parent_id]
                            if current_skill.ict_score > existing.ict_score:
                                stamped_parent = dc_replace(
                                    existing,
                                    inferred_from=list(
                                        set([*existing.inferred_from, current_skill.name])
                                    ),
                                    self_taught=existing.self_taught or current_skill.self_taught,
                                    personal_projects=existing.personal_projects
                                    or current_skill.personal_projects,
                                    years_of_experience=max(
                                        existing.years_of_experience,
                                        current_skill.years_of_experience,
                                    ),
                                    has_certification=existing.has_certification
                                    or current_skill.has_certification,
                                    ict_score=max(existing.ict_score, current_skill.ict_score),
                                )
                                inferred_skills[domain_parent_id] = stamped_parent

        return list(inferred_skills.values())

    async def _combined_llm_extraction(self, cv_text: str) -> dict[str, Any]:
        """Single combined LLM call for full CV extraction.

        Extracts: personal info, skills with evidence, work experience,
        education, and certifications in one prompt.

        Returns the parsed JSON dict from the LLM.
        """
        logger.info("Running combined LLM extraction")
        cv_text_char_limit = 6000
        cv_text_for_llm = cv_text[:cv_text_char_limit]
        prompt = _build_combined_cv_extraction_prompt(cv_text_for_llm)
        raw_output = await self._llm.generate(prompt=prompt, context=[], max_tokens=3000)
        parsed = _parse_cv_extraction_output(raw_output)
        if not parsed:
            raise ValueError("Empty extraction data parsed")
        if "error" in parsed and parsed["error"] == "not_a_cv":
            doc_type = parsed.get("document_type", "unknown")
            raise MLPipelineError(
                f"The uploaded document does not appear to be a CV/resume "
                f"(detected as: {doc_type}). "
                "Please upload a document with your work experience, education, and skills."
            )
        return parsed

    async def extract_skills(
        self,
        user_id: UUID,
        cv_id: UUID,
        cv_content: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        """New Phase 1: Extract text, classify CV, run combined LLM extraction.

        No profile is saved here — extracted data is stored on CVDocument
        for the frontend to read and requires user validation before persisting.

        Returns a dict with:
            cv_text, extracted_data
        """
        logger.info("Extracting CV text", user_id=str(user_id))
        cv_text = await self._cv_parser.extract_text(cv_content, content_type)

        if not cv_text.strip():
            raise MLPipelineError("CV text extraction returned empty content")

        is_cv, confidence = await self._classify_as_cv(cv_text)
        if not is_cv:
            raise MLPipelineError(
                "The uploaded document does not appear to be a professional CV/resume. "
                "Please upload a document with your work experience, education, and skills."
            )
        logger.debug("Document classified as CV", confidence=confidence)

        extracted_data = await self._combined_llm_extraction(cv_text)

        # Pre-normalize extracted skills against Lightcast catalog (Phase 1)
        if "skills" in extracted_data and isinstance(extracted_data["skills"], list):
            try:
                extracted_data["skills"] = await self._catalog.normalize_extracted_skills(
                    extracted_data["skills"]
                )
                logger.info(
                    "Pre-normalized extracted skills against catalog",
                    count=len(extracted_data["skills"]),
                )
            except Exception as exc:
                logger.warning("Failed to pre-normalize skills against catalog", error=str(exc))

        return {
            "cv_text": cv_text,
            "extracted_data": extracted_data,
        }

    async def finalize_diagnosis(
        self,
        user_id: UUID,
        cv_id: UUID,
        validated_skills: list[SkillDTO] | None = None,
    ) -> UserProfileDTO:
        """Phase 2: Normalize skills, compute Weighted Jaccard affinity, detect gaps, persist diagnosis.

        Reads the existing profile from Phase 1 (saved by ``extract_skills``),
        optionally updates skills if ``validated_skills`` is provided, then runs
        the full diagnosis pipeline: catalog resolution, upward inference,
        cluster affinity computation, gap detection, and persistence with
        ``is_diagnosed=True``.

        Returns the complete ``UserProfileDTO``.
        """
        # Read existing profile from Phase 1
        profile = await self._profiles.get_by_user_id(user_id)
        if not profile:
            raise MLPipelineError("No profile found. Please upload a CV first.")

        seniority = profile.seniority

        # If validated_skills provided, convert to raw_skills format
        if validated_skills is not None:
            raw_skills = [
                {
                    "name": s.name,
                    "category": s.skill_type,
                    "years_of_experience": s.years_of_experience or 0,
                    "personal_projects": s.personal_projects or False,
                    "has_certification": s.has_certification or False,
                    "is_custom": getattr(s, "is_custom", False),
                }
                for s in validated_skills
            ]
        else:
            # Use whatever raw data is available from the existing profile
            extracted_data_skills = []
            for s in profile.detected_skills:
                extracted_data_skills.append(
                    {
                        "name": s.name,
                        "category": s.nature.value if s.nature else "technical",
                        "years_of_experience": s.years_of_experience or 0,
                        "personal_projects": s.personal_projects or False,
                        "has_certification": s.has_certification or False,
                        "is_custom": getattr(s, "is_custom", False),
                    }
                )
            raw_skills = extracted_data_skills if extracted_data_skills else []

        # Embedding (static zero-vector for backwards compatibility)
        cv_embedding = [0.0] * 1024

        # Normalize user skills against the canonical catalog
        logger.info("Phase 2 — normalising skills", user_id=str(user_id))
        all_skills = await self._skills.get_all_skills()
        detected_skills = await self._normalize_user_skills(
            raw_skills,
            use_llm_fallback=False,
            existing_skills_cache=all_skills,
        )

        # Upward inference on the skill graph
        detected_skills = await self._expand_with_upward_inference(
            detected_skills,
            all_skills=all_skills,
        )

        # Load active clusters
        clusters = await self._clusters.get_all_active()
        if not clusters:
            logger.warning(
                "No tech clusters available — skipping Phase 2 diagnosis",
                user_id=str(user_id),
            )
            return UserProfileDTO(
                user_id=user_id,
                cv_id=cv_id,
                seniority=seniority.value,
                primary_specialty="Sin Diagnóstico",
                alignment_score=0.0,
                full_name=profile.full_name,
                message="Profile saved. Diagnosis skipped — no clusters configured.",
            )

        active_clusters = [c for c in clusters if c.centroid_skills]
        if not active_clusters:
            logger.warning(
                "No active clusters with centroid skills — skipping Phase 2",
                user_id=str(user_id),
            )
            return UserProfileDTO(
                user_id=user_id,
                cv_id=cv_id,
                seniority=seniority.value,
                primary_specialty="Sin Diagnóstico",
                alignment_score=0.0,
                full_name=profile.full_name,
                message="Profile saved. Diagnosis skipped — clusters have no centroid skills.",
            )

        # Compute Weighted Jaccard Similarity per cluster
        _primary_raw, _secondaries_raw, affinities_raw, domain_affinities_dto = (
            compute_affinities_and_domains(detected_skills, active_clusters)
        )
        if not affinities_raw:
            logger.warning(
                "No cluster affinities — skipping Phase 2",
                user_id=str(user_id),
            )
            return UserProfileDTO(
                user_id=user_id,
                cv_id=cv_id,
                seniority=seniority.value,
                primary_specialty="Sin Diagnóstico",
                alignment_score=0.0,
                full_name=profile.full_name,
                message="Profile saved. Could not compute cluster affinity.",
            )

        # Select Top 3 affinities with affinity_score > 0
        valid_affinities = [a for a in affinities_raw if a.affinity_score > 0]
        if not valid_affinities:
            valid_affinities = affinities_raw[:1]
        top_affinities = valid_affinities[:3]

        from dataclasses import replace as dc_replace_affinity

        primary = dc_replace_affinity(top_affinities[0], is_primary=True)
        secondaries = [dc_replace_affinity(a, is_primary=False) for a in top_affinities[1:]]

        # Detect skill gaps vs primary cluster
        primary_cluster = next((c for c in clusters if c.id == primary.cluster_id), None)
        skill_gaps = []

        if primary_cluster:
            user_tech_skills = {
                s.normalized_name for s in detected_skills if s.nature == SkillNature.TECH
            }
            primary_cluster_tech_skills = [
                s
                for s in primary_cluster.centroid_skills
                if s.nature == SkillNature.TECH and not is_concept_skill(s)
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
                    skill_gaps.append(SkillGap(skill=skill, market_importance=importance))
            skill_gaps.sort(key=lambda g: g.skill.weight * g.skill.frequency, reverse=True)

        # Persist enriched profile with is_diagnosed=True and Top 3 affinities
        logger.info(
            "Phase 2 — persisting full diagnosis with Top 3 affinities",
            user_id=str(user_id),
            primary=primary.cluster_name,
            secondaries=[s.cluster_name for s in secondaries],
        )
        from dataclasses import replace as dc_replace_profile

        diagnosed_profile = dc_replace_profile(
            profile,
            embedding=cv_embedding,
            detected_skills=detected_skills,
            seniority=seniority,
            primary_affinity=primary,
            secondary_affinities=secondaries,
            skill_gaps=skill_gaps,
            is_diagnosed=True,
        )
        await self._profiles.save(diagnosed_profile)

        logger.info(
            "Phase 2 complete — full diagnosis persisted",
            user_id=str(user_id),
            specialty=primary.cluster_name,
            score=primary.affinity_score,
        )

        # Build response DTOs for all Top 3 affinities
        user_skills_map = {s.normalized_name: s for s in detected_skills}
        primary_dto = _cluster_affinity_to_dto(primary, True, user_skills_map)
        secondaries_dto = [_cluster_affinity_to_dto(a, False, user_skills_map) for a in secondaries]
        all_affinities_dto = [primary_dto, *secondaries_dto]

        return UserProfileDTO(
            user_id=user_id,
            cv_id=cv_id,
            seniority=seniority.value,
            primary_specialty=primary.cluster_name,
            alignment_score=primary.affinity_score,
            secondary_affinities=secondaries_dto,
            all_affinities=all_affinities_dto,
            domain_affinities=domain_affinities_dto,
            detected_skills=[
                SkillDTO(
                    name=s.name,
                    skill_type=s.nature.value,
                    market_importance="consolidated",
                    market_demand_percentage=_normalize_demand_percentage(s.frequency),
                    self_taught=s.self_taught,
                    personal_projects=s.personal_projects,
                    years_of_experience=s.years_of_experience,
                    has_certification=s.has_certification,
                    ict_score=s.ict_score,
                    trend=determine_trend(s.name),
                    is_custom=getattr(s, "is_custom", False),
                )
                for s in detected_skills
            ],
            skill_gaps=[
                SkillDTO(
                    name=g.skill.name,
                    skill_type=g.skill.nature.value,
                    market_importance=g.market_importance,
                    market_demand_percentage=_normalize_demand_percentage(g.skill.frequency),
                    trend=determine_trend(g.skill.name),
                )
                for g in skill_gaps
                if not is_concept_skill(g.skill)
            ],
            full_name=diagnosed_profile.full_name,
            current_job_role=diagnosed_profile.current_job_role,
            years_experience=diagnosed_profile.years_experience,
            preferred_modality=diagnosed_profile.preferred_modality,
            location=diagnosed_profile.location,
            availability=diagnosed_profile.availability,
            work_experience=diagnosed_profile.work_experience,
            education=diagnosed_profile.education,
            certifications=diagnosed_profile.certifications,
            is_diagnosed=True,
            message="Profile generated successfully",
        )

    async def execute(
        self,
        user_id: UUID,
        cv_id: UUID,
        cv_content: bytes,
        content_type: str,
    ) -> UserProfileDTO:
        """Run the full CV analysis pipeline (extract → diagnose).

        Creates a temporary profile from extracted data then runs diagnosis.
        This method exists for backwards compatibility; the new flow uses
        extract_skills + separate finalize_diagnosis steps.
        """
        try:
            result = await self.extract_skills(
                user_id,
                cv_id,
                cv_content,
                content_type,
            )

            # Create profile from extracted data so finalize_diagnosis can run
            profile = await self._profiles.get_by_user_id(user_id)
            if not profile:
                extracted_data = result["extracted_data"]
                years_exp = extracted_data.get("years_experience")
                if isinstance(years_exp, (int, float)):
                    if years_exp >= 6:
                        seniority = SeniorityLevel.SENIOR
                    elif years_exp >= 3:
                        seniority = SeniorityLevel.MID
                    else:
                        seniority = SeniorityLevel.JUNIOR
                else:
                    seniority = _estimate_seniority(result["cv_text"])

                raw_skills = extracted_data.get("skills", [])
                skill_objects = []
                for item in raw_skills:
                    if isinstance(item, dict) and "name" in item:
                        skill_objects.append(
                            Skill(
                                name=item["name"],
                                nature=_nature_from_category(item.get("category", "technical")),
                                normalized_name=item["name"]
                                .lower()
                                .replace(" ", "")
                                .replace(".", ""),
                                self_taught=bool(item.get("self_taught", False)),
                                personal_projects=bool(item.get("personal_projects", False)),
                                years_of_experience=int(item.get("years_of_experience", 0) or 0),
                                has_certification=bool(item.get("has_certification", False)),
                            )
                        )

                phase1_profile = UserProfile(
                    user_id=user_id,
                    cv_id=cv_id,
                    embedding=[],
                    detected_skills=skill_objects,
                    seniority=seniority,
                    primary_affinity=ClusterAffinity(
                        cluster_id=uuid4(),
                        cluster_name="Sin Diagnóstico",
                        affinity_score=0.0,
                        is_primary=True,
                    ),
                    secondary_affinities=[],
                    skill_gaps=[],
                    current_job_role=extracted_data.get("current_job_role") or None,
                    professional_summary=extracted_data.get("professional_summary") or None,
                    years_experience=int(years_exp)
                    if isinstance(years_exp, (int, float))
                    else None,
                    work_experience=extracted_data.get("work_experience") or [],
                    education=extracted_data.get("education") or [],
                    certifications=extracted_data.get("certifications") or [],
                    cv_raw_text=result["cv_text"],
                    is_diagnosed=False,
                )
                await self._profiles.save_profile(phase1_profile, persist_diagnostics=False)

            return await self.finalize_diagnosis(
                user_id,
                cv_id,
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


def _normalize_demand_percentage(frequency: float | None) -> int:
    """Normalize skill frequency or importance metric to a sensible market demand percentage (20% - 98%)."""
    if frequency is None:
        return 70
    if 0.0 < frequency <= 1.0:
        return round(frequency * 100)
    # If frequency is an importance score (e.g. 1.5 - 3.0)
    scaled = (frequency / 3.0) * 100
    return min(98, max(20, round(scaled)))


def _cluster_affinity_to_dto(
    affinity: ClusterAffinity,
    is_primary: bool,
    user_skills_map: dict[str, Any],
) -> ClusterAffinityDTO:
    """Helper to convert a ClusterAffinity domain entity into a ClusterAffinityDTO."""
    return ClusterAffinityDTO(
        cluster_id=affinity.cluster_id,
        cluster_name=affinity.cluster_name,
        affinity_score=affinity.affinity_score,
        is_primary=is_primary,
        market_insights=affinity.market_insights,
        compatible_roles=affinity.compatible_roles,
        ai_insight=affinity.ai_insight,
        job_offer_count=affinity.job_offer_count,
        top_skills=affinity.top_skills,
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
                market_demand_percentage=_normalize_demand_percentage(s.frequency),
                self_taught=user_skills_map[s.normalized_name].self_taught
                if s.normalized_name in user_skills_map
                else False,
                personal_projects=user_skills_map[s.normalized_name].personal_projects
                if s.normalized_name in user_skills_map
                else False,
                years_of_experience=user_skills_map[s.normalized_name].years_of_experience
                if s.normalized_name in user_skills_map
                else 0,
                has_certification=user_skills_map[s.normalized_name].has_certification
                if s.normalized_name in user_skills_map
                else False,
                ict_score=user_skills_map[s.normalized_name].ict_score
                if s.normalized_name in user_skills_map
                else 0.0,
                trend=determine_trend(s.name),
            )
            for s in affinity.detected_skills
        ],
        skill_gaps=[
            SkillDTO(
                name=g.skill.name,
                skill_type=g.skill.nature.value,
                market_importance=g.market_importance,
                market_demand_percentage=_normalize_demand_percentage(g.skill.frequency),
                trend=determine_trend(g.skill.name),
            )
            for g in affinity.skill_gaps
            if not is_concept_skill(g.skill)
        ],
    )


# === Helpers ===


def _build_combined_cv_extraction_prompt(cv_text: str) -> str:
    """Build single combined LLM prompt for CV extraction.

    Merges Phase 1 (extraction) and Phase 1.5 (skill evidence enrichment)
    into one prompt, reducing latency and LLM calls.
    """
    return f"""You are a professional CV analyzer.

FIRST, determine if the text below is a professional CV/resume (currículum vitae).
A CV typically contains personal information, work experience, education history, and skills.

If the text is NOT a CV (e.g., it is an invoice, letter, contract, terms of service, or any other document), respond with EXACTLY:
{{"error": "not_a_cv", "document_type": "<brief description of what the document appears to be>"}}

If the text IS a CV, extract ALL of the following details in a structured JSON format:

1. Current Job Role (current_job_role): the person's most recent job title
2. Professional Summary (professional_summary): a 1-2 sentence summary of the candidate's profile
3. Years of experience (years_experience): total years of professional experience (integer or null)
4. Skills (skills): an exhaustive array of skill objects. Extract EVERY SINGLE programming language, database, framework, library, tool, cloud provider, methodology, and engineering concept mentioned in the CV. Do NOT extract soft skills (e.g., leadership, communication, teamwork, time management). Focus strictly on technical skills and tools. The list should be exhaustive (typically 20-50 items for a technical profile).

CRITICAL RULE FOR PARENTHESES AND SUB-TOOLS:
When technologies or tools are listed inside parentheses, slashes, or bullet sub-lists, you MUST extract EACH item individually as its own separate skill object in addition to the parent concept. Never group them into a single string or omit them.
Examples:
- "CI/CD (Bitbucket, Jenkins, GitHub Actions)" -> MUST extract 4 individual skills: "CI/CD", "Bitbucket", "Jenkins", "GitHub Actions".
- "Cloud: Azure (APIM, Functions, Key Vault, Service Bus, Blob Storage, DevOps)" -> MUST extract: "Azure", "Azure Functions", "Azure Key Vault", "Azure Service Bus", "Azure Blob Storage", "Azure DevOps", "APIM".
- "AWS (Lambda, EC2, Api Gateway, RDS, S3)" -> MUST extract: "AWS", "AWS Lambda", "Amazon EC2", "API Gateway", "Amazon RDS", "Amazon S3".
- "Languages: Java (Spring Boot, Spring WebFlux, Spring Cloud, Hibernate)" -> MUST extract: "Java", "Spring Boot", "Spring WebFlux", "Spring Cloud", "Hibernate".
- "TypeScript/JavaScript (Angular, Vue.js, Nuxt.js)" -> MUST extract: "TypeScript", "JavaScript", "Angular", "Vue.js", "Nuxt.js".
- "Databases: SQL • NoSQL • RabbitMQ • Apache Kafka" -> MUST extract: "SQL", "NoSQL", "RabbitMQ", "Apache Kafka".

CRITICAL NORMALIZATION RULES FOR SKILL NAMES:
- NEVER INCLUDE SPECIFIC SOFTWARE VERSION NUMBERS: Standardize to the core technology name.
  Examples: "Python 3.12" -> "Python", "Java 17" -> "Java", "Angular 14" -> "Angular", "PostgreSQL 15" -> "PostgreSQL", "React 18" -> "React", "Node 20" -> "Node.js", ".NET 8" -> ".NET".
- NEVER INCLUDE ACTION VERBS OR CONVERSATIONAL PREFIXES: Extract only the pure technology or tool noun.
  Examples: "Manejo de Git" -> "Git", "Desarrollo con React" -> "React", "Conocimiento en Docker" -> "Docker", "Administración de Linux" -> "Linux".
- NEVER INCLUDE PROFICIENCY LEVEL ADJECTIVES:
  Examples: "React avanzado" -> "React", "Senior Java" -> "Java", "Basic SQL" -> "SQL".
- SPLIT COMPOUND SLASHES: Slashes like "JavaScript/TypeScript" represent two distinct skills. Always extract each individually ("JavaScript", "TypeScript"). Only preserve true single acronyms containing slashes ("CI/CD", "TCP/IP", "I/O", "PL/SQL", "Client/Server").


For each skill, extract:
- name: the canonical technology/tool name (strictly applying the normalization rules above: no versions, no prefixes, no qualifiers)
- category: one of "technical", "tools", or "methodologies"
- years_of_experience: integer, estimated years the candidate has used this skill based on work experience dates
- self_taught: boolean, true only if the CV explicitly states this skill was self-taught
- personal_projects: boolean, true if the CV mentions using this skill in personal or open-source projects
- has_certification: boolean, true if the CV mentions an official certification for this skill

Be precise. Only set self_taught, personal_projects, or has_certification to true if there is explicit evidence in the CV text.

5. Work Experience (work_experience): array of objects with:
   - company: string
   - role: string
   - start_date: string or null
   - end_date: string or null ("Present" if current)
   - description: string or null

6. Education (education): array of objects with:
   - institution: string
   - degree: string
   - field: string or null
   - start_date: string or null
   - end_date: string or null

7. Certifications (certifications): array of objects with:
   - name: string
   - issuer: string or null
   - date: string or null

CV Text:
{cv_text}

Respond ONLY with a valid JSON object. If the text is not a CV, use the error format above.
If it IS a CV, use this schema:
{{
  "current_job_role": "string or null",
  "professional_summary": "string or null",
  "years_experience": integer or null,
  "skills": [
    {{
      "name": "string",
      "category": "technical | tools | methodologies",
      "years_of_experience": integer,
      "self_taught": boolean,
      "personal_projects": boolean,
      "has_certification": boolean
    }}
  ],
  "work_experience": [
    {{
      "company": "string",
      "role": "string",
      "start_date": "string or null",
      "end_date": "string or null",
      "description": "string or null"
    }}
  ],
  "education": [
    {{
      "institution": "string",
      "degree": "string",
      "field": "string or null",
      "start_date": "string or null",
      "end_date": "string or null"
    }}
  ],
  "certifications": [
    {{
      "name": "string",
      "issuer": "string or null",
      "date": "string or null"
    }}
  ]
}}"""


def _clean_and_unpack_skills(parsed: dict[str, Any]) -> dict[str, Any]:
    """Ensure skills grouped in parentheses or slashes are unpacked into individual items."""
    import re

    raw_skills = parsed.get("skills", [])
    if not isinstance(raw_skills, list):
        return parsed

    unpacked_skills: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for item in raw_skills:
        if not isinstance(item, dict) or "name" not in item:
            continue

        orig_name = str(item["name"]).strip()
        if not orig_name:
            continue

        # Check if the name has parentheses with items, e.g. "CI/CD (Bitbucket, Jenkins, GitHub Actions)"
        paren_match = re.search(r"^(.*?)\s*\((.*?)\)$", orig_name)
        if paren_match:
            main_part = paren_match.group(1).strip()
            sub_items = [s.strip() for s in paren_match.group(2).split(",") if s.strip()]

            candidates = [main_part] if main_part else []
            for sub in sub_items:
                if main_part.lower() in ("azure", "aws", "spring") and not sub.lower().startswith(
                    main_part.lower()
                ):
                    candidates.append(f"{main_part} {sub}")
                candidates.append(sub)

            for cand in candidates:
                cand_lower = cand.lower()
                if cand_lower not in seen_names and len(cand) >= 2:
                    seen_names.add(cand_lower)
                    new_item = dict(item)
                    new_item["name"] = cand
                    unpacked_skills.append(new_item)
        else:
            if "/" in orig_name and not orig_name.lower().startswith("ci/cd"):
                slash_parts = [p.strip() for p in orig_name.split("/") if p.strip()]
                for p in slash_parts:
                    p_lower = p.lower()
                    if p_lower not in seen_names and len(p) >= 2:
                        seen_names.add(p_lower)
                        new_item = dict(item)
                        new_item["name"] = p
                        unpacked_skills.append(new_item)
            else:
                name_lower = orig_name.lower()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    unpacked_skills.append(item)

    parsed["skills"] = unpacked_skills
    return parsed


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
        return _clean_and_unpack_skills(parsed)
    except Exception as exc:
        logger.warning(
            "Failed to parse LLM CV extraction, fallback to empty defaults", error=str(exc)
        )
        return {}


def _nature_from_category(category: str) -> SkillNature:
    """Map Phase 1.5 category string to SkillNature enum."""
    cat_lower = category.lower().strip()
    if cat_lower in ("technical", "tech"):
        return SkillNature.TECH
    if cat_lower in ("concept", "methodology", "methodologies"):
        return SkillNature.CONCEPT
    if cat_lower in ("tools", "tool"):
        return SkillNature.TECH
    return SkillNature.TECH


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
        llm_service: LLMService | None = None,
    ) -> None:
        self._job_offers = job_offer_repo
        self._skills = skill_repo
        self._embeddings = embedding_service
        self._llm = llm_service

    async def execute(self) -> dict[str, Any]:
        """Run the normalization pipeline for unnormalized job offers."""
        import difflib

        import numpy as np

        from src.ml_engine.application.skill_catalog_service import (
            SINGLE_SLASH_TERMS,
            build_skill_lookup_indexes,
            match_single_skill,
        )
        from src.ml_engine.domain.entities import Skill

        logger.info("Starting Skill Normalization Pipeline")

        # 1. Fetch unnormalized offers
        offers = await self._job_offers.get_unnormalized_offers(limit=100)
        if not offers:
            logger.info("No unnormalized offers found.")
            return {"processed_offers": 0, "new_skills": 0}

        # 2. Fetch existing skills from database and build indexes
        existing_skills = await self._skills.get_all_skills()
        alias_to_skill, norm_to_skill = build_skill_lookup_indexes(existing_skills)
        skill_map: dict[str, Skill] = {s.normalized_name: s for s in existing_skills}

        processed_offer_ids = []

        # Gather unique skills in this batch that are not matched
        unmapped_raw_skills: dict[str, str] = {}  # norm_name -> raw_name

        for offer in offers:
            processed_offer_ids.append(offer["id"])
            raw_skills = offer.get("raw_hard_skills", [])
            expanded_skills: list[str] = []
            for rs in raw_skills:
                if not isinstance(rs, str):
                    continue
                c_rs = rs.strip()
                if "/" in c_rs and c_rs.lower() not in SINGLE_SLASH_TERMS:
                    expanded_skills.extend(p.strip() for p in c_rs.split("/") if p.strip())
                else:
                    expanded_skills.append(c_rs)

            for clean_name in expanded_skills:
                norm_name = clean_name.lower().replace(" ", "").replace(".", "")

                if not norm_name:
                    continue

                # 1. Exact or already matched check
                if norm_name in skill_map:
                    continue

                # 2. Fast catalog & alias resolution (Lightcast canonicals, aliases, parentheses)
                matched = match_single_skill(clean_name, alias_to_skill, norm_to_skill)
                if matched:
                    skill_map[norm_name] = matched
                    continue

                # 3. Unmapped candidate: queue for semantic / fuzzy resolution
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
            skill_vector = unmapped_embeddings.get(norm_name)

            if norm_name in skill_map:
                continue

            # Reject obvious hallucinations, empty states, or sentences mistakenly parsed as skills
            hallucination_phrases = (
                "no mencionado",
                "not mentioned",
                "no especificado",
                "sin especificar",
                "none specified",
                "n/a",
            )
            if (
                any(phrase in clean_name.lower() for phrase in hallucination_phrases)
                or len(clean_name) > 80
            ):
                logger.info(f"Skipping hallucinated or invalid skill candidate: '{clean_name}'")
                continue

            # Semantic Match vs existing canonical skills
            best_match = None
            best_score = -1.0

            if skill_vector:
                for s in existing_skills:
                    if s.embedding is not None:
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
                skill_map[norm_name] = best_match
            else:
                # Fallback: Fuzzy matching (difflib) against canonical skill_map
                all_keys = list(skill_map.keys())
                matches = difflib.get_close_matches(norm_name, all_keys, n=1, cutoff=0.85)
                if matches:
                    matched_skill = skill_map[matches[0]]
                    logger.info(
                        f"Mapped '{clean_name}' via fuzzy matching to canonical '{matched_skill.name}'"
                    )
                    skill_map[norm_name] = matched_skill
                else:
                    logger.debug(
                        f"Quarantined unmapped offer skill candidate: '{clean_name}' (did not match canonical catalog)"
                    )

        # Re-iterate offers to build final link list (deduplicating per job offer)
        final_offer_skills_to_insert = []
        seen_offer_skill_pairs: set[tuple[Any, str]] = set()

        for offer in offers:
            raw_skills = offer.get("raw_hard_skills", [])
            expanded_skills = []
            for rs in raw_skills:
                if not isinstance(rs, str):
                    continue
                c_rs = rs.strip()
                if "/" in c_rs and c_rs.lower() not in SINGLE_SLASH_TERMS:
                    expanded_skills.extend(p.strip() for p in c_rs.split("/") if p.strip())
                else:
                    expanded_skills.append(c_rs)

            for clean_name in expanded_skills:
                norm_name = clean_name.lower().replace(" ", "").replace(".", "")
                if norm_name in skill_map:
                    target_skill = skill_map[norm_name]
                    pair_key = (offer["id"], target_skill.normalized_name)
                    if pair_key not in seen_offer_skill_pairs:
                        seen_offer_skill_pairs.add(pair_key)
                        final_offer_skills_to_insert.append(
                            {
                                "job_offer_id": offer["id"],
                                "skill_id": target_skill.id,
                                "skill_type": "hard_skill",
                            }
                        )

        # 3. Insert offer_skills relations directly with canonical IDs
        if final_offer_skills_to_insert:
            logger.info(f"Saving {len(final_offer_skills_to_insert)} offer_skills relations.")
            await self._job_offers.save_offer_skills(final_offer_skills_to_insert)

        # 5. Mark offers as normalized
        if processed_offer_ids:
            logger.info(f"Marking {len(processed_offer_ids)} offers as normalized.")
            await self._job_offers.mark_as_normalized(processed_offer_ids)

        return {
            "processed_offers": len(processed_offer_ids),
            "new_skills": 0,
            "offer_skills_linked": len(final_offer_skills_to_insert),
        }


def normalize_domain_key(domain_str: str) -> list[str]:
    """Normalize domain strings to canonical title-case names."""
    clean = domain_str.strip().lower()
    if clean in ("backend",):
        return ["Backend"]
    if clean in ("frontend",):
        return ["Frontend"]
    if clean in ("devops",):
        return ["DevOps"]
    if clean in ("cloud",):
        return ["Cloud"]
    if clean in ("cloud_devops", "cloud/devops"):
        return ["Cloud", "DevOps"]
    if clean in ("data", "data engineering", "data_engineering", "data science"):
        return ["Data"]
    if clean in ("qa", "testing", "quality assurance"):
        return ["QA"]
    if clean in ("mobile", "ios", "android"):
        return ["Mobile"]
    if clean in ("software_engineering", "security"):
        return []
    return [domain_str.strip().capitalize()]


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
                        u for u in user_tech_skills if set(u.domain_tags) & cluster_domains
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
                # Exclude CONCEPT skills from gap analysis (only recommend concrete technical tools)
                if is_concept_skill(skill):
                    continue

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
                for norm_d in normalize_domain_key(d):
                    if norm_d not in domain_scores:
                        domain_scores[norm_d] = 0.0
                    domain_scores[norm_d] += s.weight * (
                        s.frequency if s.frequency is not None else 1.0
                    )

    # Calcular promedios de demanda de mercado por dominio normalizado
    domain_demands_accum: dict[str, list[float]] = {}
    for cluster in active_clusters:
        for skill in cluster.centroid_skills:
            if skill.core_domains:
                for d in skill.core_domains:
                    for norm_d in normalize_domain_key(d):
                        if norm_d not in domain_demands_accum:
                            domain_demands_accum[norm_d] = []
                        domain_demands_accum[norm_d].append(skill.frequency)
            elif skill.domain_tags:
                for d in skill.domain_tags:
                    for norm_d in normalize_domain_key(d):
                        if norm_d not in domain_demands_accum:
                            domain_demands_accum[norm_d] = []
                        domain_demands_accum[norm_d].append(skill.frequency)

    domain_market_demand = {
        d: sum(freqs) / len(freqs) if freqs else 0.5 for d, freqs in domain_demands_accum.items()
    }

    total_domain_score = sum(domain_scores.values()) if domain_scores else 1.0
    domain_affinities_dto = [
        DomainAffinityDTO(
            domain=d,
            affinity_score=score / total_domain_score,
            market_demand=domain_market_demand.get(d, 0.5),
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
            return await self._build_user_graph(user_id, cluster_name)

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
    ) -> Any:
        """Build a knowledge graph scoped to a user's detected skills and skill gaps.

        Fetches only the user's profile data (a tiny, bounded set) rather than
        the full skill catalog, making this O(1) in catalog size.
        If cluster_name is provided, scopes the skills to that specific cluster.
        """
        from src.ml_engine.application.dtos import GraphLinkDTO, GraphNodeDTO, GraphResponseDTO

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
                neutral = {
                    s.normalized_name: s
                    for s in profile.detected_skills
                    if s.normalized_name not in acquired
                }
            else:
                acquired = {s.normalized_name: s for s in profile.detected_skills}
                gaps = {g.skill.normalized_name: g.skill for g in profile.skill_gaps}
                neutral = {}
        else:
            acquired = {s.normalized_name: s for s in profile.detected_skills}
            gaps = {g.skill.normalized_name: g.skill for g in profile.skill_gaps}
            neutral = {}

        # Fetch all skills to render as the general market backdrop
        all_market_skills = await self._skills.get_all_skills()
        market = {
            s.normalized_name: s
            for s in all_market_skills
            if s.normalized_name not in acquired
            and s.normalized_name not in gaps
            and s.normalized_name not in neutral
        }

        all_skills_to_render = (
            list(acquired.values())
            + list(gaps.values())
            + list(neutral.values())
            + list(market.values())
        )

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
                for norm_d in normalize_domain_key(d):
                    if norm_d not in domain_scores:
                        domain_scores[norm_d] = 0.0
                    domain_scores[norm_d] += s.weight * (
                        s.frequency if s.frequency is not None else 1.0
                    )

    domain_demands_accum: dict[str, list[float]] = {}
    for cluster in active_clusters:
        for skill in cluster.centroid_skills:
            if skill.core_domains:
                for d in skill.core_domains:
                    for norm_d in normalize_domain_key(d):
                        if norm_d not in domain_demands_accum:
                            domain_demands_accum[norm_d] = []
                        domain_demands_accum[norm_d].append(skill.frequency)
            elif skill.domain_tags:
                for d in skill.domain_tags:
                    for norm_d in normalize_domain_key(d):
                        if norm_d not in domain_demands_accum:
                            domain_demands_accum[norm_d] = []
                        domain_demands_accum[norm_d].append(skill.frequency)

    domain_market_demand = {
        d: sum(freqs) / len(freqs) if freqs else 0.5 for d, freqs in domain_demands_accum.items()
    }

    total_domain_score = sum(domain_scores.values()) if domain_scores else 1.0
    domain_affinities_dto = [
        DomainAffinityDTO(
            domain=d,
            affinity_score=score / total_domain_score,
            market_demand=domain_market_demand.get(d, 0.5),
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
                cv_raw_text=None,
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

        # Derive secondary affinities on-the-fly for diagnosed profiles where secondaries were not stored
        if (
            (not secondaries or len(secondaries) == 0)
            and profile.is_diagnosed
            and profile.detected_skills
            and len(active_clusters) > 1
        ):
            _p_raw, _s_raw, all_raw, _ = compute_affinities_and_domains(
                profile.detected_skills, active_clusters
            )
            if all_raw:
                valid_aff = [a for a in all_raw if a.affinity_score > 0] or all_raw[:1]
                top_aff = valid_aff[:3]
                from dataclasses import replace as dc_replace_aff

                if not primary or primary.cluster_name == "Sin Diagnóstico":
                    primary = dc_replace_aff(top_aff[0], is_primary=True)
                secondaries = [dc_replace_aff(a, is_primary=False) for a in top_aff[1:]]

        all_affinities = (
            [primary, *secondaries] if primary.cluster_name != "Sin Diagnóstico" else []
        )

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
                            self_taught=user_skills_map[s.normalized_name].self_taught
                            if s.normalized_name in user_skills_map
                            else False,
                            personal_projects=user_skills_map[s.normalized_name].personal_projects
                            if s.normalized_name in user_skills_map
                            else False,
                            years_of_experience=user_skills_map[
                                s.normalized_name
                            ].years_of_experience
                            if s.normalized_name in user_skills_map
                            else 0,
                            has_certification=user_skills_map[s.normalized_name].has_certification
                            if s.normalized_name in user_skills_map
                            else False,
                            ict_score=user_skills_map[s.normalized_name].ict_score
                            if s.normalized_name in user_skills_map
                            else 0.0,
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
                            self_taught=user_skills_map[s.normalized_name].self_taught
                            if s.normalized_name in user_skills_map
                            else False,
                            personal_projects=user_skills_map[s.normalized_name].personal_projects
                            if s.normalized_name in user_skills_map
                            else False,
                            years_of_experience=user_skills_map[
                                s.normalized_name
                            ].years_of_experience
                            if s.normalized_name in user_skills_map
                            else 0,
                            has_certification=user_skills_map[s.normalized_name].has_certification
                            if s.normalized_name in user_skills_map
                            else False,
                            ict_score=user_skills_map[s.normalized_name].ict_score
                            if s.normalized_name in user_skills_map
                            else 0.0,
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
                    is_custom=getattr(s, "is_custom", False),
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
            professional_summary=profile.professional_summary,
            years_experience=profile.years_experience,
            preferred_modality=profile.preferred_modality,
            location=profile.location,
            availability=profile.availability,
            work_experience=profile.work_experience,
            education=profile.education,
            certifications=profile.certifications,
            is_diagnosed=profile.is_diagnosed,
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
            raise HTTPException(
                status_code=404, detail="No profile found. Please upload a CV first."
            )

        active_clusters = await self._clusters.get_all_active()
        requested_cluster = next((c for c in active_clusters if c.name == cluster_name), None)
        if not requested_cluster:
            raise HTTPException(status_code=404, detail=f"Cluster '{cluster_name}' not found.")

        # Compute affinity for this cluster
        _primary, _secondaries, affinities, _ = compute_affinities_and_domains(
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
        updated_secondaries = [*existing_secondaries, new_affinity]

        # Save profile
        updated_profile = replace(profile, secondary_affinities=updated_secondaries)
        await self._profiles.save(updated_profile)

        # Return updated profile
        return await GetMyProfileUseCase(self._profiles, self._clusters).execute(user_id)


class GetClusterDiagnosticUseCase:
    """Gets or computes the diagnostic of a user profile's detected skills against a specific cluster."""

    def __init__(
        self,
        profile_repository: UserProfileRepository,
        cluster_repository: ClusterRepository,
    ) -> None:
        self._profiles = profile_repository
        self._clusters = cluster_repository

    async def execute(self, user_id: UUID, cluster_name: str) -> DiagnosticDetailDTO | None:

        from fastapi import HTTPException

        profile = await self._profiles.get_by_user_id(user_id)
        if not profile:
            raise HTTPException(
                status_code=404, detail="No profile found. Please upload a CV first."
            )

        active_clusters = await self._clusters.get_all_active()
        requested_cluster = next(
            (c for c in active_clusters if c.name.lower() == cluster_name.lower()), None
        )
        if not requested_cluster:
            raise HTTPException(status_code=404, detail=f"Cluster '{cluster_name}' not found.")

        # Compute dynamic affinity and gaps against the requested cluster
        _, _, affinities, _ = compute_affinities_and_domains(
            profile.detected_skills, [requested_cluster]
        )
        if not affinities:
            raise HTTPException(status_code=500, detail="Failed to compute affinity score.")

        affinity = affinities[0]
        active_clusters = [c for c in active_clusters if c.centroid_skills]
        domain_affinities_dto = compute_domain_affinities(profile.detected_skills, active_clusters)

        user_skills_map = {s.normalized_name: s for s in profile.detected_skills}

        # Compute dynamic financial and opportunity projections based on parametric market percentiles
        raw_insights = affinity.market_insights or {}
        avg_usd = float(raw_insights.get("average_salary_usd") or 1800.0)
        p25_usd = float(raw_insights.get("salary_p25_usd") or (avg_usd * 0.75))
        p50_usd = float(raw_insights.get("salary_median_usd") or avg_usd)
        p75_usd = float(raw_insights.get("salary_p75_usd") or (avg_usd * 1.35))
        total_demand = int(raw_insights.get("total_demand") or affinity.job_offer_count or 100)

        seniority_val = (
            profile.seniority.value.lower()
            if hasattr(profile.seniority, "value")
            else str(profile.seniority).lower()
        )

        aff_score = float(affinity.affinity_score)

        # Baseline salary is anchored to the candidate's seniority band and their affinity match
        if "senior" in seniority_val or "staff" in seniority_val:
            base_min = p50_usd
            base_target = p75_usd
            ceiling_target = round(p75_usd * 1.2, 2)
        elif "junior" in seniority_val:
            base_min = round(p25_usd * 0.85, 2)
            base_target = p25_usd
            ceiling_target = p50_usd
        else:  # MID
            base_min = p25_usd
            base_target = p50_usd
            ceiling_target = p75_usd

        # Current estimated salary based on current affinity within their seniority range
        current_estimated_salary_usd = round(
            base_min + (base_target - base_min) * max(0.15, min(1.0, aff_score)),
            2,
        )

        # Target salary when all critical gaps are closed (reaching the next seniority percentile ceiling)
        projected_salary_usd = round(
            max(current_estimated_salary_usd * 1.08, ceiling_target),
            2,
        )

        total_potential_gain = max(80.0, projected_salary_usd - current_estimated_salary_usd)

        # Distribute the potential gain across the technical gaps proportionally (exclude CONCEPT skills)
        gap_weights = []
        for g in affinity.skill_gaps:
            if is_concept_skill(g.skill):
                continue
            imp_w = (
                3.0
                if g.market_importance == "critical"
                else (2.0 if g.market_importance == "high" else 1.0)
            )
            norm_freq = float(g.skill.frequency or 1.0)
            gap_weights.append((g, imp_w * norm_freq))

        total_gap_weight = sum(w for _, w in gap_weights) or 1.0

        gap_impacts_list = []
        for g, w in gap_weights:
            share = w / total_gap_weight
            skill_boost_usd = round(total_potential_gain * share, 2)
            demand_pct = _normalize_demand_percentage(g.skill.frequency)
            opp_boost = max(1, round((total_demand - round(total_demand * aff_score)) * share))
            gap_impacts_list.append(
                {
                    "skill_name": g.skill.name,
                    "skill_type": g.skill.nature.value,
                    "market_importance": g.market_importance,
                    "salary_boost_usd": skill_boost_usd,
                    "salary_boost_pen": round(skill_boost_usd * 3.75, 2),
                    "opportunity_boost_count": opp_boost,
                    "market_demand_percentage": demand_pct,
                }
            )

        potential_gain_percentage = round(
            ((projected_salary_usd / max(1.0, current_estimated_salary_usd)) - 1.0) * 100, 1
        )

        salary_projection_dto = {
            "current_estimated_salary_usd": current_estimated_salary_usd,
            "current_estimated_salary_pen": round(current_estimated_salary_usd * 3.75, 2),
            "projected_salary_usd": projected_salary_usd,
            "projected_salary_pen": round(projected_salary_usd * 3.75, 2),
            "potential_gain_percentage": max(0.0, potential_gain_percentage),
            "cluster_average_usd": avg_usd,
            "cluster_p75_usd": p75_usd,
            "salary_p25_usd": p25_usd,
            "salary_p25_pen": round(p25_usd * 3.75, 2),
            "salary_median_usd": p50_usd,
            "salary_median_pen": round(p50_usd * 3.75, 2),
            "salary_p75_pen": round(p75_usd * 3.75, 2),
        }

        direct_matches = max(1, round(total_demand * max(0.05, min(1.0, aff_score))))
        opportunity_projection_dto = {
            "direct_matches_count": direct_matches,
            "potential_matches_count": total_demand,
            "total_cluster_offers": total_demand,
            "unlock_percentage": round((direct_matches / max(1, total_demand)) * 100, 1),
        }

        return DiagnosticDetailDTO(
            user_id=profile.user_id,
            full_name=profile.full_name,
            current_job_role=profile.current_job_role,
            seniority=profile.seniority.value,
            last_analysis_date=profile.last_analysis_date,
            cluster_name=affinity.cluster_name,
            affinity_score=affinity.affinity_score,
            job_offer_count=affinity.job_offer_count,
            top_skills=affinity.top_skills,
            market_insights=affinity.market_insights,
            compatible_roles=affinity.compatible_roles,
            ai_insight=affinity.ai_insight,
            salary_projection=salary_projection_dto,
            opportunity_projection=opportunity_projection_dto,
            gap_impacts=gap_impacts_list,
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
                    market_demand_percentage=_normalize_demand_percentage(s.frequency),
                    self_taught=user_skills_map[s.normalized_name].self_taught
                    if s.normalized_name in user_skills_map
                    else False,
                    personal_projects=user_skills_map[s.normalized_name].personal_projects
                    if s.normalized_name in user_skills_map
                    else False,
                    years_of_experience=user_skills_map[s.normalized_name].years_of_experience
                    if s.normalized_name in user_skills_map
                    else 0,
                    has_certification=user_skills_map[s.normalized_name].has_certification
                    if s.normalized_name in user_skills_map
                    else False,
                    ict_score=user_skills_map[s.normalized_name].ict_score
                    if s.normalized_name in user_skills_map
                    else 0.0,
                    trend=determine_trend(s.name),
                )
                for s in affinity.detected_skills
            ],
            skill_gaps=[
                SkillDTO(
                    name=g.skill.name,
                    skill_type=g.skill.nature.value,
                    market_importance=g.market_importance,
                    market_demand_percentage=_normalize_demand_percentage(g.skill.frequency),
                    trend=determine_trend(g.skill.name),
                )
                for g in affinity.skill_gaps
                if not is_concept_skill(g.skill)
            ],
            domain_affinities=domain_affinities_dto,
            total_profile_skills=len(profile.detected_skills),
        )
