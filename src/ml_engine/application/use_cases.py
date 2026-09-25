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
from src.ml_engine.application.skill_catalog_service import (
    PROFILE_SKILL_BLACKLIST,
    SkillCatalogService,
)
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
from src.shared.telemetry import TelemetryTracker

logger = structlog.get_logger(__name__)

CANONICAL_DOMAINS: list[str] = ["Backend", "Frontend", "Data", "DevOps", "QA", "Mobile", "Cloud"]

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
        seniority: SeniorityLevel | None = None,
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

        # Filter out abstract meta-skills/categories from user profiles
        resolved_skills = [
            s
            for s in resolved_skills
            if s.name.lower().strip() not in PROFILE_SKILL_BLACKLIST
            and s.normalized_name.lower().strip() not in PROFILE_SKILL_BLACKLIST
        ]

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
                stamped_skill = dc_replace(
                    stamped_skill, ict_score=stamped_skill.calculate_ict(seniority)
                )
                decorated_skills.append(stamped_skill)
            else:
                decorated_skills.append(dc_replace(skill, ict_score=skill.calculate_ict(seniority)))

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
        """Single streamlined LLM call for core CV extraction.

        Extracts: current job role, technical summary, years of experience,
        and exhaustive technical skills list in one prompt.

        Returns the parsed JSON dict from the LLM.
        """
        logger.info("Running combined LLM extraction")
        cv_text_char_limit = 15000
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
        async with TelemetryTracker(
            "cv_parsing",
            user_id=user_id,
            initial_metadata={
                "cv_id": str(cv_id),
                "file_size_bytes": len(cv_content),
                "content_type": content_type,
            },
        ) as parse_tracker:
            cv_text = await self._cv_parser.extract_text(cv_content, content_type)
            parse_tracker.add_metadata(
                {
                    "char_count": len(cv_text),
                    "word_count": len(cv_text.split()),
                }
            )

        if not cv_text.strip():
            raise MLPipelineError("CV text extraction returned empty content")

        is_cv, confidence = await self._classify_as_cv(cv_text)
        if not is_cv:
            raise MLPipelineError(
                "The uploaded document does not appear to be a professional CV/resume. "
                "Please upload a document with your work experience, education, and skills."
            )
        logger.debug("Document classified as CV", confidence=confidence)

        async with TelemetryTracker(
            "llm_extraction_phase1",
            user_id=user_id,
            initial_metadata={
                "cv_id": str(cv_id),
                "classification_confidence": confidence,
            },
        ) as llm_tracker:
            extracted_data = await self._combined_llm_extraction(cv_text)

            # Pre-normalize extracted skills against Lightcast catalog (Phase 1)
            raw_skills_count = 0
            if "skills" in extracted_data and isinstance(extracted_data["skills"], list):
                raw_skills_count = len(extracted_data["skills"])
                try:
                    all_skills = await self._skills.get_all_skills()
                    # Hybrid pipeline: Fast deterministic scan of raw CV text against canonical catalog
                    direct_catalog_skills = await self._catalog.scan_text_for_catalog_skills(
                        cv_text, existing_skills_cache=all_skills
                    )
                    combined_skills = list(extracted_data["skills"]) + direct_catalog_skills

                    extracted_data["skills"] = await self._catalog.normalize_extracted_skills(
                        combined_skills, existing_skills_cache=all_skills
                    )
                    logger.info(
                        "Pre-normalized extracted skills against catalog (hybrid mode)",
                        count=len(extracted_data["skills"]),
                    )
                except Exception as exc:
                    logger.warning("Failed to pre-normalize skills against catalog", error=str(exc))
                    if hasattr(self._skills, "_session") and self._skills._session is not None:
                        try:
                            await self._skills._session.rollback()
                        except Exception as rollback_err:
                            logger.debug("Rollback attempt completed", error=str(rollback_err))

            skills_list = extracted_data.get("skills", [])
            std_count = sum(
                1 for s in skills_list if isinstance(s, dict) and not s.get("is_custom", False)
            )
            custom_count = sum(
                1 for s in skills_list if isinstance(s, dict) and s.get("is_custom", False)
            )
            total_extracted = len(skills_list) if isinstance(skills_list, list) else 0

            llm_tracker.add_metadata(
                {
                    "raw_skills_count": raw_skills_count,
                    "total_skills_phase1": total_extracted,
                    "standard_skills_phase1": std_count,
                    "custom_skills_phase1": custom_count,
                    "standardization_ratio_phase1": (
                        round(std_count / total_extracted, 4) if total_extracted > 0 else 0.0
                    ),
                    "role_extracted": extracted_data.get("current_job_role"),
                    "years_experience": extracted_data.get("years_experience"),
                }
            )

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

        async with TelemetryTracker(
            "diagnosis_phase2",
            user_id=user_id,
            initial_metadata={
                "cv_id": str(cv_id),
                "validated_skills_provided": validated_skills is not None,
                "validated_skills_count": len(validated_skills)
                if validated_skills is not None
                else 0,
            },
        ) as diag_tracker:
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
                seniority=seniority,
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
                diag_tracker.add_metadata({"status_detail": "no_clusters_available"})
                return UserProfileDTO(
                    user_id=user_id,
                    cv_id=cv_id,
                    seniority=seniority.value,
                    primary_specialty="Sin Diagnóstico",
                    alignment_score=0.0,
                    full_name=profile.full_name,
                    message="Profile saved. Diagnosis skipped — no clusters configured.",
                )

            active_clusters = [
                c for c in clusters if c.centroid_skills and getattr(c, "tier", "standard") != "low"
            ]
            if not active_clusters:
                logger.warning(
                    "No active clusters with centroid skills — skipping Phase 2",
                    user_id=str(user_id),
                )
                diag_tracker.add_metadata({"status_detail": "no_active_clusters_with_centroids"})
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
                diag_tracker.add_metadata({"status_detail": "no_cluster_affinities_computed"})
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

            # Record telemetry metrics for thesis evaluation
            total_skills = len(detected_skills)
            std_count = sum(1 for s in detected_skills if not getattr(s, "is_custom", False))
            custom_count = sum(1 for s in detected_skills if getattr(s, "is_custom", False))
            inferred_count = sum(
                1 for s in detected_skills if len(getattr(s, "inferred_from", [])) > 0
            )

            diag_tracker.add_metadata(
                {
                    "total_skills": total_skills,
                    "standard_skills": std_count,
                    "custom_skills": custom_count,
                    "inferred_skills": inferred_count,
                    "standardization_ratio": (
                        round(std_count / total_skills, 4) if total_skills > 0 else 0.0
                    ),
                    "inference_ratio": (
                        round(inferred_count / total_skills, 4) if total_skills > 0 else 0.0
                    ),
                    "primary_cluster": primary.cluster_name,
                    "affinity_score": round(primary.affinity_score, 4),
                    "gaps_count": len(skill_gaps),
                    "seniority": seniority.value,
                }
            )

            logger.info(
                "Phase 2 complete — full diagnosis persisted",
                user_id=str(user_id),
                specialty=primary.cluster_name,
                score=primary.affinity_score,
            )

            # Build response DTOs for all Top 3 affinities
            user_skills_map = {s.normalized_name: s for s in detected_skills}
            primary_dto = _cluster_affinity_to_dto(primary, True, user_skills_map)
            secondaries_dto = [
                _cluster_affinity_to_dto(a, False, user_skills_map) for a in secondaries
            ]
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

        raise MLPipelineError("Diagnosis phase 2 completed without producing a profile DTO.")

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
            if getattr(c, "tier", "standard") != "low"
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
    """Build contextual LLM prompt for CV extraction.

    Instructs the LLM to analyze every section of the CV (experience, projects,
    skills, education) for implicit and explicit technologies across all domains
    including design, CMS, templating, and modern web tools.
    """
    return f"""You are an expert IT recruiter and CV parser. Analyze the CV text below.

If the text is NOT a CV/resume, respond ONLY with:
{{"error": "not_a_cv", "document_type": "<brief description>"}}

If it IS a CV, extract the core technical profile strictly following this JSON schema:
{{
  "current_job_role": "string or null",
  "professional_summary": "1-2 sentence technical summary focusing on their primary domain and stack, or null",
  "years_experience": integer or null,
  "skills": ["string"]
}}

EXTRACTION RULES:
1. EXHAUSTIVE EXTRACTION — Go through EVERY section (Skills list, Experience, Projects, Education, Certifications, Tools) and extract ALL concrete technical skills mentioned:
   - Programming languages: JavaScript, TypeScript, Python, Java, C#, PHP, Go, Rust, Ruby, Dart, Kotlin, Swift, HTML, CSS, Sass, SCSS, etc.
   - Frontend frameworks & libraries: React, React.js, Next.js, Gatsby, React Native, Vue.js, Angular, Ember, Backbone, Marionette, jQuery, Styled Components, Tailwind CSS, Bootstrap, etc.
   - Backend frameworks & runtimes: Node.js, Express, Fastify, NestJS, Django, FastAPI, Flask, Spring Boot, Laravel, Ruby on Rails, ASP.NET, etc.
   - CMS & E-commerce platforms: WordPress, Shopify, Strapi, Webflow, Drupal, Magento, etc.
   - Static site generators & Templating / View engines: Eleventy (11ty), Timber, Twig, Nunjucks, Handlebars, Blade, Liquid, Pug, EJS, Astro, Hugo, etc.
   - Design & Prototyping tools: Figma, Sketch, Adobe XD, Storybook, InVision, Zeplin, etc.
   - Databases & Search: PostgreSQL, MongoDB, MySQL, Redis, Snowflake, Oracle, Elasticsearch, Algolia, Firebase, Supabase, etc.
   - Cloud, Hosting & DevOps: AWS, GCP, Azure, Vercel, Netlify, Heroku, Docker, Kubernetes, Webpack, Vite, CI/CD, Git, GitHub, GitLab, etc.
   - Testing & QA tools: Jest, Cypress, Playwright, Selenium, PyTest, Mocha, Vitest, etc.
   - API tools & protocols: RESTful API, GraphQL, gRPC, Postman, OpenAPI, WebSockets, etc.

2. MULTI-COLUMN & OCR RECOVERY:
   - Carefully scan both columns, sidebars, bullet points, and project descriptions.
   - If an OCR or text extraction artifact truncated a clear technology name (e.g. "jQuer" -> "jQuery", "Elevent" -> "Eleventy", "WordPres" -> "WordPress"), recover and extract the canonical technology name.

3. IMPLICIT SKILL DETECTION:
   - When a project or role description mentions using a technology without naming it in a dedicated "skills" section, still extract it. Example: "Built an embeddable player with MusicKit JS and Ember" -> extract "MusicKit JS", "Ember", "JavaScript".

4. SPLIT compound mentions:
   - "TypeScript/JavaScript" -> "TypeScript", "JavaScript"
   - "HTML, CSS, Sass, JavaScript, and jQuery" -> extract all 5 individually.

5. DO NOT extract:
   - Abstract methodology categories (e.g. "Software Configuration Management", "Software Engineering", "Full Stack Development", "Front End Development", "Back End Development")
   - Pure soft skills (e.g. "Leadership", "Communication", "Teamwork", "Problem Solving")
   - Version numbers (extract "React" not "React 18", "ES6" can be extracted as "ES6" or "JavaScript")

6. Return clean, canonical technology names. Respond ONLY with the valid JSON object.

CV Text:
{cv_text}"""


def _clean_and_unpack_skills(parsed: dict[str, Any]) -> dict[str, Any]:
    """Ensure skills (strings or dicts) grouped in parentheses or slashes are unpacked into individual items."""
    import re

    raw_skills = parsed.get("skills", [])
    if not isinstance(raw_skills, list):
        return parsed

    years_exp = parsed.get("years_experience")
    default_years = int(years_exp) if isinstance(years_exp, (int, float)) and years_exp > 0 else 1

    # Common OCR / truncation correction map
    ocr_corrections: dict[str, str] = {
        "jquer": "jQuery",
        "jquerp": "jQuery",
        "jquery": "jQuery",
        "wordpres": "WordPress",
        "elevent": "Eleventy",
        "11ty": "Eleventy",
        "postgre": "PostgreSQL",
        "kuberne": "Kubernetes",
    }

    unpacked_skills: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for item in raw_skills:
        if isinstance(item, str):
            orig_name = item.strip()
            item_dict: dict[str, Any] = {
                "name": orig_name,
                "category": "technical",
                "years_of_experience": default_years,
                "self_taught": False,
                "personal_projects": False,
                "has_certification": False,
            }
        elif isinstance(item, dict) and "name" in item:
            orig_name = str(item["name"]).strip()
            item_dict = dict(item)
            if "years_of_experience" not in item_dict:
                item_dict["years_of_experience"] = default_years
            if "category" not in item_dict:
                item_dict["category"] = "technical"
            if "self_taught" not in item_dict:
                item_dict["self_taught"] = False
            if "personal_projects" not in item_dict:
                item_dict["personal_projects"] = False
            if "has_certification" not in item_dict:
                item_dict["has_certification"] = False
        else:
            continue

        if not orig_name:
            continue

        # Check OCR corrections
        norm_key = orig_name.lower().strip()
        if norm_key in ocr_corrections:
            orig_name = ocr_corrections[norm_key]
            item_dict["name"] = orig_name
        elif norm_key.startswith("jquer") and len(norm_key) <= 8:
            orig_name = "jQuery"
            item_dict["name"] = orig_name
        elif norm_key.startswith("elevent") and len(norm_key) <= 10:
            orig_name = "Eleventy"
            item_dict["name"] = orig_name
        elif norm_key.startswith("wordpres") and len(norm_key) <= 11:
            orig_name = "WordPress"
            item_dict["name"] = orig_name

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
                    new_item = dict(item_dict)
                    new_item["name"] = cand
                    unpacked_skills.append(new_item)
        else:
            if "/" in orig_name and not orig_name.lower().startswith("ci/cd"):
                slash_parts = [p.strip() for p in orig_name.split("/") if p.strip()]
                for p in slash_parts:
                    p_lower = p.lower()
                    if p_lower not in seen_names and len(p) >= 2:
                        seen_names.add(p_lower)
                        new_item = dict(item_dict)
                        new_item["name"] = p
                        unpacked_skills.append(new_item)
            else:
                name_lower = orig_name.lower()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    unpacked_skills.append(item_dict)

    parsed["skills"] = unpacked_skills
    return parsed


def _parse_cv_extraction_output(raw_output: str) -> dict[str, Any]:
    """Parse JSON block from LLM output."""
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in LLM output")
        parsed = json.loads(raw_output[start:end], strict=False)
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
    if clean in ("backend", "back-end", "back_end"):
        return ["Backend"]
    if clean in ("frontend", "front-end", "front_end", "ui", "ux", "web"):
        return ["Frontend"]
    if clean in ("devops", "ci/cd", "ci_cd", "cicd", "infra", "infrastructure", "sre"):
        return ["DevOps"]
    if clean in ("cloud", "aws", "gcp", "azure"):
        return ["Cloud"]
    if clean in ("cloud_devops", "cloud/devops"):
        return ["Cloud", "DevOps"]
    if clean in (
        "data",
        "data engineering",
        "data_engineering",
        "data science",
        "data_science",
        "analytics",
        "big data",
        "big_data",
        "database",
        "sql",
    ):
        return ["Data"]
    if clean in (
        "qa",
        "testing",
        "quality assurance",
        "quality_assurance",
        "test",
        "automation_testing",
    ):
        return ["QA"]
    if clean in ("mobile", "ios", "android", "flutter", "react native", "react_native"):
        return ["Mobile"]
    # Drop all non-canonical, umbrella or generic domains (e.g. software_engineering, engineering, management, security)
    return []


# --- Domain adjacency map and routing configuration ---
DOMAIN_ADJACENCY: dict[str, set[str]] = {
    "Frontend": {"Frontend", "Mobile"},
    "Backend": {"Backend", "Frontend", "Data"},
    "Mobile": {"Mobile", "Frontend"},
    "DevOps": {"DevOps", "Cloud", "Backend"},
    "Cloud": {"Cloud", "DevOps", "Backend"},
    "Data": {"Data", "Backend", "Cloud"},
    "QA": {"QA", "DevOps", "Backend"},
}

DOMAIN_RADAR_THRESHOLD = 0.35


def _get_cluster_primary_domain(cluster: TechCluster) -> str | None:
    """Determine the primary domain of a cluster from its top-weighted skills."""
    domain_weights: dict[str, float] = {}
    for skill in cluster.centroid_skills:
        domains: list[str] = []
        if skill.core_domains:
            for d in skill.core_domains:
                domains.extend(normalize_domain_key(d))
        elif skill.domain_tags:
            for d in skill.domain_tags:
                domains.extend(normalize_domain_key(d))

        for domain in domains:
            domain_weights[domain] = domain_weights.get(domain, 0.0) + (
                skill.weight * (skill.frequency or 1.0)
            )

    if not domain_weights:
        return None
    return max(domain_weights, key=lambda k: domain_weights[k])


def _filter_affinities_by_domain(
    affinities: list[ClusterAffinity],
    active_clusters: list[TechCluster],
    domain_scores: dict[str, float],
) -> list[ClusterAffinity]:
    """Filter affinities using the user's domain radar as primary gate.

    Rules:
    1. Compute the user's dominant domain from radar domain scores.
    2. Always allow clusters whose primary domain is adjacent to dominant.
    3. Allow non-adjacent domains ONLY if user's radar score for that domain exceeds DOMAIN_RADAR_THRESHOLD (35%).
    """
    if not domain_scores or not affinities:
        return affinities

    total = sum(domain_scores.values()) or 1.0
    normalized_scores = {d: v / total for d, v in domain_scores.items()}

    dominant_domain = max(normalized_scores, key=lambda k: normalized_scores[k])
    adjacent_domains = DOMAIN_ADJACENCY.get(dominant_domain, {dominant_domain})

    cluster_domain_map: dict[UUID, str | None] = {
        c.id: _get_cluster_primary_domain(c) for c in active_clusters if c.id is not None
    }

    filtered = []
    for aff in affinities:
        cluster_domain = cluster_domain_map.get(aff.cluster_id) if aff.cluster_id else None
        if cluster_domain is None:
            filtered.append(aff)
            continue

        # Rule 1: Always allow adjacent domains
        if cluster_domain in adjacent_domains:
            filtered.append(aff)
            continue

        # Rule 2: Allow non-adjacent ONLY if user's radar score >= 35% in that domain
        user_score_for_domain = normalized_scores.get(cluster_domain, 0.0)
        if user_score_for_domain >= DOMAIN_RADAR_THRESHOLD:
            logger.info(
                "Non-adjacent domain allowed by radar threshold",
                cluster=aff.cluster_name,
                cluster_domain=cluster_domain,
                user_radar_score=round(user_score_for_domain, 2),
                threshold=DOMAIN_RADAR_THRESHOLD,
            )
            filtered.append(aff)

    return filtered if filtered else affinities[:1]


def _compute_domain_relevance_multiplier(
    cluster: TechCluster,
    user_domain_scores: dict[str, float],
    total_user_domain_score: float,
) -> float:
    """Compute a domain relevance multiplier for a cluster based on user's domain radar.

    The multiplier ranges from 0.50 to 1.0:
    - 1.0 when the cluster's primary domain is the user's dominant domain
    - Scaled down proportionally when the cluster's domain has low user affinity
    - Minimum floor of 0.50 to aggressively penalize clusters completely outside the user's focus
    """
    if total_user_domain_score <= 0:
        return 1.0

    cluster_primary_domain = _get_cluster_primary_domain(cluster)
    if cluster_primary_domain is None:
        return 0.70

    user_radar_for_domain = user_domain_scores.get(cluster_primary_domain, 0.0)
    user_radar_normalized = user_radar_for_domain / total_user_domain_score
    max_user_domain_score = max(user_domain_scores.values()) / total_user_domain_score

    if max_user_domain_score <= 0:
        return 1.0

    relative_strength = user_radar_normalized / max_user_domain_score
    floor_multiplier = 0.50
    multiplier = floor_multiplier + (1.0 - floor_multiplier) * relative_strength

    return round(multiplier, 4)


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
        elif s.domain_tags:
            for d in s.domain_tags:
                for norm_d in normalize_domain_key(d):
                    if norm_d not in domain_scores:
                        domain_scores[norm_d] = 0.0
                    domain_scores[norm_d] += s.weight * (
                        s.frequency if s.frequency is not None else 1.0
                    )
    total_domain_score = sum(domain_scores.values()) if domain_scores else 0.0

    affinities = []
    for cluster in active_clusters:
        cluster_tech_skills = [s for s in cluster.centroid_skills if s.nature == SkillNature.TECH]

        if not cluster_tech_skills:
            continue

        cluster_weight_total = (
            sum(s.weight * (s.frequency or 1.0) for s in cluster_tech_skills) or 1.0
        )
        covered_weight = 0.0
        matched_skills = []
        partial_matches = []
        missing_skills = []

        for cluster_skill in cluster_tech_skills:
            norm_name = cluster_skill.normalized_name
            w = cluster_skill.weight
            f_s = cluster_skill.frequency or 1.0
            skill_target_weight = w * f_s

            if norm_name in user_tech_norms:
                covered_weight += skill_target_weight
                matched_skills.append(cluster_skill.name)
            else:
                cluster_domains = set(cluster_skill.domain_tags or [])
                partial_match_score = 0.0
                if cluster_domains:
                    alternative_skills = [
                        u for u in user_tech_skills if set(u.domain_tags or []) & cluster_domains
                    ]
                    if alternative_skills:
                        best_alt = max(alternative_skills, key=lambda u: u.ict_score or 10.0)
                        partial_match_score = 0.5
                        partial_matches.append((cluster_skill.name, best_alt.name))

                if partial_match_score > 0.0:
                    covered_weight += skill_target_weight * partial_match_score
                else:
                    missing_skills.append(cluster_skill.name)

        raw_coverage = covered_weight / cluster_weight_total
        matching_count = len(matched_skills) + (len(partial_matches) * 0.5)
        focus_ratio = min(1.0, matching_count / max(1, len(cluster_tech_skills)))
        score = (0.85 * raw_coverage) + (0.15 * (raw_coverage * focus_ratio))

        import math

        expected_skills = 15.0
        cluster_complexity_factor = min(
            1.0, math.log(len(cluster_tech_skills) + 1) / math.log(expected_skills + 1)
        )
        score *= cluster_complexity_factor

        domain_multiplier = _compute_domain_relevance_multiplier(
            cluster, domain_scores, total_domain_score
        )
        score = round(score * domain_multiplier, 4)

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
                job_offer_count=cluster.job_offer_count,
            )
        )

    affinities.sort(key=lambda a: (a.affinity_score, len(a.detected_skills)), reverse=True)
    if not affinities:
        return None, [], [], []

    # Filter affinities using user's domain radar as primary gate
    filtered_affinities = _filter_affinities_by_domain(affinities, active_clusters, domain_scores)
    if not filtered_affinities:
        filtered_affinities = affinities

    primary = filtered_affinities[0]
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
        job_offer_count=primary.job_offer_count,
    )

    secondaries = filtered_affinities[1:3]

    # Calcular promedios de demanda de mercado por dominio normalizado
    domain_demands_accum: dict[str, list[float]] = {d: [] for d in CANONICAL_DOMAINS}
    for cluster in active_clusters:
        for skill in cluster.centroid_skills:
            domains: list[str] = []
            if skill.core_domains:
                for d in skill.core_domains:
                    domains.extend(normalize_domain_key(d))
            elif skill.domain_tags:
                for d in skill.domain_tags:
                    domains.extend(normalize_domain_key(d))

            for norm_d in domains:
                if norm_d in domain_demands_accum and skill.frequency is not None:
                    domain_demands_accum[norm_d].append(skill.frequency)

    domain_market_demand: dict[str, float] = {}
    for d, freqs in domain_demands_accum.items():
        if freqs:
            normalized_freqs = [_normalize_demand_percentage(f) / 100.0 for f in freqs]
            domain_market_demand[d] = round(sum(normalized_freqs) / len(normalized_freqs), 4)
        else:
            domain_market_demand[d] = 0.50

    total_domain_score = sum(domain_scores.values()) if domain_scores else 1.0
    domain_affinities_dto = [
        DomainAffinityDTO(
            domain=d,
            affinity_score=round(domain_scores.get(d, 0.0) / total_domain_score, 4)
            if total_domain_score > 0
            else 0.0,
            market_demand=domain_market_demand.get(d, 0.50),
        )
        for d in CANONICAL_DOMAINS
    ]
    domain_affinities_dto.sort(
        key=lambda x: (x.affinity_score, CANONICAL_DOMAINS.index(x.domain)),
        reverse=False,
    )
    # Sort descending by score, and by canonical order on ties
    domain_affinities_dto.sort(key=lambda x: x.affinity_score, reverse=True)

    return primary, secondaries, filtered_affinities, domain_affinities_dto


class GetKnowledgeGraphUseCase:
    """Builds a Knowledge Graph representation for frontend visualization."""

    def __init__(
        self,
        skill_repository: SkillRepository,
        profile_repository: UserProfileRepository,
        cluster_repository: ClusterRepository | None = None,
    ) -> None:
        self._skills = skill_repository
        self._profiles = profile_repository
        self._clusters = cluster_repository

    async def execute(self, user_id: UUID | None = None, cluster_name: str | None = None) -> Any:
        from src.ml_engine.application.dtos import GraphLinkDTO, GraphNodeDTO, GraphResponseDTO

        # When the user is authenticated, build a focused graph scoped to their
        # own detected skills, gaps, and top cluster context skills (~60-80 nodes total).
        if user_id:
            return await self._build_user_graph(user_id, cluster_name)

        # --- Unauthenticated / global explorer path ---
        # Fetch bounded representative skills (top ~60-80 skills max) rather than full catalog
        selected_skills: list[Skill] = []
        if self._clusters:
            all_clusters = await self._clusters.get_all_active()
            target_cluster = None
            if cluster_name:
                for c in all_clusters:
                    if c.name.lower() == cluster_name.lower():
                        target_cluster = c
                        break
            if target_cluster and target_cluster.centroid_skills:
                selected_skills = target_cluster.centroid_skills[:60]
            elif all_clusters:
                # Take top representative skills across clusters
                seen_norm: set[str] = set()
                for c in all_clusters:
                    for s in c.centroid_skills:
                        if s.normalized_name not in seen_norm:
                            seen_norm.add(s.normalized_name)
                            selected_skills.append(s)
                            if len(selected_skills) >= 60:
                                break
                    if len(selected_skills) >= 60:
                        break

        if not selected_skills:
            all_skills = await self._skills.get_all_skills()
            selected_skills = sorted(
                all_skills, key=lambda s: getattr(s, "weight", 1.0), reverse=True
            )[:60]

        nodes = [
            GraphNodeDTO(
                id=s.normalized_name,
                label=s.name,
                group=s.nature.value if hasattr(s, "nature") and s.nature else "tech",
                domains=s.domain_tags if hasattr(s, "domain_tags") and s.domain_tags else [],
                status="neutral",
            )
            for s in selected_skills
        ]

        seen_links: set[tuple[str, str]] = set()
        links: list[GraphLinkDTO] = []
        skill_by_name = {s.normalized_name: s for s in selected_skills}

        for s in selected_skills:
            if hasattr(s, "relations") and s.relations:
                for rel in s.relations:
                    target_name = (
                        rel.target_skill_name.lower().replace(" ", "").replace(".", "")
                        if rel.target_skill_name
                        else ""
                    )
                    if target_name and target_name in skill_by_name:
                        u, v = s.normalized_name, target_name
                        edge: tuple[str, str] = (u, v) if u < v else (v, u)
                        if edge not in seen_links:
                            seen_links.add(edge)
                            links.append(
                                GraphLinkDTO(
                                    source=s.normalized_name,
                                    target=target_name,
                                    value=2.0,
                                    type=f"explicit_{rel.relation_type.value if hasattr(rel.relation_type, 'value') else rel.relation_type}",
                                )
                            )

        domain_map: dict[str, list[str]] = {}
        for s in selected_skills:
            if hasattr(s, "domain_tags") and s.domain_tags:
                for d in s.domain_tags:
                    domain_map.setdefault(d, [])
                    if s.normalized_name not in domain_map[d]:
                        domain_map[d].append(s.normalized_name)

        for skill_names in domain_map.values():
            for i in range(len(skill_names) - 1):
                u, v = skill_names[i], skill_names[i + 1]
                edge = (u, v) if u < v else (v, u)
                if edge not in seen_links:
                    seen_links.add(edge)
                    links.append(
                        GraphLinkDTO(
                            source=skill_names[i],
                            target=skill_names[i + 1],
                            value=0.5,
                            type="implicit_domain",
                        )
                    )

        return GraphResponseDTO(nodes=nodes, links=links)

    async def _build_user_graph(
        self,
        user_id: UUID,
        cluster_name: str | None,
    ) -> Any:
        """Build a knowledge graph scoped to a user's detected skills, skill gaps,
        and top relevant cluster market context skills (bounded to ~80 nodes total).
        """
        from src.ml_engine.application.dtos import GraphLinkDTO, GraphNodeDTO, GraphResponseDTO

        profile = await self._profiles.get_by_user_id(user_id)

        if not profile:
            return GraphResponseDTO(nodes=[], links=[])

        target_affinity = None
        if cluster_name:
            if (
                profile.primary_affinity
                and profile.primary_affinity.cluster_name.lower() == cluster_name.lower()
            ):
                target_affinity = profile.primary_affinity
            else:
                for a in profile.secondary_affinities:
                    if a.cluster_name.lower() == cluster_name.lower():
                        target_affinity = a
                        break

        if not target_affinity and profile.primary_affinity:
            target_affinity = profile.primary_affinity

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

        # Fetch top relevant market skills for cluster context (max 35-40 items)
        market: dict[str, Skill] = {}
        if self._clusters:
            all_clusters = await self._clusters.get_all_active()
            target_cluster = None
            resolved_cluster_name = cluster_name or (
                target_affinity.cluster_name if target_affinity else None
            )
            if resolved_cluster_name:
                for c in all_clusters:
                    if c.name.lower() == resolved_cluster_name.lower():
                        target_cluster = c
                        break

            if target_cluster and target_cluster.centroid_skills:
                for s in target_cluster.centroid_skills:
                    if (
                        s.normalized_name not in acquired
                        and s.normalized_name not in gaps
                        and s.normalized_name not in neutral
                    ):
                        market[s.normalized_name] = s
                        if len(market) >= 40:
                            break

            # If fewer than 25 market skills in target cluster, add top skills from other clusters
            if len(market) < 25:
                for c in all_clusters:
                    if target_cluster and c.id == target_cluster.id:
                        continue
                    for s in c.centroid_skills:
                        if (
                            s.normalized_name not in acquired
                            and s.normalized_name not in gaps
                            and s.normalized_name not in neutral
                            and s.normalized_name not in market
                        ):
                            market[s.normalized_name] = s
                            if len(market) >= 35:
                                break
                    if len(market) >= 35:
                        break

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

        seen_links: set[tuple[str, str]] = set()
        links: list[GraphLinkDTO] = []
        skill_by_name = {s.normalized_name: s for s in all_skills_to_render}

        # 1. Explicit relations
        for s in all_skills_to_render:
            if hasattr(s, "relations") and s.relations:
                for rel in s.relations:
                    target_name = (
                        rel.target_skill_name.lower().replace(" ", "").replace(".", "")
                        if rel.target_skill_name
                        else ""
                    )
                    if target_name and target_name in skill_by_name:
                        u, v = s.normalized_name, target_name
                        edge: tuple[str, str] = (u, v) if u < v else (v, u)
                        if edge not in seen_links:
                            seen_links.add(edge)
                            links.append(
                                GraphLinkDTO(
                                    source=s.normalized_name,
                                    target=target_name,
                                    value=2.0,
                                    type=f"explicit_{rel.relation_type.value if hasattr(rel.relation_type, 'value') else rel.relation_type}",
                                )
                            )

        # 2. Implicit domain connections bounded
        domain_map: dict[str, list[str]] = {}
        for s in all_skills_to_render:
            if hasattr(s, "domain_tags") and s.domain_tags:
                for d in s.domain_tags:
                    domain_map.setdefault(d, [])
                    if s.normalized_name not in domain_map[d]:
                        domain_map[d].append(s.normalized_name)

        for skill_names in domain_map.values():
            for i in range(len(skill_names) - 1):
                u, v = skill_names[i], skill_names[i + 1]
                edge = (u, v) if u < v else (v, u)
                if edge not in seen_links:
                    seen_links.add(edge)
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
        active_clusters = [
            c
            for c in active_clusters
            if c.centroid_skills and getattr(c, "tier", "standard") != "low"
        ]

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
                    job_offer_count=a.job_offer_count,
                    top_skills=a.top_skills,
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
                    job_offer_count=a.job_offer_count,
                    top_skills=a.top_skills,
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
            last_analysis_date=profile.last_analysis_date,
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
        total_demand = int(
            raw_insights.get("total_demand")
            or requested_cluster.job_offer_count
            or affinity.job_offer_count
            or 1
        )

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

        eur_rate = 0.92
        salary_projection_dto = {
            "current_estimated_salary_usd": current_estimated_salary_usd,
            "current_estimated_salary_pen": round(current_estimated_salary_usd * 3.75, 2),
            "current_estimated_salary_eur": round(current_estimated_salary_usd * eur_rate, 2),
            "projected_salary_usd": projected_salary_usd,
            "projected_salary_pen": round(projected_salary_usd * 3.75, 2),
            "projected_salary_eur": round(projected_salary_usd * eur_rate, 2),
            "potential_gain_percentage": max(0.0, potential_gain_percentage),
            "cluster_average_usd": avg_usd,
            "cluster_p75_usd": p75_usd,
            "salary_p25_usd": p25_usd,
            "salary_p25_pen": round(p25_usd * 3.75, 2),
            "salary_p25_eur": round(p25_usd * eur_rate, 2),
            "salary_median_usd": p50_usd,
            "salary_median_pen": round(p50_usd * 3.75, 2),
            "salary_median_eur": round(p50_usd * eur_rate, 2),
            "salary_p75_pen": round(p75_usd * 3.75, 2),
            "salary_p75_eur": round(p75_usd * eur_rate, 2),
            "market_tier": raw_insights.get("market_tier", "Tier 2 (Tech & Remoto Global)"),
            "salary_differential_percentage": raw_insights.get(
                "salary_differential_percentage", 0.0
            ),
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
            job_offer_count=requested_cluster.job_offer_count or affinity.job_offer_count,
            top_skills=affinity.top_skills,
            market_insights=requested_cluster.market_insights or affinity.market_insights,
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
