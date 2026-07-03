"""ML Engine use cases."""

import json
from collections import deque
from collections.abc import Awaitable, Callable
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

                stamped_skill = dc_replace(
                    skill,
                    self_taught=self_taught,
                    personal_projects=personal_projects,
                    years_of_experience=years_exp,
                    has_certification=has_cert,
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

        # Start with the explicitly detected skills, keyed by ID for O(1) dedup
        inferred_skills: dict[UUID, Skill] = {s.id: s for s in skills if s.id}
        to_process: deque[Skill] = deque(s for s in skills if s.id)

        _upward_types = {SkillRelationType.BELONGS_TO, SkillRelationType.REQUIRES}

        while to_process:
            current_skill = to_process.popleft()
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

        return list(inferred_skills.values())

    async def _enrich_skills_evidence(
        self, cv_text: str, raw_skill_names: list[str]
    ) -> list[dict[str, Any]] | list[str]:
        """Phase 1.5: Enrich raw skill names with evidence via a second LLM call.

        Returns a list of dicts with evidence for each skill (name, category,
        years_of_experience, personal_projects, has_certification).
        If the LLM call fails, returns the original list of skill names as-is.
        """
        if not raw_skill_names:
            return raw_skill_names

        logger.info(
            "Phase 1.5 — enriching skills evidence",
            skill_count=len(raw_skill_names),
        )
        try:
            prompt = _build_skill_evidence_prompt(cv_text, raw_skill_names)
            raw_output = await self._llm.generate(
                prompt=prompt, context=[], max_tokens=2500
            )
            parsed = _parse_cv_extraction_output(raw_output)
            enriched = parsed.get("skills", [])
            if not enriched:
                logger.warning("Phase 1.5 returned empty skills list, using originals")
                return raw_skill_names
            return enriched
        except Exception as exc:
            logger.warning(
                "Phase 1.5 LLM enrichment failed, using original skill names",
                error=str(exc),
            )
            return raw_skill_names

    async def execute(
        self,
        user_id: UUID,
        cv_id: UUID,
        cv_content: bytes,
        content_type: str,
        on_phase1_complete: "Callable[[UUID], Awaitable[None]] | None" = None,
    ) -> UserProfileDTO:
        """Run the 3-phase CV analysis pipeline.

        Phase 1 — Profile Extraction (~5-8s):
            1. Extract text from CV file.
            2. Classify document as CV via heuristic.
            3. Run lightweight LLM extraction (role, summary, years, flat skill names).
            4. Persist basic profile (no diagnostics yet).
            5. Call ``on_phase1_complete`` so the caller can mark the CV as
               'completed' and let the frontend know the profile is ready.

        Phase 1.5 — Skill Evidence Enrichment (~10-15s, after Phase 1 callback):
            6. Second LLM call to enrich each skill with category, years of
               experience, personal projects, and certification evidence.

        Phase 2 — Diagnosis (~3-5s, continues after Phase 1.5):
            7. Normalise user skills against the canonical catalog.
            8. Upward-inference on the skill knowledge graph.
            9. Load active clusters and compute Weighted Jaccard affinity.
            10. Detect skill gaps vs primary cluster.
            11. Persist enriched profile with ``is_diagnosed=True``.

        The three-phase design lets the frontend render a useful profile in < 10s
        while skill evidence enrichment and diagnosis finish silently in the background.
        """
        try:
            # ── Phase 1: Fast LLM extraction ─────────────────────────────────

            # Step 1: Extract text from CV
            logger.info("Phase 1 — extracting CV text", user_id=str(user_id))
            cv_text = await self._cv_parser.extract_text(cv_content, content_type)

            if not cv_text.strip():
                raise MLPipelineError("CV text extraction returned empty content")

            # Step 1.5: Validate document is actually a CV
            is_cv, confidence = await self._classify_as_cv(cv_text)
            if not is_cv:
                raise MLPipelineError(
                    "The uploaded document does not appear to be a professional CV/resume. "
                    "Please upload a document with your work experience, education, and skills."
                )
            logger.debug("Document classified as CV", confidence=confidence)

            # Step 2: Run LLM structured extraction (lightweight).
            # Truncate to 4000 chars to handle 2-page CVs while keeping
            # the context window responsive. max_tokens=1000 is enough
            # since this phase only extracts role, summary, years, and flat skill names.
            logger.info("Phase 1 — LLM structured extraction", user_id=str(user_id))
            CV_TEXT_CHAR_LIMIT = 4000
            cv_text_for_llm = cv_text[:CV_TEXT_CHAR_LIMIT]
            prompt = _build_cv_extraction_prompt(cv_text_for_llm)
            raw_llm_output = await self._llm.generate(
                prompt=prompt, context=[], max_tokens=1000
            )
            extracted_data = _parse_cv_extraction_output(raw_llm_output)
            if not extracted_data:
                raise ValueError("Empty extraction data parsed")
            if "error" in extracted_data and extracted_data["error"] == "not_a_cv":
                doc_type = extracted_data.get("document_type", "unknown")
                raise MLPipelineError(
                    f"The uploaded document does not appear to be a CV/resume "
                    f"(detected as: {doc_type}). "
                    "Please upload a document with your work experience, education, and skills."
                )

            # Step 3: Persist Phase 1 profile immediately (no diagnostics, no skills).
            # This lets the CV status be marked 'completed' right after, so the
            # frontend polling sees it and renders the profile within ~7s.
            logger.info("Phase 1 — persisting basic profile", user_id=str(user_id))
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

            phase1_profile = UserProfile(
                user_id=user_id,
                cv_id=cv_id,
                embedding=[],
                detected_skills=[],
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
                years_experience=int(years_exp) if isinstance(years_exp, (int, float)) else None,
                cv_raw_text=cv_text,
                is_diagnosed=False,
            )
            await self._profiles.save_profile(phase1_profile, persist_diagnostics=False)

            # Notify the caller (router) so it can mark the CV as 'completed'
            # and commit the session — the frontend polling will pick this up.
            if on_phase1_complete:
                await on_phase1_complete(cv_id)

            logger.info(
                "Phase 1 complete — basic profile saved and CV marked completed",
                user_id=str(user_id),
            )

            # ── Phase 1.5: Skill evidence enrichment ───────────────────────────
            # Second LLM call to enrich flat skill names with evidence details.
            # Runs after the Phase 1 callback so the frontend already sees
            # status="completed" before this starts.
            raw_skills = extracted_data.get("skills", [])
            if raw_skills:
                logger.info(
                    "Phase 1.5 — enriching skills evidence",
                    user_id=str(user_id),
                    skill_count=len(raw_skills),
                )
                raw_skills = await self._enrich_skills_evidence(cv_text, raw_skills)

            # ── Phase 2: Skill enrichment + cluster affinity ──────────────────

            # Embedding (static zero-vector for backwards compatibility)
            cv_embedding = [0.0] * 1024

            # Normalize user skills (no LLM fallback).
            # Phase 1.5 enriched skills with evidence; unresolvable skills
            # default to catalog data.
            logger.info("Phase 2 — normalising skills", user_id=str(user_id))

            # Load the full skill catalogue once and share it across
            # resolve_skills and _expand_with_upward_inference to avoid a
            # redundant DB round-trip.
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
                    full_name=phase1_profile.full_name,
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
                    full_name=phase1_profile.full_name,
                    message="Profile saved. Diagnosis skipped — clusters have no centroid skills.",
                )

            # Compute Weighted Jaccard Similarity per cluster
            primary, _secondaries, _affinities, _ = compute_affinities_and_domains(
                detected_skills, active_clusters
            )
            if not primary:
                logger.warning(
                    "No primary cluster affinity — skipping Phase 2",
                    user_id=str(user_id),
                )
                return UserProfileDTO(
                    user_id=user_id,
                    cv_id=cv_id,
                    seniority=seniority.value,
                    primary_specialty="Sin Diagnóstico",
                    alignment_score=0.0,
                    full_name=phase1_profile.full_name,
                    message="Profile saved. Could not compute cluster affinity.",
                )

            # Detect skill gaps vs primary cluster
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
                            SkillGap(skill=skill, market_importance=importance)
                        )
                skill_gaps.sort(key=lambda g: g.skill.weight * g.skill.frequency, reverse=True)

            # Persist enriched profile with is_diagnosed=True
            logger.info("Phase 2 — persisting full diagnosis", user_id=str(user_id))
            diagnosed_profile = UserProfile(
                user_id=user_id,
                cv_id=cv_id,
                embedding=cv_embedding,
                detected_skills=detected_skills,
                seniority=seniority,
                primary_affinity=primary,
                secondary_affinities=[],
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
                cv_raw_text=cv_text,
                is_diagnosed=True,
            )
            await self._profiles.save(diagnosed_profile)

            logger.info(
                "Phase 2 complete — full diagnosis persisted",
                user_id=str(user_id),
                specialty=primary.cluster_name,
                score=primary.affinity_score,
            )

            # Compute Domain Affinities for DTO
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
    """Build structured prompt for Phase 1 LLM CV extraction.

    Lightweight extraction: role, summary, years, and skills as flat strings.
    Skills evidence enrichment happens in a separate Phase 1.5 call.
    """
    return f"""You are a professional CV analyzer.

FIRST, determine if the text below is a professional CV/resume (currículum vitae).
A CV typically contains personal information, work experience, education history, and skills.

If the text is NOT a CV (e.g., it is an invoice, letter, contract, terms of service, or any other document), respond with EXACTLY:
{{"error": "not_a_cv", "document_type": "<brief description of what the document appears to be>"}}

If the text IS a CV, extract the following details in a structured JSON format:
1. Current Job Role (current_job_role)
2. Professional Summary (professional_summary): a 1-2 sentence summary of the candidate's profile
3. Years of experience (years_experience, integer): total years of professional experience
4. Technical & Soft Skills (skills): list of skill NAMES as flat strings. Extract EVERY SINGLE programming language, database, framework, library, tool, cloud provider, methodology, architecture, and soft skill mentioned in the CV. Do NOT summarize, group, or omit any. The list should be exhaustive (typically 20-50 items for a technical profile).

CV Text:
{cv_text}

Respond ONLY with a valid JSON object. If the text is not a CV, use the error format above.
If it IS a CV, use this schema:
{{
  "current_job_role": "string or null",
  "professional_summary": "string or null",
  "years_experience": integer or null,
  "skills": ["string", "string", ...]
}}"""


def _build_skill_evidence_prompt(cv_text: str, skill_names: list[str]) -> str:
    """Build prompt for Phase 1.5 skill evidence enrichment.

    Takes the CV text and the list of skill names from Phase 1, asks the LLM
    to extract evidence details (category, years of experience, etc.) for each.
    """
    skill_list = "\n".join(f"- {s}" for s in skill_names)
    return f"""You are a professional CV analyzer.

Given the CV text below and a list of skill names extracted from it, find evidence in the CV for each skill and return a JSON object with a "skills" array.

For each skill, extract:
- name: the skill name exactly as provided
- category: one of "technical" (languages, databases, frameworks, cloud), "soft" (soft skills), "tools" (software/tools), "methodologies" (methodologies, architectures)
- years_of_experience: integer, estimated years the candidate has used this skill based on work experience dates
- personal_projects: boolean, true if the CV mentions using this skill in personal or open-source projects
- has_certification: boolean, true if the CV mentions an official certification for this skill

Be precise. Only set personal_projects or has_certification to true if there is explicit evidence in the CV text.

CV Text:
{cv_text}

Skills to analyze:
{skill_list}

Respond ONLY with a valid JSON object using this exact schema:
{{
  "skills": [
    {{
      "name": "string",
      "category": "technical | soft | tools | methodologies",
      "years_of_experience": integer,
      "personal_projects": boolean,
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

        # Fetch all non-ESCO skills to render as the general market backdrop
        non_esco_skills = await self._skills.get_non_esco_skills()
        market = {
            s.normalized_name: s
            for s in non_esco_skills
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
        from dataclasses import replace

        from fastapi import HTTPException

        profile = await self._profiles.get_by_user_id(user_id)
        if not profile:
            raise HTTPException(
                status_code=404, detail="No profile found. Please upload a CV first."
            )

        # Find if it already exists
        all_affinities = [profile.primary_affinity, *profile.secondary_affinities]
        # Clean 'Sin Diagnóstico' out if it was empty
        all_affinities = [a for a in all_affinities if a.cluster_name != "Sin Diagnóstico"]

        affinity = next(
            (a for a in all_affinities if a.cluster_name.lower() == cluster_name.lower()), None
        )

        if not affinity:
            # We must compute it on the fly!
            active_clusters = await self._clusters.get_all_active()
            requested_cluster = next(
                (c for c in active_clusters if c.name.lower() == cluster_name.lower()), None
            )
            if not requested_cluster:
                raise HTTPException(status_code=404, detail=f"Cluster '{cluster_name}' not found.")

            # Compute
            _, _, affinities, _ = compute_affinities_and_domains(
                profile.detected_skills, [requested_cluster]
            )
            if not affinities:
                raise HTTPException(status_code=500, detail="Failed to compute affinity score.")

            new_affinity = affinities[0]
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
                job_offer_count=new_affinity.job_offer_count,
                top_skills=new_affinity.top_skills,
            )

            # Save
            existing_secondaries = [
                a for a in profile.secondary_affinities if a.cluster_name != requested_cluster.name
            ]
            updated_secondaries = [*existing_secondaries, new_affinity]
            updated_profile = replace(profile, secondary_affinities=updated_secondaries)
            await self._profiles.save(updated_profile)

            # Reload profile
            profile = await self._profiles.get_by_user_id(user_id)
            if not profile:
                raise HTTPException(status_code=500, detail="Profile lost after saving diagnostic.")

            all_affinities = [profile.primary_affinity, *profile.secondary_affinities]
            affinity = next(
                (a for a in all_affinities if a.cluster_name.lower() == cluster_name.lower()), None
            )
            if not affinity:
                raise HTTPException(status_code=500, detail="Failed to retrieve computed affinity.")

        # Expose top skills and job offer count
        # In case we need domain affinities for the radar chart
        active_clusters = await self._clusters.get_all_active()
        active_clusters = [c for c in active_clusters if c.centroid_skills]
        domain_affinities_dto = compute_domain_affinities(profile.detected_skills, active_clusters)

        user_skills_map = {s.normalized_name: s for s in profile.detected_skills}

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
                    market_demand_percentage=round(g.skill.frequency * 100)
                    if g.skill.frequency is not None
                    else None,
                    trend=determine_trend(g.skill.name),
                )
                for g in affinity.skill_gaps
            ],
            domain_affinities=domain_affinities_dto,
            total_profile_skills=len(profile.detected_skills),
        )
