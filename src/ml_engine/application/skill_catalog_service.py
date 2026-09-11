"""Service for O(1) skill resolution with LLM fallback."""

import json
import re
from typing import Any
from uuid import uuid4

import structlog

from src.ml_engine.domain.entities import Skill, SkillNature
from src.ml_engine.domain.ports import LLMService, SkillRepository

logger = structlog.get_logger(__name__)


def _match_single_skill(
    raw: str,
    alias_to_skill: dict[str, Skill],
    norm_to_skill: dict[str, Skill],
) -> Skill | None:
    """Matches a raw skill string to a canonical Skill entity using multiple strategies."""
    clean = raw.strip()
    if not clean:
        return None

    # 1. Exact alias lookup
    lower = clean.lower()
    if lower in alias_to_skill:
        return alias_to_skill[lower]

    # 2. Normalized name lookup (spaces/dots/hyphens stripped)
    norm = lower.replace(" ", "").replace(".", "").replace("-", "")
    if norm in norm_to_skill:
        return norm_to_skill[norm]

    # 3. Check inside parentheses: e.g. "NoSQL (MongoDB)" -> "MongoDB"
    paren_match = re.search(r"\((.*?)\)", clean)
    if paren_match:
        inner = paren_match.group(1).strip()
        inner_lower = inner.lower()
        if inner_lower in alias_to_skill:
            return alias_to_skill[inner_lower]
        inner_norm = inner_lower.replace(" ", "").replace(".", "").replace("-", "")
        if inner_norm in norm_to_skill:
            return norm_to_skill[inner_norm]

        # Also check outside parentheses: e.g. "MongoDB (NoSQL)" -> "MongoDB"
        outer = re.sub(r"\(.*?\)", "", clean).strip()
        outer_lower = outer.lower()
        if outer_lower in alias_to_skill:
            return alias_to_skill[outer_lower]
        outer_norm = outer_lower.replace(" ", "").replace(".", "").replace("-", "")
        if outer_norm in norm_to_skill:
            return norm_to_skill[outer_norm]

    # 4. Check stripping common conversational prefixes
    for prefix in ("framework ", "librería ", "libreria ", "library ", "lenguaje ", "language "):
        if lower.startswith(prefix):
            stripped = clean[len(prefix) :].strip()
            res = _match_single_skill(stripped, alias_to_skill, norm_to_skill)
            if res:
                return res

    return None


class SkillCatalogService:
    def __init__(self, skill_repository: SkillRepository, llm_service: LLMService):
        self._skills = skill_repository
        self._llm = llm_service

    async def normalize_extracted_skills(
        self,
        raw_skills: list[dict[str, Any]],
        existing_skills_cache: list[Skill] | None = None,
    ) -> list[dict[str, Any]]:
        """Pre-normalizes LLM-extracted skill dicts against canonical Lightcast catalog in Phase 1.

        Updates 'name' to the canonical skill name if matched, retaining 'original_raw_name'.
        Marks 'is_custom': False, 'in_catalog': True for catalog skills,
        and 'is_custom': True, 'in_catalog': False for unrecognized skills.
        Deduplicates items by canonical name while merging evidence.
        """
        if not raw_skills:
            return []

        if existing_skills_cache is not None:
            existing_skills = existing_skills_cache
        else:
            existing_skills = await self._skills.get_all_skills()

        alias_to_skill: dict[str, Skill] = {}
        norm_to_skill: dict[str, Skill] = {}
        for skill in existing_skills:
            norm_to_skill[skill.normalized_name] = skill
            for alias in skill.aliases:
                alias_to_skill[alias.lower()] = skill

        dedup_map: dict[str, dict[str, Any]] = {}

        for item in raw_skills:
            if not isinstance(item, dict):
                continue
            raw_name = str(item.get("name", "")).strip()
            if not raw_name:
                continue

            matched = _match_single_skill(raw_name, alias_to_skill, norm_to_skill)

            norm_key: str
            processed_item: dict[str, Any] = dict(item)

            if matched:
                canonical_name = matched.name
                norm_key = canonical_name.lower()
                processed_item["name"] = canonical_name
                processed_item["original_raw_name"] = raw_name
                processed_item["in_catalog"] = True
                processed_item["is_custom"] = False
                processed_item["category"] = (
                    "concept" if matched.nature == SkillNature.CONCEPT else "technical"
                )
            else:
                norm_key = raw_name.lower()
                processed_item["name"] = raw_name
                processed_item["in_catalog"] = False
                processed_item["is_custom"] = True

            if norm_key in dedup_map:
                # Merge evidence
                existing = dedup_map[norm_key]
                existing["years_of_experience"] = max(
                    int(existing.get("years_of_experience", 0) or 0),
                    int(processed_item.get("years_of_experience", 0) or 0),
                )
                existing["personal_projects"] = bool(
                    existing.get("personal_projects", False)
                ) or bool(processed_item.get("personal_projects", False))
                existing["has_certification"] = bool(
                    existing.get("has_certification", False)
                ) or bool(processed_item.get("has_certification", False))
                existing["self_taught"] = bool(existing.get("self_taught", False)) or bool(
                    processed_item.get("self_taught", False)
                )
            else:
                dedup_map[norm_key] = processed_item

        return list(dedup_map.values())

    async def resolve_skills(
        self,
        raw_strings: list[str],
        use_llm_fallback: bool = True,
        existing_skills_cache: list[Skill] | None = None,
    ) -> list[Skill]:
        """Resolves a list of raw skill strings to canonical Skill entities.

        Args:
            raw_strings: Raw skill strings extracted from the CV.
            use_llm_fallback: Whether to use the LLM to classify unresolved
                skills. Phase 2 enrichment sets this to False to avoid an
                extra LLM call. When False, unresolved skills are preserved as
                custom Skill entities instead of being discarded.
            existing_skills_cache: Pre-loaded skill catalogue to avoid a
                redundant DB round-trip. If omitted the catalogue is loaded
                from the database.
        """
        # Clean inputs
        clean_strings = []
        for raw_str in raw_strings:
            if isinstance(raw_str, str) and raw_str.strip():
                clean_strings.append(raw_str.strip())

        if not clean_strings:
            return []

        # 1. Load existing skills (use cache if provided to avoid double load)
        if existing_skills_cache is not None:
            existing_skills = existing_skills_cache
        else:
            existing_skills = await self._skills.get_all_skills()

        # Build lookup maps
        alias_to_skill = {}
        norm_to_skill = {}
        for skill in existing_skills:
            norm_to_skill[skill.normalized_name] = skill
            for alias in skill.aliases:
                alias_to_skill[alias.lower()] = skill

        resolved_skills = []
        unresolved_strings = []

        # 2. Match with aliases and smart clean rules
        for raw in clean_strings:
            matched = _match_single_skill(raw, alias_to_skill, norm_to_skill)
            if matched:
                resolved_skills.append(matched)
            else:
                unresolved_strings.append(raw)

        if not unresolved_strings:
            # Deduplicate by ID
            return list({sk.id: sk for sk in resolved_skills if sk.id}.values())

        if not use_llm_fallback:
            # Preserve custom skills so they are not dropped in Phase 2!
            seen_norms = {sk.normalized_name for sk in resolved_skills}
            for raw in unresolved_strings:
                clean_name = raw.strip()
                norm_name = clean_name.lower().replace(" ", "").replace(".", "").replace("-", "")
                if norm_name in seen_norms:
                    continue
                seen_norms.add(norm_name)
                custom_skill = Skill(
                    id=uuid4(),
                    name=clean_name,
                    nature=SkillNature.TECH,
                    normalized_name=norm_name,
                    domain_tags=["custom"],
                    core_domains=[],
                    aliases=[clean_name.lower()],
                    weight=1.0,
                    is_custom=True,
                )
                resolved_skills.append(custom_skill)

            return list({sk.id: sk for sk in resolved_skills if sk.id}.values())

        # 3. LLM Fallback for missing skills
        logger.info("Resolving unknown skills via LLM", count=len(unresolved_strings))
        new_skills = await self._classify_with_llm(list(set(unresolved_strings)))

        if new_skills:
            # Dedup by normalized name before saving
            unique_new_skills = {}
            for ns in new_skills:
                unique_new_skills[ns.normalized_name] = ns
            saved_skills = await self._skills.save_skills(list(unique_new_skills.values()))
            resolved_skills.extend(saved_skills)

        # Deduplicate
        unique_resolved = {sk.id: sk for sk in resolved_skills if sk.id}
        unique_list = list(unique_resolved.values())
        if len(unique_list) < len(resolved_skills):
            for sk in resolved_skills:
                if not sk.id and sk not in unique_list:
                    unique_list.append(sk)

        return unique_list

    async def _classify_with_llm(self, unknown_strings: list[str]) -> list[Skill]:
        """Uses LLM to classify unknown skill strings into canonical nodes."""
        prompt = f"""
You are an expert IT Skill Classifier.
Analyze the following list of raw skills and normalize them into a Knowledge Graph format.
For each skill, provide:
- canonical_name: The standard, capitalized name of the technology/concept (e.g., "React", "PostgreSQL", "Microservices").
- nature: Must be exactly one of: "concept", "tech".
- domain_tags: A list of specific, customized tags representing detailed sub-domains or categories (e.g., ["web", "frontend", "spa"], ["database", "relational"], ["microservices", "api"], ["cloud", "serverless"]).
- core_domains: A list of general core domains. MUST be selected strictly from: ["Backend", "Frontend", "Mobile", "QA", "DevOps", "Cloud", "Data"]. If a skill doesn't fit any, return an empty list.
- aliases: A list of common alternate spellings or raw inputs that should map to this (including the raw input provided).

Raw skills to classify:
{json.dumps(unknown_strings, indent=2)}

Return ONLY a valid JSON array of objects, with no markdown formatting or extra text.
Format:
{{
  "skills": [
    {{
      "canonical_name": "...",
      "nature": "...",
      "domain_tags": ["...", "..."],
      "core_domains": ["...", "..."],
      "aliases": ["...", "..."]
    }}
  ]
}}
"""
        try:
            raw_output = await self._llm.generate(prompt=prompt, context=[])
            # Clean possible markdown block
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3]
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:-3]

            data = json.loads(raw_output.strip())
            items = data.get("skills", [])

            new_skills = []
            for item in items:
                try:
                    nature = SkillNature(item["nature"].lower())
                except ValueError:
                    nature = SkillNature.TECH

                norm_name = item["canonical_name"].lower().replace(" ", "").replace(".", "")

                # Make sure aliases contain the original raw strings
                aliases = set([a.lower() for a in item.get("aliases", [])])

                new_skills.append(
                    Skill(
                        name=item["canonical_name"],
                        nature=nature,
                        normalized_name=norm_name,
                        domain_tags=item.get("domain_tags", []),
                        core_domains=item.get("core_domains", []),
                        aliases=list(aliases),
                        weight=1.0,
                    )
                )
            return new_skills
        except Exception as e:
            logger.error("Failed to classify skills with LLM", error=str(e))
            # Fallback for resiliency: create basic tech skills
            fallback_skills = []
            for raw in unknown_strings:
                norm_name = raw.lower().replace(" ", "").replace(".", "")
                fallback_skills.append(
                    Skill(
                        name=raw,
                        nature=SkillNature.TECH,
                        normalized_name=norm_name,
                        domain_tags=["Unknown"],
                        core_domains=[],
                        aliases=[raw.lower()],
                    )
                )
            return fallback_skills
