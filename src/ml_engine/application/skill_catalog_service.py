"""Service for O(1) skill resolution with LLM fallback."""

import json

import structlog

from src.ml_engine.domain.entities import Skill, SkillNature
from src.ml_engine.domain.ports import LLMService, SkillRepository

logger = structlog.get_logger(__name__)


class SkillCatalogService:
    def __init__(self, skill_repository: SkillRepository, llm_service: LLMService):
        self._skills = skill_repository
        self._llm = llm_service

    async def resolve_skills(self, raw_strings: list[str]) -> list[Skill]:
        """Resolves a list of raw skill strings to canonical Skill entities."""
        # Clean inputs
        clean_strings = []
        for raw_str in raw_strings:
            if isinstance(raw_str, str) and raw_str.strip():
                clean_strings.append(raw_str.strip())

        if not clean_strings:
            return []

        # 1. Load existing skills to check cache
        # In a real heavy app, we might just query the DB for the specific aliases.
        # But we'll use get_all_skills for now to keep it simple.
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

        # 2. Exact match (O(1))
        for raw in clean_strings:
            norm_raw = raw.lower()
            if norm_raw in alias_to_skill:
                resolved_skills.append(alias_to_skill[norm_raw])
            else:
                norm_name = raw.lower().replace(" ", "").replace(".", "")
                if norm_name in norm_to_skill:
                    resolved_skills.append(norm_to_skill[norm_name])
                else:
                    unresolved_strings.append(raw)

        if not unresolved_strings:
            # Deduplicate by ID
            return list({sk.id: sk for sk in resolved_skills}.values())

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
- nature: Must be exactly one of: "concept", "tech", "soft".
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
