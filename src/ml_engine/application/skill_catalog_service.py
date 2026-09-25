import difflib
import json
import re
from typing import Any
from uuid import uuid4

import structlog

from src.ml_engine.domain.entities import Skill, SkillNature
from src.ml_engine.domain.ports import EmbeddingService, LLMService, SkillRepository

logger = structlog.get_logger(__name__)

SINGLE_SLASH_TERMS: set[str] = {"ci/cd", "tcp/ip", "i/o", "pl/sql", "client/server"}

# Abstract meta-skills/categories valid for cluster centroids (market demand)
# but prohibited on individual user profiles to prevent artificial affinity distortions.
PROFILE_SKILL_BLACKLIST: frozenset[str] = frozenset(
    {
        "software configuration management",
        "software engineering",
        "full stack development",
        "front end (software engineering)",
        "back end (software engineering)",
        "devops",
    }
)


def build_skill_lookup_indexes(
    skills: list[Skill],
) -> tuple[dict[str, Skill], dict[str, Skill]]:
    """Build fast lookup dictionaries (alias_to_skill, norm_to_skill) for skill resolution.

    Indexes:
    - Canonical normalized_name (e.g. 'python(programminglanguage)')
    - Normalized name without punctuation
    - Base name without parentheses (e.g. 'Python (Programming Language)' -> 'python')
    - Full canonical name lowercase
    - All declared aliases (both raw lowercase and punctuation-free)
    """
    alias_to_skill: dict[str, Skill] = {}
    norm_to_skill: dict[str, Skill] = {}

    def _index_one(skill: Skill) -> None:
        # 1. Canonical normalized_name
        norm = skill.normalized_name.lower().strip()
        norm_to_skill[norm] = skill
        norm_clean = re.sub(r"[^a-z0-9]", "", norm)
        if norm_clean:
            norm_to_skill[norm_clean] = skill

        # 2. Canonical display name lowercase
        name_lower = skill.name.strip().lower()
        if name_lower:
            alias_to_skill[name_lower] = skill

        # 3. Base name without parentheses: e.g. "Python (Programming Language)" -> "python"
        base_name = re.sub(r"\s*\(.*?\)\s*", "", skill.name).strip().lower()
        if base_name:
            alias_to_skill[base_name] = skill
            base_clean = re.sub(r"[^a-z0-9]", "", base_name)
            if base_clean:
                norm_to_skill[base_clean] = skill

        # 4. Declared aliases
        for alias in skill.aliases:
            alias_clean = alias.strip().lower()
            if alias_clean:
                alias_to_skill[alias_clean] = skill
                alias_norm = re.sub(r"[^a-z0-9]", "", alias_clean)
                if alias_norm:
                    norm_to_skill[alias_norm] = skill

    for s in skills:
        _index_one(s)

    return alias_to_skill, norm_to_skill


def match_single_skill(
    raw: str,
    alias_to_skill: dict[str, Skill],
    norm_to_skill: dict[str, Skill],
) -> Skill | None:
    """Matches a raw skill string to a canonical Skill entity using multiple strategies."""
    clean = raw.strip()
    if not clean:
        return None

    lower = clean.lower()

    # Reject obvious hallucinations, empty states, or sentences mistakenly parsed as skills
    hallucination_phrases = (
        "no mencionado",
        "not mentioned",
        "no especificado",
        "sin especificar",
        "none specified",
        "n/a",
    )
    if any(phrase in lower for phrase in hallucination_phrases) or len(clean) > 80:
        return None

    # 1. Exact alias lookup
    if lower in alias_to_skill:
        return alias_to_skill[lower]

    # 2. Normalized name lookup (spaces/dots/hyphens stripped)
    norm = re.sub(r"[^a-z0-9]", "", lower)
    if norm and norm in norm_to_skill:
        return norm_to_skill[norm]

    # 3. Check inside parentheses: e.g. "NoSQL (MongoDB)" -> "MongoDB"
    paren_match = re.search(r"\((.*?)\)", clean)
    if paren_match:
        inner = paren_match.group(1).strip()
        inner_lower = inner.lower()
        if inner_lower in alias_to_skill:
            return alias_to_skill[inner_lower]
        inner_norm = re.sub(r"[^a-z0-9]", "", inner_lower)
        if inner_norm and inner_norm in norm_to_skill:
            return norm_to_skill[inner_norm]

        # Also check outside parentheses: e.g. "MongoDB (NoSQL)" -> "MongoDB"
        outer = re.sub(r"\(.*?\)", "", clean).strip()
        outer_lower = outer.lower()
        if outer_lower in alias_to_skill:
            return alias_to_skill[outer_lower]
        outer_norm = re.sub(r"[^a-z0-9]", "", outer_lower)
        if outer_norm and outer_norm in norm_to_skill:
            return norm_to_skill[outer_norm]

    # 4. Check stripping common conversational prefixes
    for prefix in (
        "framework ",
        "librería ",
        "libreria ",
        "library ",
        "lenguaje ",
        "language ",
        "herramienta ",
        "tool ",
    ):
        if lower.startswith(prefix):
            stripped = clean[len(prefix) :].strip()
            res = match_single_skill(stripped, alias_to_skill, norm_to_skill)
            if res:
                return res

    return None


_match_single_skill = match_single_skill


class SkillCatalogService:
    def __init__(
        self,
        skill_repository: SkillRepository,
        llm_service: LLMService,
        embedding_service: EmbeddingService | None = None,
    ):
        self._skills = skill_repository
        self._llm = llm_service
        self._embedding = embedding_service

    async def normalize_extracted_skills(
        self,
        raw_skills: list[dict[str, Any]],
        existing_skills_cache: list[Skill] | None = None,
    ) -> list[dict[str, Any]]:
        """Pre-normalizes LLM-extracted skill dicts against canonical Lightcast catalog in Phase 1.

        Updates 'name' to the canonical skill name if matched, retaining 'original_raw_name'.
        Marks 'is_custom': False, 'in_catalog': True for catalog skills,
        and 'is_custom': True, 'in_catalog': False for unrecognized skills.
        If fuzzy similarity is high (0.78-0.88), provides 'suggested_canonical' for UI chips.
        Deduplicates items by canonical name while merging evidence.
        """
        if not raw_skills:
            return []

        if existing_skills_cache is not None:
            existing_skills = existing_skills_cache
        else:
            existing_skills = await self._skills.get_all_skills()

        alias_to_skill, norm_to_skill = build_skill_lookup_indexes(existing_skills)
        all_alias_keys = list(alias_to_skill.keys())

        expanded_raw: list[dict[str, Any]] = []
        for item in raw_skills:
            if not isinstance(item, dict):
                continue
            r_name = str(item.get("name", "")).strip()
            if "/" in r_name and r_name.lower() not in SINGLE_SLASH_TERMS:
                parts = [p.strip() for p in r_name.split("/") if p.strip()]
                for p in parts:
                    clone = dict(item)
                    clone["name"] = p
                    expanded_raw.append(clone)
            else:
                expanded_raw.append(item)

        dedup_map: dict[str, dict[str, Any]] = {}
        for item in expanded_raw:
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
                processed_item["suggested_canonical"] = None
                processed_item["category"] = (
                    "concept" if matched.nature == SkillNature.CONCEPT else "technical"
                )
            else:
                # Fuzzy fallback matching against canonical catalog
                clean_lower = raw_name.lower()
                close_matches = difflib.get_close_matches(
                    clean_lower, all_alias_keys, n=1, cutoff=0.78
                )

                if close_matches:
                    best_match_key = close_matches[0]
                    matched_candidate = alias_to_skill[best_match_key]
                    ratio = difflib.SequenceMatcher(None, clean_lower, best_match_key).ratio()

                    if ratio >= 0.88:
                        canonical_name = matched_candidate.name
                        norm_key = canonical_name.lower()
                        processed_item["name"] = canonical_name
                        processed_item["original_raw_name"] = raw_name
                        processed_item["in_catalog"] = True
                        processed_item["is_custom"] = False
                        processed_item["suggested_canonical"] = None
                        processed_item["category"] = (
                            "concept"
                            if matched_candidate.nature == SkillNature.CONCEPT
                            else "technical"
                        )
                    else:
                        norm_key = raw_name.lower()
                        processed_item["name"] = raw_name
                        processed_item["in_catalog"] = False
                        processed_item["is_custom"] = True
                        processed_item["suggested_canonical"] = matched_candidate.name
                else:
                    norm_key = raw_name.lower()
                    processed_item["name"] = raw_name
                    processed_item["in_catalog"] = False
                    processed_item["is_custom"] = True
                    processed_item["suggested_canonical"] = None

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

        # Exclude blacklisted abstract meta-skills from Phase 1 UI output
        filtered_items = [
            item
            for item in dedup_map.values()
            if str(item.get("name", "")).strip().lower() not in PROFILE_SKILL_BLACKLIST
        ]
        return filtered_items

    async def scan_text_for_catalog_skills(
        self,
        text: str,
        existing_skills_cache: list[Skill] | None = None,
    ) -> list[dict[str, Any]]:
        """Fast deterministic scanner that searches for canonical skill names and aliases in raw CV text.

        Ensures explicit technical keywords mentioned in the CV are never dropped by non-deterministic LLM omissions.
        """
        if not text or not text.strip():
            return []

        if existing_skills_cache is not None:
            existing_skills = existing_skills_cache
        else:
            existing_skills = await self._skills.get_all_skills()

        text_lower = text.lower()
        matched_skills: dict[str, Skill] = {}

        for skill in existing_skills:
            # Check base name without parentheses (e.g. "React" from "React (JavaScript Framework)")
            base_name = re.sub(r"\s*\(.*?\)\s*", "", skill.name).strip()
            if not base_name or len(base_name) < 2:
                continue

            base_lower = base_name.lower()
            if base_lower in PROFILE_SKILL_BLACKLIST:
                continue

            terms_to_search = [base_name] + [a for a in skill.aliases if a and len(a) >= 2]

            found = False
            for term in terms_to_search:
                term_clean = term.strip()
                term_lower = term_clean.lower()
                if term_lower in PROFILE_SKILL_BLACKLIST:
                    continue

                if len(term_clean) <= 2:
                    # For short acronyms / 2-char languages (e.g. C#, GO, JS, TS, R), use case-sensitive word boundary
                    pattern = r"(?<!\w)" + re.escape(term_clean) + r"(?!\w)"
                    if re.search(pattern, text):
                        found = True
                        break
                elif len(term_clean) >= 3:
                    # Case-insensitive word boundary match
                    pattern = r"(?<!\w)" + re.escape(term_lower) + r"(?!\w)"
                    if re.search(pattern, text_lower):
                        found = True
                        break

            if found:
                matched_skills[skill.name.lower()] = skill

        results: list[dict[str, Any]] = []
        for sk in matched_skills.values():
            results.append(
                {
                    "name": sk.name,
                    "category": "concept" if sk.nature == SkillNature.CONCEPT else "technical",
                    "years_of_experience": 1,
                    "self_taught": False,
                    "personal_projects": False,
                    "has_certification": False,
                    "in_catalog": True,
                    "is_custom": False,
                }
            )

        logger.info("Deterministic catalog scan found skills in text", count=len(results))
        return results

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
        # Clean inputs and split compound terms like "JavaScript/TypeScript"
        clean_strings = []
        for raw_str in raw_strings:
            if isinstance(raw_str, str) and raw_str.strip():
                trimmed = raw_str.strip()
                if "/" in trimmed and trimmed.lower() not in SINGLE_SLASH_TERMS:
                    parts = [p.strip() for p in trimmed.split("/") if p.strip()]
                    clean_strings.extend(parts)
                else:
                    clean_strings.append(trimmed)

        if not clean_strings:
            return []

        # 1. Load existing skills (use cache if provided to avoid double load)
        if existing_skills_cache is not None:
            existing_skills = existing_skills_cache
        else:
            existing_skills = await self._skills.get_all_skills()

        # Build lookup maps
        alias_to_skill, norm_to_skill = build_skill_lookup_indexes(existing_skills)

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
