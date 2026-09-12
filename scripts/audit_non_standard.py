"""Audit all non-standard skills against Lightcast catalog."""

import asyncio
import re
from sqlalchemy import text
from src.shared.database import AsyncSessionLocal


async def main() -> None:
    async with AsyncSessionLocal() as session:
        # 1. Fetch Lightcast skills
        res_lc = await session.execute(
            text("""
            SELECT s.skill_id, s.name, ss.standard_code
            FROM skills s
            JOIN skill_standards ss ON ss.skill_id = s.skill_id
            WHERE ss.standard_name = 'Lightcast';
        """)
        )
        lc_skills = res_lc.fetchall()
        lc_exact = {s[1].lower(): s for s in lc_skills}
        lc_tokens = {}
        for s in lc_skills:
            tokens = set(re.findall(r"[a-z0-9]+", s[1].lower()))
            norm = re.sub(r"[^a-z0-9]", "", s[1].lower())
            lc_tokens[s[0]] = (s[1], tokens, norm)

        # 2. Fetch all non-standard skills
        res_non = await session.execute(
            text("""
            SELECT s.skill_id, s.name, s.nature,
                   (SELECT COUNT(*) FROM offer_skills os WHERE os.skill_id = s.skill_id) as offers,
                   (SELECT COUNT(*) FROM profile_skills ps WHERE ps.skill_id = s.skill_id) as profiles,
                   (SELECT COUNT(*) FROM cluster_skills cs WHERE cs.skill_id = s.skill_id) as clusters
            FROM skills s
            LEFT JOIN skill_standards ss ON ss.skill_id = s.skill_id
            WHERE ss.skill_id IS NULL
            ORDER BY offers DESC, s.name ASC;
        """)
        )
        non_skills = res_non.fetchall()

        print(f"Total non-standard skills: {len(non_skills)}")

        exact_matches = []
        token_matches = []
        concatenated_junk = []
        no_matches = []

        for s_id, name, nature, offers, profiles, clusters in non_skills:
            clean = name.strip()
            lower = clean.lower()
            norm = re.sub(r"[^a-z0-9]", "", lower)

            # Check if concatenated Spanish or junk
            is_concatenated = False
            if (
                any(
                    w in lower
                    for w in [
                        "código",
                        "codigo",
                        "herramientas",
                        "manejo",
                        "ofimática",
                        "ofimatica",
                        "desarrollo",
                        "programación",
                    ]
                )
                and len(lower) > 15
            ):
                is_concatenated = True
            elif norm == lower and len(lower) > 12 and not any(c in lower for c in (" ", "-", ".")):
                # single long unspaced word like softwaredecódigoabierto or microsoftdynamics365customerservicecloud
                is_concatenated = True

            # 1. Exact match (case-insensitive)
            if lower in lc_exact:
                exact_matches.append((name, lc_exact[lower][1], offers, profiles, clusters))
                continue

            # 2. Token / base matching
            name_tokens = set(re.findall(r"[a-z0-9]+", lower))
            sig_tokens = {
                t
                for t in name_tokens
                if len(t) > 2
                and t not in ("software", "development", "programming", "language", "tools", "de")
            }

            candidates = []
            for lc_id, (lc_name, lc_toks, lc_norm) in lc_tokens.items():
                if sig_tokens and sig_tokens.issubset(lc_toks):
                    candidates.append(lc_name)
                elif norm and norm == lc_norm:
                    candidates.append(lc_name)
                elif norm and len(norm) >= 4 and norm in lc_norm:
                    candidates.append(lc_name)

            if is_concatenated:
                concatenated_junk.append((name, candidates[:3], offers, profiles, clusters))
            elif candidates:
                token_matches.append((name, candidates[:3], offers, profiles, clusters))
            else:
                no_matches.append((name, offers, profiles, clusters))

        print(f"\n--- 1. EXACT / DIRECT MATCHES ({len(exact_matches)}) ---")
        for orig, target, off, prof, clust in exact_matches:
            print(f"  '{orig}' -> '{target}' (offers={off}, prof={prof}, clust={clust})")

        print(f"\n--- 2. TOKEN / SUBSET MATCH CANDIDATES ({len(token_matches)}) ---")
        for orig, cands, off, prof, clust in token_matches:
            print(f"  '{orig}' (offers={off}, prof={prof}, clust={clust}) -> {cands}")

        print(f"\n--- 3. CONCATENATED / SPANISH JUNK ({len(concatenated_junk)}) ---")
        for orig, cands, off, prof, clust in concatenated_junk:
            print(f"  '{orig}' (offers={off}, prof={prof}, clust={clust}) -> {cands}")

        print(f"\n--- 4. NO DIRECT LIGHTCAST MATCH ({len(no_matches)}) ---")
        for orig, off, prof, clust in no_matches:
            print(f"  '{orig}' (offers={off}, prof={prof}, clust={clust})")


if __name__ == "__main__":
    asyncio.run(main())
