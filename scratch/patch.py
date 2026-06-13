import os
import re

USE_CASES_PATH = "src/ml_engine/application/use_cases.py"
with open(USE_CASES_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Insert the helper function at the bottom
helper_func = """

def compute_affinities_and_domains(detected_skills, active_clusters):
    from src.ml_engine.domain.entities import SkillType, ClusterAffinity
    from src.ml_engine.application.dtos import DomainAffinityDTO

    user_hard_skills = [
        s for s in detected_skills
        if s.skill_type in (SkillType.HARD_SKILL, SkillType.TOOL)
    ]
    user_hard_norms = {s.normalized_name: s for s in user_hard_skills}

    affinities = []
    for cluster in active_clusters:
        cluster_hard_skills = [
            s for s in cluster.centroid_skills
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
    \"\"\"Gets the logged-in user's profile and computes real-time affinities against active clusters.\"\"\"
    def __init__(self, profile_repository, cluster_repository):
        self._profiles = profile_repository
        self._clusters = cluster_repository

    async def execute(self, user_id) -> "UserProfileDTO | None":
        from src.ml_engine.application.dtos import UserProfileDTO, ClusterAffinityDTO, SkillDTO
        
        profile = await self._profiles.get_by_user_id(user_id)
        if not profile:
            return None

        active_clusters = await self._clusters.get_all_active()
        active_clusters = [c for c in active_clusters if c.centroid_skills]

        primary, secondaries, all_affinities, domain_affinities_dto = compute_affinities_and_domains(
            profile.detected_skills, active_clusters
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
                SkillDTO(name=s.name, skill_type=s.skill_type.value, market_importance="consolidated", market_demand_percentage=round(s.frequency * 100) if s.frequency is not None else 100)
                for s in profile.detected_skills
            ],
            skill_gaps=[
                SkillDTO(name=g.skill.name, skill_type=g.skill.skill_type.value, market_importance=g.market_importance, market_demand_percentage=round(g.skill.frequency * 100) if g.skill.frequency is not None else None)
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
            message="Profile retrieved successfully"
        )
"""
if "GetMyProfileUseCase" not in content:
    content += helper_func

# Now replace the inline calculation in ProfileUserFromCVUseCase.execute
old_calc_block = '''            # Step 6: Compute Weighted Jaccard Similarity per cluster
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
                        market_insights=cluster.market_insights,
                        compatible_roles=cluster.compatible_roles,
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
                market_insights=primary.market_insights,
                compatible_roles=primary.compatible_roles,
            )
            secondaries = affinities[1:3]  # Top 2 secondary affinities'''

new_calc_block = '''            # Step 6: Compute Weighted Jaccard Similarity per cluster
            primary, secondaries, affinities, _ = compute_affinities_and_domains(detected_skills, active_clusters)'''

content = content.replace(old_calc_block, new_calc_block)

old_domain_block = '''            # Compute Domain Affinities
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
            domain_affinities_dto.sort(key=lambda x: x.affinity_score, reverse=True)'''

new_domain_block = '''            # Compute Domain Affinities
            _, _, _, domain_affinities_dto = compute_affinities_and_domains(detected_skills, active_clusters)'''

content = content.replace(old_domain_block, new_domain_block)

with open(USE_CASES_PATH, "w", encoding="utf-8") as f:
    f.write(content)

# Update router.py
ROUTER_PATH = "src/ml_engine/interface/router.py"
with open(ROUTER_PATH, "r", encoding="utf-8") as f:
    router_content = f.read()

# Replace get_my_profile in router.py
old_get_my_profile = '''@router.get(
    "/me",
    response_model=UserProfileDTO,
    summary="Get logged-in user's computed profile",
)
async def get_my_profile(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> UserProfileDTO:
    """
    Get the computed profile of the authenticated developer.
    Returns HTTP 404 if no CV has been analyzed yet.
    """
    from uuid import UUID

    from fastapi import HTTPException

    repo = SQLUserProfileRepository(session)
    profile = await repo.get_by_user_id(UUID(current_user_id))
    if not profile:
        raise HTTPException(status_code=404, detail="No profile found. Please upload a CV first.")

    return _map_entity_to_dto(profile)'''

new_get_my_profile = '''@router.get(
    "/me",
    response_model=UserProfileDTO,
    summary="Get logged-in user's computed profile",
)
async def get_my_profile(
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> UserProfileDTO:
    """
    Get the computed profile of the authenticated developer.
    Returns HTTP 404 if no CV has been analyzed yet.
    """
    from uuid import UUID
    from fastapi import HTTPException
    from src.ml_engine.application.use_cases import GetMyProfileUseCase
    from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository

    repo = SQLUserProfileRepository(session)
    cluster_repo = SQLClusterRepository(session)
    use_case = GetMyProfileUseCase(repo, cluster_repo)
    
    dto = await use_case.execute(UUID(current_user_id))
    if not dto:
        raise HTTPException(status_code=404, detail="No profile found. Please upload a CV first.")

    return dto'''

router_content = router_content.replace(old_get_my_profile, new_get_my_profile)

# Also update update_my_profile and update_my_skills to use _map_entity_to_dto, but since they modify fields, they don't return the full affinities right now unless we call GetMyProfileUseCase.execute again!
old_update_my_profile = '''    updated_profile = replace(profile, **kwargs)
    await repo.save(updated_profile)

    return _map_entity_to_dto(updated_profile)'''
new_update_my_profile = '''    updated_profile = replace(profile, **kwargs)
    await repo.save(updated_profile)

    from src.ml_engine.application.use_cases import GetMyProfileUseCase
    from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
    dto = await GetMyProfileUseCase(repo, SQLClusterRepository(session)).execute(UUID(current_user_id))
    return dto'''
router_content = router_content.replace(old_update_my_profile, new_update_my_profile)

old_update_my_skills = '''    updated_profile = replace(profile, detected_skills=detected_skills, skill_gaps=skill_gaps)
    await repo.save(updated_profile)

    return _map_entity_to_dto(updated_profile)'''
new_update_my_skills = '''    updated_profile = replace(profile, detected_skills=detected_skills, skill_gaps=skill_gaps)
    await repo.save(updated_profile)

    from src.ml_engine.application.use_cases import GetMyProfileUseCase
    from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
    dto = await GetMyProfileUseCase(repo, SQLClusterRepository(session)).execute(UUID(current_user_id))
    return dto'''
router_content = router_content.replace(old_update_my_skills, new_update_my_skills)

with open(ROUTER_PATH, "w", encoding="utf-8") as f:
    f.write(router_content)

print("Patch applied successfully.")
