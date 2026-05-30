"""SQLAlchemy implementation of RoadmapRepository."""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.genai.domain.entities import Roadmap
from src.genai.domain.ports import RoadmapRepository
from src.genai.infrastructure.models import RoadmapModel
from src.ml_engine.infrastructure.models import DiagnosticModel, ProfileModel
from src.shared.exceptions import MLPipelineError


class SQLRoadmapRepository(RoadmapRepository):
    """SQLAlchemy implementation of RoadmapRepository using JSONB storage."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, roadmap: Roadmap) -> Roadmap:
        # Find profile for user_id to get diagnostic
        profile_stmt = select(ProfileModel).where(ProfileModel.user_id == roadmap.user_id)
        profile_result = await self._session.execute(profile_stmt)
        profile_model = profile_result.scalar_one_or_none()
        if not profile_model:
            raise MLPipelineError(f"No profile found for user_id {roadmap.user_id}")

        # Find latest diagnostic
        diag_stmt = (
            select(DiagnosticModel)
            .where(DiagnosticModel.profile_id == profile_model.profile_id)
            .order_by(DiagnosticModel.created_at.desc())
            .limit(1)
        )
        diag_result = await self._session.execute(diag_stmt)
        diag_model = diag_result.scalar_one_or_none()
        if not diag_model:
            raise MLPipelineError(
                f"No diagnostic found for profile of user_id {roadmap.user_id}"
            )

        # Build JSON serialization
        roadmap_json: dict[str, Any] = {
            "user_id": str(roadmap.user_id),
            "specialty": roadmap.specialty,
            "seniority": roadmap.seniority,
            "generated_by_model": roadmap.generated_by_model,
            "phases": [
                {
                    "phase_number": p.phase_number,
                    "title": p.title,
                    "description": p.description,
                    "skills_to_acquire": p.skills_to_acquire,
                    "complexity": str(p.complexity),
                    "estimated_weeks": p.estimated_weeks,
                    "sfia_reference": p.sfia_reference,
                    "swecom_reference": p.swecom_reference,
                    "resources": [
                        {
                            "title": r.title,
                            "resource_type": r.resource_type,
                            "url": r.url,
                            "estimated_hours": r.estimated_hours,
                        }
                        for r in p.resources
                    ]
                }
                for p in roadmap.phases
            ]
        }

        # Check if roadmap model already exists
        result = await self._session.execute(
            select(RoadmapModel).where(RoadmapModel.roadmap_id == roadmap.id)
        )
        model = result.scalar_one_or_none()

        if not model:
            model = RoadmapModel(
                roadmap_id=roadmap.id,
                diagnostic_id=diag_model.diagnostic_id,
                roadmap_json=roadmap_json,
                status="completed",
            )
            self._session.add(model)
        else:
            model.roadmap_json = roadmap_json
            model.status = "completed"

        await self._session.flush()
        return roadmap

    async def get_by_user_id(self, user_id: UUID) -> list[Roadmap]:
        # Find profile for user_id to get diagnostics
        profile_stmt = select(ProfileModel).where(ProfileModel.user_id == user_id)
        profile_result = await self._session.execute(profile_stmt)
        profile_model = profile_result.scalar_one_or_none()
        if not profile_model:
            return []

        # Find diagnostics for profile
        diag_stmt = select(DiagnosticModel.diagnostic_id).where(
            DiagnosticModel.profile_id == profile_model.profile_id
        )
        diag_result = await self._session.execute(diag_stmt)
        diag_ids = [row[0] for row in diag_result.all()]

        if not diag_ids:
            return []

        # Find roadmaps linked to those diagnostics
        stmt = (
            select(RoadmapModel)
            .where(RoadmapModel.diagnostic_id.in_(diag_ids))
            .order_by(RoadmapModel.created_at.desc())
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()

        return [m.to_entity() for m in models]

    async def get_latest_by_user_id(self, user_id: UUID) -> Roadmap | None:
        # Find profile for user_id
        profile_stmt = select(ProfileModel).where(ProfileModel.user_id == user_id)
        profile_result = await self._session.execute(profile_stmt)
        profile_model = profile_result.scalar_one_or_none()
        if not profile_model:
            return None

        # Find diagnostics for profile
        diag_stmt = select(DiagnosticModel.diagnostic_id).where(
            DiagnosticModel.profile_id == profile_model.profile_id
        )
        diag_result = await self._session.execute(diag_stmt)
        diag_ids = [row[0] for row in diag_result.all()]

        if not diag_ids:
            return None

        # Find latest roadmap linked to those diagnostics
        stmt = (
            select(RoadmapModel)
            .where(RoadmapModel.diagnostic_id.in_(diag_ids))
            .order_by(RoadmapModel.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()

        return model.to_entity() if model else None
