"""GenAI FastAPI router."""

from fastapi import APIRouter

from src.dependencies import SessionDep
from src.genai.application.dtos import RoadmapDTO, RoadmapRequestDTO
from src.genai.application.use_cases import GenerateRoadmapUseCase
from src.genai.infrastructure.langchain_chain import get_llm_service
from src.genai.infrastructure.roadmap_repository import SQLRoadmapRepository
from src.genai.infrastructure.vector_store import PGVectorStore
from src.shared.security import CurrentUserIdDep

router = APIRouter(prefix="/roadmap", tags=["GenAI — Roadmap Generation"])


@router.post(
    "/generate",
    response_model=RoadmapDTO,
    status_code=202,
    summary="Generate personalized learning roadmap",
)
async def generate_roadmap(
    request: RoadmapRequestDTO,
    current_user_id: CurrentUserIdDep,
    session: SessionDep,
) -> RoadmapDTO:
    """
    Generate a personalized learning roadmap using RAG + LLM.

    This endpoint:
    1. Retrieves relevant SFIA 9 / IEEE SWECOM context from pgvector
    2. Injects context into a structured LLM prompt
    3. Generates a phased roadmap validated against industry standards
    4. Returns a structured JSON roadmap

    The roadmap complexity adapts to the user's seniority level.
    """
    use_case = GenerateRoadmapUseCase(
        llm_service=get_llm_service(),
        vector_store=PGVectorStore(),
        roadmap_repository=SQLRoadmapRepository(session),
    )
    return await use_case.execute(request)
