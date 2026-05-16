"""GenAI use cases — RAG-powered roadmap generation."""

import json
from uuid import uuid4

import structlog

from src.genai.application.dtos import RoadmapDTO, RoadmapPhaseDTO, RoadmapRequestDTO
from src.genai.domain.entities import PhaseComplexity, Roadmap, RoadmapPhase
from src.genai.domain.ports import LLMService, RoadmapRepository, VectorStorePort
from src.shared.exceptions import RAGPipelineError

logger = structlog.get_logger(__name__)


class GenerateRoadmapUseCase:
    """
    RAG-powered roadmap generation.

    Flow:
    1. Query pgvector for relevant SFIA9/SWECOM context
       (filtered by specialty + seniority)
    2. Build structured prompt with context + user gaps
    3. Call LLM (Groq/OpenAI via LangChain)
    4. Parse and validate LLM output
    5. Persist and return roadmap
    """

    def __init__(
        self,
        llm_service: LLMService,
        vector_store: VectorStorePort,
        roadmap_repository: RoadmapRepository,
    ) -> None:
        self._llm = llm_service
        self._vector_store = vector_store
        self._roadmaps = roadmap_repository

    async def execute(self, request: RoadmapRequestDTO) -> RoadmapDTO:
        try:
            logger.info(
                "Generating roadmap",
                user_id=str(request.user_id),
                specialty=request.specialty,
                seniority=request.seniority,
            )

            # Step 1: Retrieve relevant SFIA9/SWECOM context
            query = f"{request.specialty} {request.seniority} " + " ".join(request.skill_gaps[:5])
            context_docs = await self._vector_store.similarity_search(
                query=query,
                k=6,
                filter_metadata={"seniority": request.seniority},
            )

            # Step 2: Build prompt
            prompt = _build_roadmap_prompt(request, context_docs)

            # Step 3: Call LLM
            raw_output = await self._llm.generate(prompt=prompt, context=context_docs)

            # Step 4: Parse output (expect JSON from LLM)
            phases = _parse_llm_phases(raw_output, request.seniority)

            # Step 5: Build and persist
            roadmap = Roadmap(
                id=uuid4(),
                user_id=request.user_id,
                specialty=request.specialty,
                seniority=request.seniority,
                phases=phases,
                total_estimated_weeks=Roadmap.compute_total_weeks(phases),
                generated_by_model=self._llm.__class__.__name__,
            )
            saved = await self._roadmaps.save(roadmap)

            logger.info(
                "Roadmap generated",
                roadmap_id=str(saved.id),
                phases=len(phases),
                total_weeks=saved.total_estimated_weeks,
            )

            return _to_dto(saved)

        except RAGPipelineError:
            raise
        except Exception as exc:
            logger.exception("Roadmap generation failed", error=str(exc))
            raise RAGPipelineError("Failed to generate roadmap") from exc


def _build_roadmap_prompt(request: RoadmapRequestDTO, context: list[str]) -> str:
    """Build the structured RAG prompt."""
    context_text = (
        "\n---\n".join(context) if context else "No specific standards context available."
    )
    gaps_text = ", ".join(request.skill_gaps) if request.skill_gaps else "No gaps identified."

    return f"""You are a technical career advisor specializing in software engineering competencies.

Based on the following industry standards context:
{context_text}

Generate a personalized learning roadmap for a {request.seniority}-level developer
specializing in {request.specialty}.

Skill gaps to address: {gaps_text}

Requirements:
- Generate 3-5 learning phases
- Each phase must map to SFIA 9 competencies where possible
- Phases must be progressive (foundational → advanced)
- Be specific and technical, not generic

Respond ONLY with a valid JSON array of phases in this exact format:
[
  {{
    "phase_number": 1,
    "title": "Phase title",
    "description": "What the developer will learn",
    "skills_to_acquire": ["skill1", "skill2"],
    "complexity": "foundational|intermediate|advanced|expert",
    "estimated_weeks": 4,
    "sfia_reference": "PROG" or null,
    "swecom_reference": "Software Construction" or null
  }}
]"""


def _parse_llm_phases(raw_output: str, seniority: str) -> list[RoadmapPhase]:
    """Parse LLM JSON output into domain entities."""
    try:
        # Extract JSON array from output (LLM may include text before/after)
        start = raw_output.find("[")
        end = raw_output.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in LLM output")

        data = json.loads(raw_output[start:end])

        phases = []
        for item in data:
            complexity_map = {
                "foundational": PhaseComplexity.FOUNDATIONAL,
                "intermediate": PhaseComplexity.INTERMEDIATE,
                "advanced": PhaseComplexity.ADVANCED,
                "expert": PhaseComplexity.EXPERT,
            }
            complexity = complexity_map.get(
                item.get("complexity", "foundational"), PhaseComplexity.FOUNDATIONAL
            )
            phases.append(
                RoadmapPhase(
                    phase_number=item["phase_number"],
                    title=item["title"],
                    description=item["description"],
                    skills_to_acquire=item.get("skills_to_acquire", []),
                    complexity=complexity,
                    estimated_weeks=item.get("estimated_weeks", 4),
                    sfia_reference=item.get("sfia_reference"),
                    swecom_reference=item.get("swecom_reference"),
                )
            )
        return sorted(phases, key=lambda p: p.phase_number)

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("LLM output parsing failed", error=str(exc), raw=raw_output[:500])
        raise RAGPipelineError("LLM returned invalid roadmap format") from exc


def _to_dto(roadmap: Roadmap) -> RoadmapDTO:
    """Convert domain entity to DTO."""
    return RoadmapDTO(
        id=roadmap.id,
        user_id=roadmap.user_id,
        specialty=roadmap.specialty,
        seniority=roadmap.seniority,
        phases=[
            RoadmapPhaseDTO(
                phase_number=p.phase_number,
                title=p.title,
                description=p.description,
                skills_to_acquire=p.skills_to_acquire,
                complexity=p.complexity.value,
                estimated_weeks=p.estimated_weeks,
                sfia_reference=p.sfia_reference,
                swecom_reference=p.swecom_reference,
            )
            for p in roadmap.phases
        ],
        total_estimated_weeks=roadmap.total_estimated_weeks,
        generated_by_model=roadmap.generated_by_model,
    )
