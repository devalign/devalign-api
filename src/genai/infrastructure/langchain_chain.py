"""LLM service implementations.

Supports:
- GroqLLMService  → Groq API (llama-3.1-70b-versatile) — default for development
- OpenAILLMService → OpenAI GPT-4o — for production quality

Both implement LLMService port — swappable via config.
"""

import structlog

from src.config import settings
from src.genai.domain.ports import LLMService
from src.shared.exceptions import ExternalServiceError, RAGPipelineError

logger = structlog.get_logger(__name__)


class GroqLLMService(LLMService):
    """LLM service using Groq API via LangChain."""

    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile") -> None:
        self._api_key = api_key
        self._model = model

    async def generate(self, prompt: str, context: list[str]) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                api_key=self._api_key,
                model_name=self._model,
                temperature=0.3,  # Low temperature for structured output
                max_tokens=4096,
            )

            messages = [
                SystemMessage(content="You are a technical career advisor for software engineers."),
                HumanMessage(content=prompt),
            ]

            response = await llm.ainvoke(messages)
            content = response.content

            if not isinstance(content, str):
                raise RAGPipelineError("LLM returned non-string content")

            logger.debug("Groq LLM response", model=self._model, chars=len(content))
            return content

        except (RAGPipelineError, ExternalServiceError):
            raise
        except Exception as exc:
            logger.error("Groq LLM call failed", error=str(exc))
            raise ExternalServiceError(f"Groq LLM service error: {exc}") from exc


class OpenAILLMService(LLMService):
    """LLM service using OpenAI API via LangChain."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model = model

    async def generate(self, prompt: str, context: list[str]) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=self._api_key,
                model=self._model,
                temperature=0.3,
            )

            messages = [
                SystemMessage(content="You are a technical career advisor for software engineers."),
                HumanMessage(content=prompt),
            ]

            response = await llm.ainvoke(messages)
            content = response.content

            if not isinstance(content, str):
                raise RAGPipelineError("LLM returned non-string content")

            return content

        except (RAGPipelineError, ExternalServiceError):
            raise
        except Exception as exc:
            logger.error("OpenAI LLM call failed", error=str(exc))
            raise ExternalServiceError(f"OpenAI LLM service error: {exc}") from exc


def get_llm_service() -> LLMService:
    """Factory: returns the configured LLM service."""
    if settings.LLM_PROVIDER == "openai":
        return OpenAILLMService(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
        )
    # Default: Groq
    return GroqLLMService(
        api_key=settings.GROQ_API_KEY,
        model=settings.LLM_MODEL,
    )
