"""Native LLM client using direct HTTP requests to Groq or OpenAI."""

from typing import Any

import httpx
import structlog

from src.config import settings
from src.ml_engine.domain.ports import LLMService

logger = structlog.get_logger(__name__)


class NativeLLMClient(LLMService):
    """Lightweight, native LLM client using httpx.

    Supports Groq and OpenAI chat completion endpoints.
    """

    def __init__(self) -> None:
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL

        if self.provider == "groq":
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            self.api_key = settings.GROQ_API_KEY
        else:
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.api_key = settings.OPENAI_API_KEY

    async def generate(self, prompt: str, context: list[Any] | None = None) -> str:
        """Call the provider API to generate response content."""
        if not self.api_key:
            raise ValueError(f"API key for {self.provider} LLM provider is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build messages payload.
        # Optional context can be formatted as previous messages.
        messages: list[dict[str, str]] = []
        if context:
            for msg in context:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
                else:
                    messages.append({"role": "user", "content": str(msg)})

        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }

        # Request structured JSON format if instructions indicate JSON
        if "json" in prompt.lower():
            payload["response_format"] = {"type": "json_object"}

        logger.info(
            "Sending LLM chat completion request",
            provider=self.provider,
            model=self.model,
            messages_count=len(messages),
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return str(content)
            except httpx.HTTPStatusError as e:
                logger.error(
                    "LLM request failed with HTTP status error",
                    status_code=e.response.status_code,
                    response_body=e.response.text,
                )
                raise
            except Exception as e:
                logger.error("LLM request failed with unexpected error", error=str(e))
                raise


def get_llm_service() -> LLMService:
    """Factory dependency for LLMService."""
    return NativeLLMClient()
