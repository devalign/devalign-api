"""Native LLM client using direct HTTP requests to Groq or OpenAI."""

import asyncio
from typing import Any

import httpx
import structlog

from src.config import settings
from src.ml_engine.domain.ports import LLMService
from src.shared.exceptions import RateLimitError

logger = structlog.get_logger(__name__)


class NativeLLMClient(LLMService):
    """Lightweight, native LLM client using httpx.

    Supports Groq and OpenAI chat completion endpoints.
    """

    def __init__(self) -> None:
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0))

        if self.provider == "groq":
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            self.api_key = settings.GROQ_API_KEY
        else:
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.api_key = settings.OPENAI_API_KEY

    async def generate(self, prompt: str, context: list[Any] | None = None, max_tokens: int | None = None) -> str:
        """Call the provider API to generate response content with retry on 5xx only."""
        if not self.api_key:
            raise ValueError(f"API key for {self.provider} LLM provider is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if self.provider == "openai":
            payload["store"] = True

        if "json" in prompt.lower():
            payload["response_format"] = {"type": "json_object"}

        logger.info(
            "Sending LLM chat completion request",
            provider=self.provider,
            model=self.model,
            messages_count=len(messages),
            max_tokens=max_tokens,
        )

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                response = await self._client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return str(content)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body = e.response.text
                last_exc = e

                if status == 429:
                    logger.warning(
                        "LLM rate limit exceeded",
                        provider=self.provider,
                        response_body=body,
                    )
                    raise RateLimitError(
                        f"Rate limit exceeded for {self.provider}. "
                        f"Response: {body}"
                    ) from e

                if status >= 500 and attempt == 0:
                    logger.warning(
                        "LLM server error, retrying once",
                        status_code=status,
                        provider=self.provider,
                    )
                    await asyncio.sleep(1)
                    continue

                logger.error(
                    "LLM request failed with HTTP status error",
                    status_code=status,
                    provider=self.provider,
                    response_body=body,
                )
                raise
            except Exception as e:
                logger.error("LLM request failed with unexpected error", error=str(e), exc_info=True)
                raise

        # Only reached if all retries exhausted
        raise httpx.HTTPStatusError(
            f"LLM request failed after retries. Last error: {last_exc}",
            request=last_exc.request,  # type: ignore[union-attr]
            response=last_exc.response,  # type: ignore[union-attr]
        )


def get_llm_service() -> LLMService:
    """Factory dependency for LLMService."""
    return NativeLLMClient()
