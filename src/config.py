"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # === Application ===
    APP_ENV: Literal["development", "staging", "production", "test"] = "development"
    DEBUG: bool = False
    VERSION: str = "0.1.0"
    PROJECT_NAME: str = "Devalign API"
    API_V1_PREFIX: str = "/api/v1"

    # === CORS ===
    CORS_ORIGINS: str | list[str] = Field(
        default=["http://localhost:3000", "https://devalign.vercel.app"]
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # === Database ===
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/devalign_dev",
        description="Async PostgreSQL connection string (Supabase or local pgvector)",
    )

    # === Supabase ===
    SUPABASE_URL: str = Field(default="", description="Supabase project URL")
    SUPABASE_ANON_KEY: str = Field(default="", description="Supabase anon/public key")
    SUPABASE_SERVICE_ROLE_KEY: str = Field(
        default="", description="Supabase service role key (server-side only)"
    )
    SUPABASE_JWT_SECRET: str = Field(default="", description="Supabase JWT secret key")

    # === LLM Configuration ===
    LLM_PROVIDER: Literal["groq", "openai"] = "groq"
    LLM_MODEL: str = "llama-3.1-70b-versatile"
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    VOYAGE_API_KEY: str = Field(default="", description="Voyage AI API key")

    # === Embeddings Configuration ===
    EMBEDDING_PROVIDER: Literal["openai", "voyage"] = "voyage"
    EMBEDDING_MODEL: str = "voyage-4-lite"

    # === Security ===
    SECRET_KEY: str = Field(
        default="change-me-in-production",
        description="JWT signing key — MUST be changed in production",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — use as FastAPI dependency."""
    return Settings()


# Module-level singleton for non-DI usage
settings = get_settings()
