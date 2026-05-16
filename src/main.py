"""Devalign API — Application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings

# Module routers
from src.delivery.interface.router import router as delivery_router
from src.genai.interface.router import router as genai_router
from src.ml_engine.interface.router import router as ml_router
from src.scraper.interface.router import router as scraper_router
from src.shared.database import engine
from src.shared.logging import configure_logging
from src.shared.middleware import RequestLoggingMiddleware

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events."""
    # Startup
    configure_logging()
    logger.info(
        "Starting Devalign API",
        env=settings.APP_ENV,
        version=settings.VERSION,
    )
    yield
    # Shutdown
    await engine.dispose()
    logger.info("Devalign API shutdown complete")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "ML-powered API for tech skills gap analysis "
            "and personalized learning roadmap generation."
        ),
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        lifespan=lifespan,
    )

    # === Middleware ===
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    # === Routers ===
    app.include_router(delivery_router, prefix=settings.API_V1_PREFIX)
    app.include_router(ml_router, prefix=settings.API_V1_PREFIX)
    app.include_router(genai_router, prefix=settings.API_V1_PREFIX)
    app.include_router(scraper_router, prefix=settings.API_V1_PREFIX)

    # === Health check ===
    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        """Service health check endpoint."""
        return {"status": "healthy", "version": settings.VERSION, "env": settings.APP_ENV}

    return app


app = create_app()
