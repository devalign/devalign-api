"""Application middleware: request logging and global error handling."""

import time
import uuid
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.shared.exceptions import DevalignException

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs every request with duration and injects a request_id."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:  # type: ignore[type-arg]
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Bind request_id to all logs within this request context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        logger.info("Request started")

        try:
            response: Response = await call_next(request)
        except DevalignException as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(
                "Request failed with known error",
                status_code=exc.status_code,
                detail=exc.detail,
                duration_ms=duration_ms,
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail, "request_id": request_id},
            )
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.exception(
                "Unhandled exception",
                duration_ms=duration_ms,
                exc_info=exc,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Request completed",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        # Inject request_id into response headers for tracing
        response.headers["X-Request-ID"] = request_id
        return response
