"""Custom exception hierarchy for Devalign API."""

from http import HTTPStatus


class DevalignException(Exception):  # noqa: N818
    """Base exception for all application-level errors."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


# === 4xx Client Errors ===


class ValidationError(DevalignException):
    """Raised when input data fails validation."""

    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    detail = "Validation error"


class NotFoundError(DevalignException):
    """Raised when a requested resource does not exist."""

    status_code = HTTPStatus.NOT_FOUND
    detail = "Resource not found"


class ConflictError(DevalignException):
    """Raised when a resource already exists."""

    status_code = HTTPStatus.CONFLICT
    detail = "Resource already exists"


class AuthenticationError(DevalignException):
    """Raised when authentication credentials are invalid."""

    status_code = HTTPStatus.UNAUTHORIZED
    detail = "Authentication required"


class AuthorizationError(DevalignException):
    """Raised when the user lacks permission to perform an action."""

    status_code = HTTPStatus.FORBIDDEN
    detail = "Insufficient permissions"


class FileTooLargeError(DevalignException):
    """Raised when an uploaded file exceeds the size limit."""

    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    detail = "File size exceeds the maximum allowed limit"


class UnsupportedFileTypeError(DevalignException):
    """Raised when an uploaded file type is not supported."""

    status_code = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    detail = "File type not supported"


# === 5xx Server Errors ===


class ExternalServiceError(DevalignException):
    """Raised when an external service (LLM, Supabase) fails."""

    status_code = HTTPStatus.BAD_GATEWAY
    detail = "External service unavailable"


class MLPipelineError(DevalignException):
    """Raised when the ML inference pipeline fails."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    detail = "ML pipeline error"


class RAGPipelineError(DevalignException):
    """Raised when the RAG/LLM generation pipeline fails."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    detail = "Roadmap generation error"
