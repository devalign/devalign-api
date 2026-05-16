"""Security utilities: JWT validation and auth dependencies."""

from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.config import settings
from src.shared.exceptions import AuthenticationError

logger = structlog.get_logger(__name__)

# Bearer token extractor
bearer_scheme = HTTPBearer(auto_error=False)


def decode_jwt_token(token: str) -> dict[str, object]:
    """Decode and validate a Supabase JWT token."""
    try:
        payload: dict[str, object] = jwt.decode(
            token,
            settings.SUPABASE_ANON_KEY,  # Supabase uses anon key for JWT validation
            algorithms=["HS256"],
            options={"verify_aud": False},  # Supabase doesn't require aud by default
        )
        return payload
    except JWTError as exc:
        logger.warning("JWT validation failed", error=str(exc))
        raise AuthenticationError("Invalid or expired token") from exc


async def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str:
    """FastAPI dependency: extract and validate current user ID from JWT."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_jwt_token(credentials.credentials)
    user_id = payload.get("sub")

    if not isinstance(user_id, str) or not user_id:
        raise AuthenticationError("Invalid token: missing user ID")

    return user_id


# Type alias for cleaner dependency injection
CurrentUserIdDep = Annotated[str, Depends(get_current_user_id)]
