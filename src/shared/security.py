"""Security utilities: JWT validation and auth dependencies."""

import json
import time
import urllib.request
from typing import Annotated, Any, cast

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.config import settings
from src.shared.exceptions import AuthenticationError

logger = structlog.get_logger(__name__)

# Bearer token extractor
bearer_scheme = HTTPBearer(auto_error=False)

# In-memory cache for Supabase public JWKS keys (TTL: 1 hour)
_jwks_cache: dict[str, Any] = {"keys": None, "expires_at": 0.0}
_JWKS_TTL_SECONDS = 3600


def _get_supabase_jwks(force_refresh: bool = False) -> dict[str, Any]:
    """Fetch and cache Supabase public JWKS keys for ES256/RS256 verification."""
    now = time.time()
    if not force_refresh and _jwks_cache["keys"] is not None and now < _jwks_cache["expires_at"]:
        return cast("dict[str, Any]", _jwks_cache["keys"])

    supabase_url = str(getattr(settings, "SUPABASE_URL", "")).rstrip("/")
    if not supabase_url:
        raise AuthenticationError("SUPABASE_URL is not configured")

    jwks_url = f"{supabase_url}/auth/v1/.well-known/jwks.json"
    try:
        with urllib.request.urlopen(jwks_url, timeout=5) as response:
            jwks_data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        _jwks_cache["keys"] = jwks_data
        _jwks_cache["expires_at"] = now + _JWKS_TTL_SECONDS
        return jwks_data
    except Exception as exc:
        logger.error("Failed to fetch Supabase JWKS", url=jwks_url, error=str(exc))
        if _jwks_cache["keys"] is not None:
            return cast("dict[str, Any]", _jwks_cache["keys"])
        raise AuthenticationError("Unable to retrieve authentication keys") from exc


def decode_jwt_token(token: str) -> dict[str, object]:
    """Decode and validate a Supabase JWT token (ES256 via JWKS or legacy HS256)."""
    try:
        unverified_header = jwt.get_unverified_header(token)
        alg = unverified_header.get("alg", "")

        if alg in ("ES256", "RS256"):
            jwks = _get_supabase_jwks()
            try:
                payload: dict[str, object] = jwt.decode(
                    token,
                    jwks,
                    algorithms=["ES256", "RS256"],
                    options={"verify_aud": False, "verify_signature": True},
                )
                return payload
            except JWTError:
                # Key rotation or cache invalidation: retry once with fresh keys if cache was used
                if _jwks_cache["keys"] is not None:
                    fresh_jwks = _get_supabase_jwks(force_refresh=True)
                    retry_payload: dict[str, object] = jwt.decode(
                        token,
                        fresh_jwks,
                        algorithms=["ES256", "RS256"],
                        options={"verify_aud": False, "verify_signature": True},
                    )
                    return retry_payload
                raise

        secret = settings.SUPABASE_JWT_SECRET or settings.SECRET_KEY
        if not secret:
            logger.error("Neither SUPABASE_JWT_SECRET nor SECRET_KEY is configured")
            raise AuthenticationError("Authentication service configuration error")

        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_aud": False, "verify_signature": True},
        )
        return payload
    except AuthenticationError:
        raise
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


async def get_optional_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str | None:
    """FastAPI dependency: extract and validate user ID from JWT if present, return None otherwise."""
    if credentials is None:
        return None
    try:
        payload = decode_jwt_token(credentials.credentials)
        user_id = payload.get("sub")
        if isinstance(user_id, str) and user_id:
            return user_id
    except Exception:
        pass
    return None


async def get_current_user_payload(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> dict[str, object]:
    """FastAPI dependency: extract and validate full JWT payload."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_jwt_token(credentials.credentials)


# Type aliases for cleaner dependency injection
CurrentUserIdDep = Annotated[str, Depends(get_current_user_id)]
CurrentUserPayloadDep = Annotated[dict[str, object], Depends(get_current_user_payload)]
OptionalUserIdDep = Annotated[str | None, Depends(get_optional_user_id)]
