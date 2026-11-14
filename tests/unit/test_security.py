"""Unit tests for JWT decoding and security utilities."""

from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

from src.config import settings
from src.shared.exceptions import AuthenticationError
from src.shared.security import (
    decode_jwt_token,
    get_current_user_id,
    get_optional_user_id,
)


def _create_token(payload: dict[str, object], secret: str | None = None) -> str:
    key = secret or settings.SUPABASE_JWT_SECRET or settings.SECRET_KEY
    return jwt.encode(payload, key, algorithm="HS256")


def test_decode_valid_jwt_token() -> None:
    """A properly signed token must decode successfully."""
    uid = str(uuid4())
    token = _create_token({"sub": uid, "email": "dev@example.com"})

    payload = decode_jwt_token(token)
    assert payload["sub"] == uid
    assert payload["email"] == "dev@example.com"


def test_decode_forged_jwt_token_fails() -> None:
    """A token signed with an invalid secret must be rejected."""
    uid = str(uuid4())
    forged_token = _create_token({"sub": uid}, secret="wrong-secret-key-attacker")

    with pytest.raises(AuthenticationError, match="Invalid or expired token"):
        decode_jwt_token(forged_token)


def test_decode_tampered_jwt_token_fails() -> None:
    """A malformed or tampered token string must be rejected."""
    with pytest.raises(AuthenticationError, match="Invalid or expired token"):
        decode_jwt_token("invalid.token.payload")


@pytest.mark.asyncio
async def test_get_current_user_id_success() -> None:
    """Valid credentials return the subject string."""
    uid = str(uuid4())
    token = _create_token({"sub": uid})
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    result = await get_current_user_id(creds)
    assert result == uid


@pytest.mark.asyncio
async def test_get_current_user_id_missing_credentials_raises_401() -> None:
    """Missing credentials raise HTTP 401 Unauthorized."""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_id(None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_optional_user_id() -> None:
    """Optional user ID returns None when absent and UID when present."""
    assert await get_optional_user_id(None) is None

    uid = str(uuid4())
    token = _create_token({"sub": uid})
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    assert await get_optional_user_id(creds) == uid

    bad_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad")
    assert await get_optional_user_id(bad_creds) is None
