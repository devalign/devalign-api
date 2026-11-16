"""Unit tests for JWT decoding and security utilities."""

import base64
import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

from src.config import settings
from src.shared import security
from src.shared.exceptions import AuthenticationError
from src.shared.security import (
    decode_jwt_token,
    get_current_user_id,
    get_optional_user_id,
)


@pytest.fixture(autouse=True)
def reset_jwks_cache() -> None:
    """Ensure in-memory JWKS cache is empty before each test."""
    security._jwks_cache = {"keys": None, "expires_at": 0.0}


def _create_token(payload: dict[str, object], secret: str | None = None) -> str:
    key = secret or settings.SUPABASE_JWT_SECRET or settings.SECRET_KEY
    return jwt.encode(payload, key, algorithm="HS256")


def _create_es256_pair_and_token(
    payload: dict[str, object], kid: str = "key-test-1"
) -> tuple[str, dict[str, Any]]:
    """Generate an EC P-256 keypair, signed ES256 token, and matching JWKS payload."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    numbers = private_key.public_key().public_numbers()
    x_b64 = base64.urlsafe_b64encode(numbers.x.to_bytes(32, byteorder="big")).decode().rstrip("=")
    y_b64 = base64.urlsafe_b64encode(numbers.y.to_bytes(32, byteorder="big")).decode().rstrip("=")

    jwks = {
        "keys": [
            {
                "kty": "EC",
                "crv": "P-256",
                "x": x_b64,
                "y": y_b64,
                "kid": kid,
                "alg": "ES256",
                "use": "sig",
            }
        ]
    }
    token = jwt.encode(payload, priv_pem, algorithm="ES256", headers={"kid": kid})
    return token, jwks


def _mock_jwks_response(jwks_data: dict[str, Any]) -> MagicMock:
    """Create a mock HTTP response object returning JSON bytes for urlopen."""
    response = MagicMock()
    response.read.return_value = json.dumps(jwks_data).encode("utf-8")
    response.__enter__.return_value = response
    return response


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


def test_decode_valid_es256_jwt_token() -> None:
    """An ES256 token validated via Supabase JWKS must decode successfully."""
    uid = str(uuid4())
    token, jwks = _create_es256_pair_and_token({"sub": uid, "email": "es256@example.com"})

    with patch("urllib.request.urlopen", return_value=_mock_jwks_response(jwks)):
        payload = decode_jwt_token(token)

    assert payload["sub"] == uid
    assert payload["email"] == "es256@example.com"


def test_decode_es256_token_key_rotation_refreshes_jwks() -> None:
    """When a cached JWKS fails to verify a token, forced refresh retrieves new keys."""
    uid = str(uuid4())
    _, stale_jwks = _create_es256_pair_and_token({"sub": "stale"}, kid="stale-kid")
    fresh_token, fresh_jwks = _create_es256_pair_and_token(
        {"sub": uid, "email": "rotated@example.com"}, kid="fresh-kid"
    )

    # Pre-populate cache with stale keys
    security._jwks_cache = {"keys": stale_jwks, "expires_at": 9999999999.0}

    with patch("urllib.request.urlopen", return_value=_mock_jwks_response(fresh_jwks)):
        payload = decode_jwt_token(fresh_token)

    assert payload["sub"] == uid
    assert payload["email"] == "rotated@example.com"


def test_decode_es256_jwks_network_failure_raises_authentication_error() -> None:
    """When JWKS cannot be retrieved and cache is empty, AuthenticationError is raised."""
    uid = str(uuid4())
    token, _ = _create_es256_pair_and_token({"sub": uid})

    with (
        patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")),
        pytest.raises(AuthenticationError, match="Unable to retrieve authentication keys"),
    ):
        decode_jwt_token(token)


def test_decode_es256_missing_supabase_url_raises_authentication_error() -> None:
    """When SUPABASE_URL is not set, ES256 validation raises AuthenticationError."""
    uid = str(uuid4())
    token, _ = _create_es256_pair_and_token({"sub": uid})

    with (
        patch.object(settings, "SUPABASE_URL", ""),
        pytest.raises(AuthenticationError, match="SUPABASE_URL is not configured"),
    ):
        decode_jwt_token(token)


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
