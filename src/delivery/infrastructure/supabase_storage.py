"""Supabase Storage adapter for CV uploads."""

import contextlib
import re
import unicodedata
import uuid
from uuid import UUID

import structlog
from supabase import Client

from src.delivery.domain.ports import StorageService
from src.shared.exceptions import ExternalServiceError

logger = structlog.get_logger(__name__)

CV_BUCKET = "cvs"


def _sanitize_filename(filename: str) -> str:
    """Normalize unicode characters and replace non-alphanumeric with underscores to prevent InvalidKey in cloud storage."""
    # Decompose unicode characters (accents, bullet points, etc.)
    normalized = unicodedata.normalize("NFKD", filename)
    # Convert to ASCII, ignoring non-ASCII characters like bullet points
    ascii_encoded = normalized.encode("ascii", "ignore").decode("ascii")
    # Replace non-alphanumeric (except dot, underscore, hyphen) with underscores
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", ascii_encoded)
    # Remove consecutive underscores
    return re.sub(r"_+", "_", safe_name)


class SupabaseStorageService(StorageService):
    """Implements StorageService port using Supabase Storage."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def upload_cv(
        self,
        user_id: UUID,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        """Upload CV to Supabase Storage bucket."""
        # Unique path: cvs/{user_id}/{uuid}_{filename}
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = _sanitize_filename(filename)
        storage_path = f"{user_id}/{unique_id}_{safe_filename}"

        # Ensure the bucket exists
        with contextlib.suppress(Exception):
            self._client.storage.create_bucket(CV_BUCKET, options={"public": False})

        try:
            self._client.storage.from_(CV_BUCKET).upload(
                path=storage_path,
                file=content,
                file_options={"content-type": content_type, "upsert": "true"},
            )
            logger.info("CV uploaded to storage", path=storage_path)
            return storage_path
        except Exception as exc:
            logger.error("Supabase storage upload failed", error=str(exc))
            raise ExternalServiceError("Failed to upload CV to storage") from exc

    async def get_signed_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """Generate a signed URL for temporary file access."""
        try:
            result = self._client.storage.from_(CV_BUCKET).create_signed_url(
                path=storage_path,
                expires_in=expires_in,
            )
            return str(result["signedURL"])
        except Exception as exc:
            logger.error("Failed to create signed URL", path=storage_path, error=str(exc))
            raise ExternalServiceError("Failed to generate download URL") from exc
