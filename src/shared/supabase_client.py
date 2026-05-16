"""Supabase client singleton."""

from functools import lru_cache

from supabase import Client, create_client

from src.config import settings


@lru_cache
def get_supabase_client() -> Client:
    """Return cached Supabase anon client (for public operations)."""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)


@lru_cache
def get_supabase_admin_client() -> Client:
    """Return cached Supabase admin client (service role — server-side only)."""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
