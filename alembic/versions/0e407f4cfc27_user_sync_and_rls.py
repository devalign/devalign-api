"""user_sync_and_rls

Revision ID: 0e407f4cfc27
Revises: cd103c833107
Create Date: 2026-06-28 19:02:12.782722

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0e407f4cfc27"
down_revision: str | None = "cd103c833107"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _supabase_env_exists() -> bool:
    """Return True when the 'auth' schema is present (i.e. we are on Supabase).

    Plain-PostgreSQL CI environments do not have the Supabase-managed 'auth'
    schema, so any DDL that references auth.users or auth.uid() must be
    skipped there.
    """
    conn = op.get_bind()
    result = conn.execute(
        sa.text("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'auth'")
    )
    return result.scalar() is not None


def upgrade() -> None:
    # ── 1. Always: create the trigger *function* (safe on plain PG too) ──────
    op.execute("""
        CREATE OR REPLACE FUNCTION public.handle_new_user()
        RETURNS trigger AS $$
        BEGIN
          INSERT INTO public.users (user_id, email, full_name, avatar_url, created_at)
          VALUES (
            new.id,
            new.email,
            COALESCE(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name', ''),
            COALESCE(new.raw_user_meta_data->>'avatar_url', new.raw_user_meta_data->>'picture', ''),
            COALESCE(new.created_at, now())
          )
          ON CONFLICT (user_id) DO NOTHING;
          RETURN new;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';
    """)

    if not _supabase_env_exists():
        # Running in plain-PostgreSQL CI — skip Supabase-only DDL.
        return

    # ── 2. Attach trigger on auth.users (Supabase only) ──────────────────────
    op.execute("""
        CREATE OR REPLACE TRIGGER on_auth_user_created
          AFTER INSERT ON auth.users
          FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
    """)

    # ── 3. Backfill existing users from auth.users (Supabase only) ───────────
    op.execute("""
        INSERT INTO public.users (user_id, email, full_name, avatar_url, created_at)
        SELECT
          id,
          email,
          COALESCE(raw_user_meta_data->>'full_name', raw_user_meta_data->>'name', ''),
          COALESCE(raw_user_meta_data->>'avatar_url', raw_user_meta_data->>'picture', ''),
          COALESCE(created_at, now())
        FROM auth.users
        ON CONFLICT (user_id) DO NOTHING;
    """)

    # ── 4. Enable Row Level Security (Supabase only) ──────────────────────────
    op.execute("ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE public.cv_documents ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE public.diagnostics ENABLE ROW LEVEL SECURITY;")

    # ── 5. RLS Policies that reference auth.uid() (Supabase only) ────────────
    op.execute("""
        CREATE POLICY "Users can manage their own user record"
        ON public.users FOR ALL TO authenticated USING (user_id = auth.uid());
    """)
    op.execute("""
        CREATE POLICY "Users can manage their own profile"
        ON public.profiles FOR ALL TO authenticated USING (user_id = auth.uid());
    """)
    op.execute("""
        CREATE POLICY "Users can manage their own CVs"
        ON public.cv_documents FOR ALL TO authenticated USING (user_id = auth.uid());
    """)
    op.execute("""
        CREATE POLICY "Users can view their own diagnostics"
        ON public.diagnostics FOR ALL TO authenticated
        USING (profile_id IN (SELECT profile_id FROM public.profiles WHERE user_id = auth.uid()));
    """)


def downgrade() -> None:
    if not _supabase_env_exists():
        # Nothing Supabase-specific was applied in upgrade — just drop the function.
        op.execute("DROP FUNCTION IF EXISTS public.handle_new_user();")
        return

    # ── 1. Drop RLS Policies ─────────────────────────────────────────────────
    op.execute(
        'DROP POLICY IF EXISTS "Users can view their own diagnostics" ON public.diagnostics;'
    )
    op.execute('DROP POLICY IF EXISTS "Users can manage their own CVs" ON public.cv_documents;')
    op.execute('DROP POLICY IF EXISTS "Users can manage their own profile" ON public.profiles;')
    op.execute('DROP POLICY IF EXISTS "Users can manage their own user record" ON public.users;')

    # ── 2. Disable Row Level Security (RLS) ──────────────────────────────────
    op.execute("ALTER TABLE public.diagnostics DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE public.cv_documents DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE public.profiles DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE public.users DISABLE ROW LEVEL SECURITY;")

    # ── 3. Drop Trigger and Function ─────────────────────────────────────────
    op.execute("DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;")
    op.execute("DROP FUNCTION IF EXISTS public.handle_new_user();")
