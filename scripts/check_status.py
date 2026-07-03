import asyncio, asyncpg, sys
sys.path.insert(0, "src")
from config import settings

async def main():
    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    
    rows = await conn.fetch(
        "SELECT u.user_id, u.email, c.status, c.error_message, c.uploaded_at "
        "FROM users u LEFT JOIN cv_documents c ON u.user_id = c.user_id "
        "WHERE u.email LIKE $1 ORDER BY c.uploaded_at DESC",
        "test@example.com"
    )
    for r in rows:
        print(f"User: {r['user_id']}, Email: {r['email']}, Status: {r['status']}, Error: {r['error_message']}, Uploaded: {r['uploaded_at']}")
    
    profile = await conn.fetchrow(
        "SELECT profile_id, user_id, full_name, current_job_role, is_diagnosed, "
        "cv_raw_text IS NOT NULL as has_raw_text, primary_specialty, alignment_score "
        "FROM profiles WHERE user_id = $1",
        "f01b1256-d8b0-4e2e-98c2-1553a71dee1d"
    )
    if profile:
        print(f"Profile: {dict(profile)}")
    else:
        print("No profile for f01b1256")
    
    await conn.close()

asyncio.run(main())
