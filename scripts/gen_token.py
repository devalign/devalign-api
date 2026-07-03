import sys
sys.path.insert(0, "src")
from jose import jwt
from config import settings
import uuid

payload = {
    "sub": str(uuid.uuid4()),
    "email": "test@example.com",
    "user_metadata": {"full_name": "Test User"},
    "exp": 9999999999,
}
token = jwt.encode(payload, settings.SUPABASE_ANON_KEY, algorithm="HS256")
print(token)
