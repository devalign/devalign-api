import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Fetching clusters...")
clusters_resp = supabase.table("clusters").select("cluster_id, name, created_at").execute()
for c in clusters_resp.data:
    print(f"ID: {c['cluster_id']} | Name: {c['name']} | Created At: {c['created_at']}")
