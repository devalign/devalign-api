import os
import requests
import json
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 1. Fetch clusters and their skills
print("Fetching clusters and skills...")
clusters = supabase.table("clusters").select("cluster_id, name, description, job_offer_count").execute().data
cluster_skills = supabase.table("cluster_skills").select("cluster_id, skill_id, importance_score").execute().data
skills = supabase.table("skills").select("skill_id, name").execute().data

skills_map = {s["skill_id"]: s["name"] for s in skills}

# Group skills by cluster
cluster_to_skills = {}
for cs in cluster_skills:
    c_id = cs["cluster_id"]
    s_name = skills_map.get(cs["skill_id"])
    if s_name:
        if c_id not in cluster_to_skills:
            cluster_to_skills[c_id] = []
        cluster_to_skills[c_id].append((s_name, cs["importance_score"]))

# Sort skills by importance score for each cluster
for c_id in cluster_to_skills:
    cluster_to_skills[c_id].sort(key=lambda x: x[1], reverse=True)

clusters_input = []
for i, c in enumerate(clusters):
    c_id = c["cluster_id"]
    top_skills = [s[0] for s in cluster_to_skills.get(c_id, [])[:5]]
    clusters_input.append({
        "index": i,
        "cluster_id": c_id,
        "current_name": c["name"],
        "top_skills": top_skills
    })

print(f"Loaded {len(clusters_input)} clusters.")

# 2. Build prompt
prompt = (
    "You are a technical recruiter expert in IT. You are given a list of tech clusters, each characterized by its top skills. "
    "Your task is to assign a unique, concise, professional title (max 4 words) in Spanish for each IT specialty/cluster. "
    "For example: 'Desarrollador Java Backend', 'Ingeniero de Datos', 'Administrador de Bases de Datos', 'Desarrollador React Frontend'.\n\n"
    "CRITICAL RULES:\n"
    "1. Every single cluster name MUST be unique and distinct. Do NOT reuse names like 'Analista de Datos' or 'Desarrollador Full Stack' for different clusters. Differentiate them using their specific dominant skills.\n"
    "2. Be precise: if a cluster contains 'power bi' or 'tableau', prefer names like 'Analista de BI' or 'Especialista en Power BI'. If it contains only 'sql' and basic sheets, name it 'Analista de Datos Jr' or 'Analista de Soporte SQL'.\n"
    "3. Respond ONLY with the list of cluster names in the exact format shown below, without any intro or outro text.\n\n"
    "Input clusters:\n"
)

for c in clusters_input:
    prompt += f"Cluster {c['index']}: {', '.join(c['top_skills'])}\n"

prompt += "\nOutput format:\n"
for c in clusters_input:
    prompt += f"Cluster {c['index']}: <Name>\n"

print("Prompting Groq...")
response = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
    json={
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
    },
)

if response.status_code == 200:
    content = response.json()["choices"][0]["message"]["content"]
    print("Groq Response:\n")
    print(content)
    
    # Parse results
    lines = content.strip().split("\n")
    names_map = {}
    for line in lines:
        if ":" in line:
            parts = line.split(":", 1)
            idx_str = parts[0].replace("Cluster", "").strip()
            name = parts[1].strip()
            try:
                idx = int(idx_str)
                names_map[idx] = name
            except ValueError:
                pass
                
    # Display comparison
    print("\nComparison:")
    for c in clusters_input:
        new_name = names_map.get(c["index"], "FAILED TO PARSE")
        print(f"Skills: {c['top_skills']}")
        print(f"  Old Name: {c['current_name']}")
        print(f"  New Name: {new_name}")
else:
    print("Error:", response.status_code, response.text)
