import os
import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
print("Groq API Key present:", GROQ_API_KEY is not None)

clusters_data = [
    {"index": 0, "skills": ["dominio de hojas de cálculo", "sql"]},
    {"index": 1, "skills": ["dominio de hojas de cálculo", "power bi", "sql"]},
    {"index": 2, "skills": ["sql"]},
    {"index": 3, "skills": ["dominio de hojas de cálculo"]},
    {"index": 4, "skills": ["qa"]},
    {"index": 5, "skills": ["dominio de hojas de cálculo", "power bi", "python", "sql"]},
]

prompt = "You are a technical recruiter expert in IT. You are given a list of clusters, each characterized by its top skills. Assign a unique, concise, professional title (max 4 words) in Spanish for each IT specialty/cluster (e.g. 'Desarrollador Java Backend', 'Ingeniero de Datos', 'Administrador de Bases de Datos'). The names must be completely unique and distinct from each other, even if some clusters share similar skills. For instance, do not name multiple clusters 'Analista de Datos'; differentiate them based on their specific skills (e.g., 'Analista de Excel', 'Analista de Datos y Power BI', 'Analista de Business Intelligence').\n\nInput:\n"
for c in clusters_data:
    prompt += f"Cluster {c['index']}: {', '.join(c['skills'])}\n"

prompt += "\nOutput format must be precisely:\nCluster <num>: <Name>\nReturn ONLY the output list."

print("Prompting Groq...")
try:
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
    print("Status:", response.status_code)
    if response.status_code == 200:
        print(response.json()["choices"][0]["message"]["content"])
    else:
        print(response.text)
except Exception as e:
    print("Error:", e)
