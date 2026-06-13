import os
import sys
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
from kmodes.kmodes import KModes
from kneed import KneeLocator
from uuid import uuid4

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Missing Supabase credentials in .env")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def main():
    logger.info("Fetching job offers from Supabase...")
    # Fetch job offers with skills
    response = supabase.table("job_offers").select("job_offer_id, raw_hard_skills").execute()
    data = response.data

    if not data:
        logger.error("No job offers found in the database.")
        sys.exit(1)

    logger.info(f"Loaded {len(data)} job offers.")

    # 1. Build the binary matrix
    all_skills = set()
    valid_offers = []
    
    for row in data:
        skills = row.get("raw_hard_skills") or []
        if skills:
            # Clean and normalize names
            clean_skills = [s.strip().lower().replace(" ", "").replace(".", "") for s in skills if s.strip()]
            if clean_skills:
                all_skills.update(clean_skills)
                valid_offers.append({"id": row["job_offer_id"], "skills": clean_skills})

    all_skills = sorted(list(all_skills))
    logger.info(f"Found {len(all_skills)} unique skills across offers.")

    # Create DataFrame (Rows: Offers, Cols: Skills)
    matrix_data = []
    for offer in valid_offers:
        row_dict = {"offer_id": offer["id"]}
        offer_skill_set = set(offer["skills"])
        for skill in all_skills:
            row_dict[skill] = 1 if skill in offer_skill_set else 0
        matrix_data.append(row_dict)

    df = pd.DataFrame(matrix_data)
    X = df[all_skills].values

    # 2. Elbow Method to find optimal K
    logger.info("Running Elbow Method (K=2 to 10)...")
    costs = []
    K_range = range(2, 11)
    
    for k in K_range:
        km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0, random_state=42)
        km.fit(X)
        costs.append(km.cost_)
        logger.info(f"K={k} -> Cost: {km.cost_}")

    # Use kneed to find the elbow point
    kneedle = KneeLocator(list(K_range), costs, curve="convex", direction="decreasing")
    optimal_k = kneedle.elbow or 6 # fallback to 6 if no clear elbow
    logger.info(f"Optimal K found: {optimal_k}")

    # 3. Train final model with optimal K
    logger.info(f"Training final K-Modes model with K={optimal_k}...")
    km_final = KModes(n_clusters=optimal_k, init='Huang', n_init=10, verbose=0, random_state=42)
    clusters = km_final.fit_predict(X)
    
    df["cluster_label"] = clusters

    # 4. Generate cluster probabilistic centroids
    logger.info("Generating cluster centroids and frequencies...")
    cluster_results = []
    
    for k in range(optimal_k):
        cluster_mask = df["cluster_label"] == k
        cluster_df = df[cluster_mask]
        cluster_size = len(cluster_df)
        
        # Calculate frequency of each skill in this cluster
        frequencies = cluster_df[all_skills].mean()
        
        # Filter skills that appear in at least 15% of the offers in this cluster
        top_skills = frequencies[frequencies >= 0.15].sort_values(ascending=False)
        
        # Give it a generic name based on top 3 skills
        top_3_names = list(top_skills.index[:3])
        cluster_name = f"Specialty: {' + '.join([n.title() for n in top_3_names])}"
        
        skills_data = []
        for skill_name, freq in top_skills.items():
            # weight logic from original PRD idea: 
            # critical if freq > 0.6 (weight 3), important > 0.3 (weight 2), desirable (weight 1)
            weight = 1.0
            if freq >= 0.6: weight = 3.0
            elif freq >= 0.3: weight = 2.0
            
            skills_data.append({
                "name": skill_name,
                "frequency": float(freq),
                "weight": float(weight)
            })
            
        cluster_results.append({
            "name": cluster_name,
            "description": f"Dominant skills: {', '.join([n.title() for n in top_3_names])}. Size: {cluster_size} offers.",
            "job_offer_count": cluster_size,
            "skills": skills_data
        })
        
        logger.info(f"Cluster {k} ({cluster_size} offers): {cluster_name}")

    # 5. Save to Supabase (Replace fake data)
    logger.info("Saving results to Supabase...")
    
    # Ensure canonical skills exist in `skills` table
    logger.info("Ensuring canonical skills exist in DB...")
    existing_skills_resp = supabase.table("skills").select("skill_id, name").execute()
    existing_skills_map = {s["name"]: s["skill_id"] for s in existing_skills_resp.data}
    
    new_skills_to_insert = []
    for s_name in all_skills:
        if s_name not in existing_skills_map:
            new_skills_to_insert.append({
                "name": s_name,
                "category": "hard_skill",
                "weight": 1.0
            })
    
    if new_skills_to_insert:
        logger.info(f"Inserting {len(new_skills_to_insert)} new canonical skills...")
        inserted_skills = supabase.table("skills").insert(new_skills_to_insert).execute()
        for s in inserted_skills.data:
            existing_skills_map[s["name"]] = s["skill_id"]
    
    # Delete existing fake diagnostics and clusters
    logger.info("Deleting existing fake diagnostics and clusters...")
    
    # We must clear diagnostics first due to RESTRICT foreign key constraint
    old_diagnostics = supabase.table("diagnostics").select("diagnostic_id").execute()
    for od in old_diagnostics.data:
        supabase.table("diagnostics").delete().eq("diagnostic_id", od["diagnostic_id"]).execute()
        
    old_clusters = supabase.table("clusters").select("cluster_id").execute()
    for oc in old_clusters.data:
        supabase.table("clusters").delete().eq("cluster_id", oc["cluster_id"]).execute()
        
    logger.info("Inserting new clusters...")
    for cr in cluster_results:
        # Insert Cluster
        cluster_id = str(uuid4())
        supabase.table("clusters").insert({
            "cluster_id": cluster_id,
            "name": cr["name"],
            "description": cr["description"],
            "job_offer_count": cr["job_offer_count"]
        }).execute()
        
        # Insert Cluster Skills
        cs_inserts = []
        for cs in cr["skills"]:
            s_id = existing_skills_map.get(cs["name"])
            if s_id:
                cs_inserts.append({
                    "cluster_id": cluster_id,
                    "skill_id": s_id,
                    "importance_score": cs["frequency"] * cs["weight"]
                })
        
        if cs_inserts:
            supabase.table("cluster_skills").insert(cs_inserts).execute()

    logger.info("Done! K-Modes model trained and real clusters deployed to DB.")

if __name__ == "__main__":
    main()
