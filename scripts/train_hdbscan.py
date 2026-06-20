import json
import logging
import os
import sys
from uuid import uuid4

import hdbscan
import numpy as np
import pandas as pd
import requests
import umap
from dotenv import load_dotenv
from sklearn.metrics import silhouette_score
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Missing Supabase credentials in .env")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configurable Hyperparameters
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 15
UMAP_MIN_DIST = 0.0
UMAP_METRIC = 'cosine'

HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_METRIC = 'euclidean'


def generate_cluster_names_batch(clusters_skills):
    if not GROQ_API_KEY:
        return None
    try:
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
        for idx, skills in enumerate(clusters_skills):
            prompt += f"Cluster {idx}: {', '.join(skills)}\n"

        prompt += "\nOutput format:\n"
        for idx in range(len(clusters_skills)):
            prompt += f"Cluster {idx}: <Name>\n"

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
            lines = content.strip().split("\n")
            names_map = {}
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    idx_str = parts[0].replace("Cluster", "").strip()
                    name = parts[1].strip().replace('"', '')
                    try:
                        idx = int(idx_str)
                        names_map[idx] = name
                    except ValueError:
                        pass

            results = []
            for idx in range(len(clusters_skills)):
                if idx in names_map:
                    results.append(names_map[idx])
                else:
                    logger.warning(f"Missing cluster name for index {idx} in batch response")
                    return None
            return results
    except Exception as e:
        logger.warning(f"LLM batch naming failed: {e}")
    return None


def generate_cluster_name(top_skills):
    if not GROQ_API_KEY:
        return f"Specialty: {' + '.join([n.title() for n in top_skills[:3]])}"
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a technical recruiter expert in IT. Give a concise, professional title (max 4 words) for an IT specialty based on these top skills. Example: 'Backend Java Developer', 'Data Engineer', 'DevOps Cloud Engineer'. Reply ONLY with the title.",
                    },
                    {"role": "user", "content": f"Skills: {', '.join(top_skills)}"},
                ],
                "temperature": 0.2,
                "max_tokens": 15,
            },
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip().replace('"', "")
    except Exception as e:
        logger.warning(f"LLM naming failed: {e}")
    return f"Specialty: {' + '.join([n.title() for n in top_skills[:3]])}"


def main():
    logger.info("Fetching skills, embeddings, and aliases from Supabase...")
    skills_resp = supabase.table("skills").select("name, embedding, skill_aliases(alias_name)").execute()

    alias_to_canonical = {}
    skill_to_embedding = {}

    if skills_resp.data:
        for s in skills_resp.data:
            canonical = s["name"].strip()
            alias_to_canonical[canonical.lower()] = canonical
            for a in s.get("skill_aliases", []):
                alias_to_canonical[a["alias_name"].lower().strip()] = canonical

            emb_str = s.get("embedding")
            if emb_str:
                try:
                    if isinstance(emb_str, str):
                        if emb_str.startswith('[') and emb_str.endswith(']'):
                            emb = json.loads(emb_str)
                        else:
                            emb = [float(x) for x in emb_str.replace('[', '').replace(']', '').split(',') if x.strip()]
                    else:
                        emb = list(emb_str)

                    if len(emb) == 1024:
                        skill_to_embedding[canonical.lower()] = emb
                except Exception as e:
                    logger.warning(f"Failed to parse embedding for skill '{canonical}': {e}")

    logger.info(f"Loaded {len(skill_to_embedding)} skills with embeddings.")

    logger.info("Fetching job offers from Supabase...")
    # Fetch job offers with raw_hard_skills (paginated)
    offers = []
    limit = 1000
    offset = 0
    while True:
        resp = (
            supabase.table("job_offers")
            .select("job_offer_id, job_title, raw_hard_skills")
            .range(offset, offset + limit - 1)
            .execute()
        )
        if not resp.data:
            break
        offers.extend(resp.data)
        if len(resp.data) < limit:
            break
        offset += limit

    logger.info(f"Loaded {len(offers)} job offers.")

    # Represent each job offer as a dense vector by mean-pooling its skill embeddings
    valid_offers = []
    for row in offers:
        raw_skills = row.get("raw_hard_skills") or []
        offer_skills = set()

        for raw_s in raw_skills:
            clean_s = raw_s.lower().strip()
            if not clean_s:
                continue

            if clean_s in alias_to_canonical:
                offer_skills.add(alias_to_canonical[clean_s])
            else:
                offer_skills.add(clean_s.replace(" ", "").replace(".", ""))

        # Find embeddings
        embeddings = []
        for s in offer_skills:
            emb = skill_to_embedding.get(s.lower())
            if emb:
                embeddings.append(emb)

        if embeddings:
            offer_vector = np.mean(embeddings, axis=0)
            valid_offers.append({
                "id": row["job_offer_id"],
                "job_title": row.get("job_title", ""),
                "skills": list(offer_skills),
                "vector": offer_vector
            })

    logger.info(f"Offers with valid skill embeddings: {len(valid_offers)} / {len(offers)}.")
    if not valid_offers:
        logger.error("No job offers have skills with valid embeddings.")
        sys.exit(1)

    vectors = np.array([o["vector"] for o in valid_offers])

    logger.info(f"Applying UMAP reduction (1024d -> {UMAP_N_COMPONENTS}d)...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42
    )
    X_reduced = reducer.fit_transform(vectors)

    logger.info("Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        prediction_data=True
    )
    labels = clusterer.fit_predict(X_reduced)

    unique_labels = set(labels)
    non_noise_labels = sorted([l for l in unique_labels if l != -1])
    n_clusters = len(non_noise_labels)
    noise_count = int(np.sum(labels == -1))
    noise_pct = (noise_count / len(labels)) * 100

    logger.info("HDBSCAN clustering results:")
    logger.info(f"- Total offers: {len(labels)}")
    logger.info(f"- Clusters found: {n_clusters}")
    logger.info(f"- Noise points: {noise_count} ({noise_pct:.2f}%)")

    if n_clusters < 2:
        logger.error("HDBSCAN found less than 2 clusters. Unable to compute Silhouette Score or proceed.")
        sys.exit(1)

    # Calculate Silhouette Score on non-noise points
    non_noise_mask = labels != -1
    X_clustered = X_reduced[non_noise_mask]
    labels_clustered = labels[non_noise_mask]

    sil_score = 0.0
    try:
        sil_score = silhouette_score(X_clustered, labels_clustered, metric='euclidean')
        logger.info(f"Silhouette Score (Euclidean on UMAP space, excluding noise): {sil_score:.4f}")
    except Exception as e:
        logger.error(f"Failed to calculate silhouette score: {e}")

    # Compute reduced-space centroids for noise-reassignment
    centroids_reduced = {}
    for k in non_noise_labels:
        cluster_points = X_reduced[labels == k]
        centroids_reduced[k] = np.mean(cluster_points, axis=0)

    # Reassign noise points to closest cluster centroid in UMAP space
    final_labels = []
    noise_reassigned_count = 0
    for i, label in enumerate(labels):
        if label != -1:
            final_labels.append(label)
        else:
            # Noise point: find closest centroid
            point = X_reduced[i]
            best_k = -1
            min_dist = float('inf')
            for k, centroid in centroids_reduced.items():
                dist = np.linalg.norm(point - centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_k = k
            final_labels.append(best_k)
            noise_reassigned_count += 1

    final_labels = np.array(final_labels)
    logger.info(f"Reassigned {noise_reassigned_count} noise offers to nearest cluster.")

    # Compute 1024d centroid vectors for each cluster (using core non-noise members)
    cluster_centroids_1024d = {}
    for k in non_noise_labels:
        core_indices = [idx for idx, label in enumerate(labels) if label == k]
        # In case a cluster consists only of noise points (shouldn't happen), fallback to final_labels
        if not core_indices:
            core_indices = [idx for idx, label in enumerate(final_labels) if label == k]
        core_vectors = vectors[core_indices]
        cluster_centroids_1024d[k] = np.mean(core_vectors, axis=0).tolist()

    # Build skill occurrences DataFrame for analysis
    skill_counts = {}
    for offer in valid_offers:
        for s in offer["skills"]:
            skill_counts[s] = skill_counts.get(s, 0) + 1

    all_skills = sorted(list(skill_counts.keys()))

    matrix_data = []
    for idx, offer in enumerate(valid_offers):
        row_dict = {
            "offer_id": offer["id"],
            "job_title": offer["job_title"],
            "cluster_label": final_labels[idx]
        }
        offer_skill_set = set(offer["skills"])
        for skill in all_skills:
            row_dict[skill] = 1 if skill in offer_skill_set else 0
        matrix_data.append(row_dict)

    df = pd.DataFrame(matrix_data)

    # Process cluster results
    cluster_results = []
    for k in non_noise_labels:
        cluster_mask = df["cluster_label"] == k
        cluster_df = df[cluster_mask]
        cluster_size = len(cluster_df)

        # Calculate compatible roles (top 5 job titles)
        top_titles_series = cluster_df["job_title"].value_counts().head(5)
        compatible_roles = []
        for title, count in top_titles_series.items():
            match_level = (
                "Alta"
                if count > (cluster_size * 0.1)
                else ("Media" if count > (cluster_size * 0.05) else "Baja")
            )
            if str(title).strip():
                compatible_roles.append(
                    {
                        "title": str(title).strip().title(),
                        "match": match_level,
                        "frequency": int(count),
                    }
                )

        # Calculate frequency of each skill in this cluster
        frequencies = cluster_df[all_skills].mean()

        # Filter skills that appear in at least 15% of the offers in this cluster
        top_skills = frequencies[frequencies >= 0.15].sort_values(ascending=False)

        # LLM Generates cluster name (will be batch named later)
        top_5_names = list(top_skills.index[:5])

        skills_data = []
        for skill_name, freq in top_skills.items():
            weight = 1.0
            if freq >= 0.6:
                weight = 3.0
            elif freq >= 0.3:
                weight = 2.0

            skills_data.append(
                {"name": skill_name, "frequency": float(freq), "weight": float(weight)}
            )

        cluster_results.append(
            {
                "cluster_index": k,
                "name": None,
                "description": None,
                "top_5_names": top_5_names,
                "job_offer_count": cluster_size,
                "skills": skills_data,
                "compatible_roles": compatible_roles,
                "centroid_vec": cluster_centroids_1024d[k],
                "offer_ids": cluster_df["offer_id"].tolist(),
            }
        )

    # Batch generate unique names
    logger.info("Batch generating unique cluster names...")
    all_clusters_skills = [cr["top_5_names"] for cr in cluster_results]
    batch_names = generate_cluster_names_batch(all_clusters_skills)

    if batch_names and len(batch_names) == len(cluster_results):
        for idx, name in enumerate(batch_names):
            cr = cluster_results[idx]
            cr["name"] = name
            top_5_names = cr["top_5_names"]
            cluster_size = cr["job_offer_count"]
            cr["description"] = f"Dominant skills: {', '.join([n.title() for n in top_5_names])}. Size: {cluster_size} offers."
            logger.info(f"Cluster {cr['cluster_index']} ({cluster_size} offers): {name}")
    else:
        logger.warning("Batch naming failed or mismatched size. Falling back to individual naming.")
        for cr in cluster_results:
            name = generate_cluster_name(cr["top_5_names"])
            cr["name"] = name
            top_5_names = cr["top_5_names"]
            cluster_size = cr["job_offer_count"]
            cr["description"] = f"Dominant skills: {', '.join([n.title() for n in top_5_names])}. Size: {cluster_size} offers."
            logger.info(f"Cluster {cr['cluster_index']} ({cluster_size} offers): {name}")

    # Save to Supabase
    logger.info("Saving results to Supabase...")

    # Ensure canonical skills exist in `skills` table (mapping skill names to IDs)
    existing_skills_resp = supabase.table("skills").select("skill_id, name").execute()
    existing_skills_map = {s["name"]: s["skill_id"] for s in existing_skills_resp.data}

    new_skills_to_insert = []
    for s_name in all_skills:
        if s_name not in existing_skills_map:
            new_skills_to_insert.append({"name": s_name, "category": "hard_skill", "weight": 1.0})

    if new_skills_to_insert:
        logger.info(f"Inserting {len(new_skills_to_insert)} new canonical skills...")
        inserted_skills = supabase.table("skills").insert(new_skills_to_insert).execute()
        for s in inserted_skills.data:
            existing_skills_map[s["name"]] = s["skill_id"]

    # Reset job_offers cluster_id to NULL
    logger.info("Unlinking job offers from clusters...")
    supabase.table("job_offers").update({"cluster_id": None}).neq(
        "job_offer_id", "00000000-0000-0000-0000-000000000000"
    ).execute()

    # Delete existing diagnostics and clusters
    logger.info("Deleting existing diagnostics and clusters...")

    old_diagnostics = supabase.table("diagnostics").select("diagnostic_id").execute()
    for od in old_diagnostics.data:
        supabase.table("diagnostics").delete().eq("diagnostic_id", od["diagnostic_id"]).execute()

    old_clusters = supabase.table("clusters").select("cluster_id").execute()
    for oc in old_clusters.data:
        supabase.table("clusters").delete().eq("cluster_id", oc["cluster_id"]).execute()

    logger.info("Inserting new clusters and linking job offers...")
    for cr in cluster_results:
        # Insert Cluster with centroid_vec
        cluster_id = str(uuid4())
        supabase.table("clusters").insert(
            {
                "cluster_id": cluster_id,
                "name": cr["name"],
                "description": cr["description"],
                "job_offer_count": cr["job_offer_count"],
                "compatible_roles": cr["compatible_roles"],
                "centroid_vec": cr["centroid_vec"]
            }
        ).execute()

        # Link job offers to this cluster in chunks of 100
        offer_ids = cr["offer_ids"]
        for i in range(0, len(offer_ids), 100):
            chunk = offer_ids[i : i + 100]
            supabase.table("job_offers").update({"cluster_id": cluster_id}).in_(
                "job_offer_id", chunk
            ).execute()

        # Insert Cluster Skills
        cs_inserts = []
        for cs in cr["skills"]:
            s_id = existing_skills_map.get(cs["name"])
            if s_id:
                cs_inserts.append(
                    {
                        "cluster_id": cluster_id,
                        "skill_id": s_id,
                        "importance_score": cs["frequency"] * cs["weight"],
                    }
                )

        if cs_inserts:
            supabase.table("cluster_skills").insert(cs_inserts).execute()

    logger.info(f"Done! HDBSCAN model trained and clusters deployed. Silhouette Score: {sil_score:.4f}")


if __name__ == "__main__":
    main()
