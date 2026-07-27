import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

HYP_FILE = BASE / "FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses.csv"
EMB_FILE = BASE / "FINAL_WORKSPACE/embeddings/dino_pre2016_node2vec_embeddings.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUT_DIR / "dino_pre2016_priority_hypotheses_node2vec_scored.csv"

def cosine_similarity(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denom == 0:
        return 0.0

    return float(np.dot(v1, v2) / denom)

hyp = pd.read_csv(HYP_FILE).fillna("")
emb = pd.read_csv(EMB_FILE).fillna("")

emb_cols = [c for c in emb.columns if c.startswith("emb_")]

embedding_dict = {
    str(row["node"]): row[emb_cols].values.astype(float)
    for _, row in emb.iterrows()
}

source_target_scores = []
bridge_mean_scores = []
bridge_max_scores = []
coverage_flags = []

for _, row in hyp.iterrows():

    source = str(row["Source"]).strip()
    target = str(row["Target"]).strip()

    bridges = [
        b.strip()
        for b in str(row.get("Bridge_Nodes", "")).split(";")
        if b.strip()
    ]

    source_vec = embedding_dict.get(source)
    target_vec = embedding_dict.get(target)

    if source_vec is not None and target_vec is not None:
        st_score = cosine_similarity(source_vec, target_vec)
        has_source_target = 1
    else:
        st_score = 0.0
        has_source_target = 0

    bridge_scores = []

    for b in bridges:

        b_vec = embedding_dict.get(b)

        if b_vec is None:
            continue

        if source_vec is not None:
            bridge_scores.append(
                cosine_similarity(source_vec, b_vec)
            )

        if target_vec is not None:
            bridge_scores.append(
                cosine_similarity(target_vec, b_vec)
            )

    if bridge_scores:
        bridge_mean = float(np.mean(bridge_scores))
        bridge_max = float(np.max(bridge_scores))
    else:
        bridge_mean = 0.0
        bridge_max = 0.0

    source_target_scores.append(st_score)
    bridge_mean_scores.append(bridge_mean)
    bridge_max_scores.append(bridge_max)
    coverage_flags.append(has_source_target)

hyp["Embedding_Source_Target_Similarity"] = source_target_scores
hyp["Embedding_Bridge_Mean_Similarity"] = bridge_mean_scores
hyp["Embedding_Bridge_Max_Similarity"] = bridge_max_scores
hyp["Embedding_Coverage"] = coverage_flags

# ===============================
# NORMALIZE STRUCTURAL SCORE
# ===============================

for col in ["Score", "Bridge_Score", "Common_Neighbors", "Adamic_Adar", "Jaccard"]:
    if col not in hyp.columns:
        hyp[col] = 0

    hyp[col] = pd.to_numeric(
        hyp[col],
        errors="coerce"
    ).fillna(0)

def minmax(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())

hyp["Score_norm"] = minmax(hyp["Score"])
hyp["Bridge_Score_norm"] = minmax(hyp["Bridge_Score"])
hyp["Common_Neighbors_norm"] = minmax(hyp["Common_Neighbors"])
hyp["Adamic_Adar_norm"] = minmax(hyp["Adamic_Adar"])
hyp["Jaccard_norm"] = minmax(hyp["Jaccard"])

# ===============================
# NODE2VEC INTEGRATED SCORE
# ===============================

hyp["Node2Vec_Integrated_Score"] = (
    hyp["Score_norm"] * 0.35 +
    hyp["Bridge_Score_norm"] * 0.20 +
    hyp["Common_Neighbors_norm"] * 0.15 +
    hyp["Embedding_Source_Target_Similarity"] * 0.15 +
    hyp["Embedding_Bridge_Mean_Similarity"] * 0.10 +
    hyp["Jaccard_norm"] * 0.05
)

hyp = hyp.sort_values(
    by="Node2Vec_Integrated_Score",
    ascending=False
)

hyp.to_csv(
    OUTFILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== NODE2VEC HYPOTHESIS SCORING ==========")
print("Input hypotheses:", len(hyp))
print("Embedding-covered source-target pairs:", hyp["Embedding_Coverage"].sum())

print("\nSaved:")
print(OUTFILE)

print("\nTop Node2Vec-ranked hypotheses:")
print(
    hyp[
        [
            "Source",
            "Source_Type",
            "Target",
            "Target_Type",
            "Hypothesis_Class",
            "Score",
            "Embedding_Source_Target_Similarity",
            "Embedding_Bridge_Mean_Similarity",
            "Node2Vec_Integrated_Score",
            "Bridge_Nodes"
        ]
    ].head(30).to_string(index=False)
)
