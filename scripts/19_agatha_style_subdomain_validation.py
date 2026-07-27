import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

KG = BASE / "FINAL_WORKSPACE/kg"
ML = BASE / "FINAL_WORKSPACE/ml"
OUT = BASE / "FINAL_WORKSPACE/agatha_style_validation"
OUT.mkdir(parents=True, exist_ok=True)

PRE_FILE = KG / "dino_pre2016_semantic_edges.csv"
POST_FILE = KG / "dino_post2015_semantic_edges.csv"
AI_FILE = ML / "dino_pre2016_hypotheses_ai_ranked.csv"

OUT_CANDIDATES = OUT / "agatha_style_subdomain_candidates_ranked.csv"
OUT_METRICS = OUT / "agatha_style_subdomain_validation_metrics.csv"

K_VALUES = [10, 20, 50, 100, 200]

SUBDOMAINS = [
    ("ENV_FACTOR", "SXT_GENE"),
    ("ENV_FACTOR", "TOXIN"),
    ("SXT_GENE", "TOXIN"),
    ("SXT_GENE", "BIOLOGICAL_PROCESS"),
    ("DINO_TAXON", "SXT_GENE"),
    ("DINO_TAXON", "TOXIN"),
    ("BIOLOGICAL_PROCESS", "TOXIN"),
]

HUB_TERMS = {
    "saxitoxin", "stx", "toxin", "toxins", "toxicity",
    "paralytic_shellfish_toxins", "paralytic_shellfish_poisoning",
    "dinoflagellate", "dinoflagellates",
    "cyanobacteria", "cyanobacterial", "cyanobacterium",
    "biosynthesis", "toxin_production", "expression", "regulation"
}

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")

def pair_key(a, b):
    a, b = norm(a), norm(b)
    return tuple(sorted([a, b]))

def remove_hub(a, b):
    return norm(a) in HUB_TERMS or norm(b) in HUB_TERMS

def get_nodes_by_type(edges, node_type):
    s = edges.loc[edges["source_type"] == node_type, "source"]
    t = edges.loc[edges["target_type"] == node_type, "target"]
    return sorted(set(s.astype(str).map(norm)) | set(t.astype(str).map(norm)))

def precision_at_k(y, k):
    yk = y[:k]
    return float(np.sum(yk) / k) if k > 0 else 0.0

def hits_at_k(y, k):
    return int(np.sum(y[:k]))

def reciprocal_rank(y):
    for i, val in enumerate(y, start=1):
        if val == 1:
            return 1.0 / i
    return 0.0

def average_precision_at_k(y, k):
    yk = y[:k]
    hits = 0
    score = 0.0
    for i, val in enumerate(yk, start=1):
        if val == 1:
            hits += 1
            score += hits / i
    return score / max(1, min(np.sum(y), k))

print("\nLoading files...")

pre = pd.read_csv(PRE_FILE).fillna("")
post = pd.read_csv(POST_FILE).fillna("")
ai = pd.read_csv(AI_FILE).fillna("")

for df in [pre, post]:
    df["source"] = df["source"].map(norm)
    df["target"] = df["target"].map(norm)

pre_pairs = set(pair_key(a, b) for a, b in zip(pre["source"], pre["target"]))
post_pairs = set(pair_key(a, b) for a, b in zip(post["source"], post["target"]))

# Score lookup from AI-ranked hypotheses
ai["Source_norm"] = ai["Source"].map(norm)
ai["Target_norm"] = ai["Target"].map(norm)
ai["pair"] = ai.apply(lambda r: pair_key(r["Source_norm"], r["Target_norm"]), axis=1)

score_cols = [
    c for c in [
        "Final_AI_Rank_Score",
        "ML_Probability",
        "Node2Vec_Integrated_Score",
        "Score"
    ] if c in ai.columns
]

if not score_cols:
    raise ValueError("No usable score columns found in AI file.")

score_df = (
    ai.groupby("pair", as_index=False)[score_cols]
    .max()
)

score_lookup = {
    row["pair"]: {c: row[c] for c in score_cols}
    for _, row in score_df.iterrows()
}

all_candidates = []

print("\nGenerating subdomain candidates...")

for type_a, type_b in SUBDOMAINS:

    nodes_a = get_nodes_by_type(pre, type_a)
    nodes_b = get_nodes_by_type(pre, type_b)

    subdomain = f"{type_a}__{type_b}"

    print(f"{subdomain}: {len(nodes_a)} x {len(nodes_b)}")

    for a, b in product(nodes_a, nodes_b):

        if a == b:
            continue

        if remove_hub(a, b):
            continue

        pk = pair_key(a, b)

        # AGATHA-style: candidate must be undiscovered in training
        if pk in pre_pairs:
            continue

        label = 1 if pk in post_pairs else 0

        scores = score_lookup.get(pk, {})

        row = {
            "Subdomain": subdomain,
            "Source": a,
            "Source_Type": type_a,
            "Target": b,
            "Target_Type": type_b,
            "Pair": "||".join(pk),
            "Temporal_Label": label
        }

        for c in score_cols:
            row[c] = float(scores.get(c, 0.0))

        all_candidates.append(row)

candidates = pd.DataFrame(all_candidates)

if candidates.empty:
    raise ValueError("No AGATHA-style candidates generated. Check input KG files.")

candidates.to_csv(OUT_CANDIDATES, index=False, encoding="utf-8-sig")

print("\nSaved candidates:")
print(OUT_CANDIDATES)
print("Candidates:", len(candidates))
print("Future positives:", int(candidates["Temporal_Label"].sum()))
print("Future negatives:", int((candidates["Temporal_Label"] == 0).sum()))

metrics = []

for subdomain, g in candidates.groupby("Subdomain"):

    positives = int(g["Temporal_Label"].sum())
    total = len(g)

    if total < 2:
        continue

    for score_col in score_cols:

        ranked = g.sort_values(score_col, ascending=False).copy()
        y = ranked["Temporal_Label"].astype(int).values
        scores = ranked[score_col].astype(float).values

        row_base = {
            "Subdomain": subdomain,
            "Score_Column": score_col,
            "Candidates": total,
            "Future_Positives": positives,
            "Future_Negatives": total - positives,
            "RR": reciprocal_rank(y)
        }

        try:
            row_base["ROC_AUC"] = roc_auc_score(y, scores) if len(set(y)) > 1 else np.nan
        except Exception:
            row_base["ROC_AUC"] = np.nan

        try:
            row_base["PR_AUC"] = average_precision_score(y, scores) if len(set(y)) > 1 else np.nan
        except Exception:
            row_base["PR_AUC"] = np.nan

        for k in K_VALUES:
            if len(y) >= k:
                row_base[f"Precision@{k}"] = precision_at_k(y, k)
                row_base[f"Hits@{k}"] = hits_at_k(y, k)
                row_base[f"AP@{k}"] = average_precision_at_k(y, k)
            else:
                row_base[f"Precision@{k}"] = np.nan
                row_base[f"Hits@{k}"] = np.nan
                row_base[f"AP@{k}"] = np.nan

        metrics.append(row_base)

metrics_df = pd.DataFrame(metrics)

# Overall across all subdomains
for score_col in score_cols:

    ranked = candidates.sort_values(score_col, ascending=False).copy()
    y = ranked["Temporal_Label"].astype(int).values
    scores = ranked[score_col].astype(float).values

    row = {
        "Subdomain": "ALL_SUBDOMAINS",
        "Score_Column": score_col,
        "Candidates": len(candidates),
        "Future_Positives": int(candidates["Temporal_Label"].sum()),
        "Future_Negatives": int((candidates["Temporal_Label"] == 0).sum()),
        "RR": reciprocal_rank(y),
        "ROC_AUC": roc_auc_score(y, scores) if len(set(y)) > 1 else np.nan,
        "PR_AUC": average_precision_score(y, scores) if len(set(y)) > 1 else np.nan
    }

    for k in K_VALUES:
        row[f"Precision@{k}"] = precision_at_k(y, k)
        row[f"Hits@{k}"] = hits_at_k(y, k)
        row[f"AP@{k}"] = average_precision_at_k(y, k)

    metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)

metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

print("\nSaved metrics:")
print(OUT_METRICS)

print("\nOverall metrics:")
print(
    metrics_df[metrics_df["Subdomain"] == "ALL_SUBDOMAINS"]
    .to_string(index=False)
)
