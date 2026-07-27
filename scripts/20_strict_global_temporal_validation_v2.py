import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ==========================================================
# PATHS
# ==========================================================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

KG = BASE / "FINAL_WORKSPACE/kg"
ML = BASE / "FINAL_WORKSPACE/ml"

OUT = BASE / "FINAL_WORKSPACE/strict_validation_v2"
OUT.mkdir(parents=True, exist_ok=True)

PRE_FILE = KG / "dino_pre2016_semantic_edges.csv"
POST_FILE = KG / "dino_post2015_semantic_edges.csv"

OUT_CANDIDATES = OUT / "strict_global_candidates.csv"
OUT_RESULTS = OUT / "strict_global_ml_results.csv"

# ==========================================================
# SETTINGS
# ==========================================================

K_VALUES = [10, 20, 50, 100, 200]

HUB_TERMS = {
    "saxitoxin",
    "stx",
    "toxin",
    "toxins",
    "toxicity",
    "paralytic_shellfish_toxins",
    "paralytic_shellfish_poisoning",
    "biosynthesis",
    "toxin_production",
    "expression",
    "regulation",
    "dinoflagellate",
    "dinoflagellates",
    "cyanobacteria",
    "cyanobacterial",
    "cyanobacterium"
}

RANDOM_STATE = 42

# ==========================================================
# HELPERS
# ==========================================================

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")

def pair_key(a, b):
    return tuple(sorted([norm(a), norm(b)]))

def is_hub(x):
    return norm(x) in HUB_TERMS

def precision_at_k(y, k):
    yk = y[:k]
    return np.sum(yk) / k if k > 0 else 0

def hits_at_k(y, k):
    return int(np.sum(y[:k]))

def reciprocal_rank(y):
    for i, val in enumerate(y, start=1):
        if val == 1:
            return 1 / i
    return 0

# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading data...")

pre = pd.read_csv(PRE_FILE).fillna("")
post = pd.read_csv(POST_FILE).fillna("")

for df in [pre, post]:

    df["source"] = df["source"].map(norm)
    df["target"] = df["target"].map(norm)

# ==========================================================
# BUILD PRE-2016 GRAPH
# ==========================================================

print("\nBuilding graph...")

G = nx.Graph()

for _, r in pre.iterrows():

    s = r["source"]
    t = r["target"]

    if is_hub(s) or is_hub(t):
        continue

    if s == t:
        continue

    G.add_edge(s, t)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# ==========================================================
# EXISTING EDGES
# ==========================================================

pre_pairs = set(
    pair_key(a, b)
    for a, b in zip(pre["source"], pre["target"])
)

post_pairs = set(
    pair_key(a, b)
    for a, b in zip(post["source"], post["target"])
)

# ==========================================================
# GENERATE STRICT CANDIDATES
# ==========================================================

print("\nGenerating candidate pairs...")

nodes = sorted(G.nodes())

candidates = []

for a, b in combinations(nodes, 2):

    pk = pair_key(a, b)

    # undiscovered only
    if pk in pre_pairs:
        continue

    label = 1 if pk in post_pairs else 0

    candidates.append({
        "source": a,
        "target": b,
        "label": label
    })

cand = pd.DataFrame(candidates)

print("Candidates:", len(cand))
print("Future positives:", int(cand["label"].sum()))
print("Future negatives:", int((cand["label"] == 0).sum()))

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

print("\nExtracting graph features...")

features = []

for _, r in cand.iterrows():

    u = r["source"]
    v = r["target"]

    # graph features
    common_neighbors = len(list(nx.common_neighbors(G, u, v)))

    try:
        jaccard = next(
            nx.jaccard_coefficient(G, [(u, v)])
        )[2]
    except:
        jaccard = 0

    try:
        adamic = next(
            nx.adamic_adar_index(G, [(u, v)])
        )[2]
    except:
        adamic = 0

    try:
        pref_attach = next(
            nx.preferential_attachment(G, [(u, v)])
        )[2]
    except:
        pref_attach = 0

    deg_u = G.degree(u)
    deg_v = G.degree(v)

    features.append({
        "source": u,
        "target": v,
        "label": r["label"],

        "common_neighbors": common_neighbors,
        "jaccard": jaccard,
        "adamic_adar": adamic,
        "preferential_attachment": pref_attach,
        "degree_u": deg_u,
        "degree_v": deg_v
    })

feat = pd.DataFrame(features)

feat.to_csv(OUT_CANDIDATES, index=False)

print("\nSaved:")
print(OUT_CANDIDATES)

# ==========================================================
# ML DATA
# ==========================================================

X = feat[
    [
        "common_neighbors",
        "jaccard",
        "adamic_adar",
        "preferential_attachment",
        "degree_u",
        "degree_v"
    ]
]

y = feat["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE
)

# ==========================================================
# MODELS
# ==========================================================

models = {

    "LogisticRegression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE
        ),

    "GradientBoosting":
        GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

    "ExtraTrees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE
        ),

    "SVM":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),

    "MLP":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=2000,
                random_state=RANDOM_STATE
            ))
        ])
}

# ==========================================================
# TRAIN + EVALUATE
# ==========================================================

results = []

print("\nTraining models...")

for name, model in models.items():

    print(f"\n{name}")

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    temp = pd.DataFrame({
        "truth": y_test.values,
        "score": probs
    })

    temp = temp.sort_values("score", ascending=False)

    y_ranked = temp["truth"].values

    row = {
        "Model": name,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "RR": reciprocal_rank(y_ranked)
    }

    for k in K_VALUES:

        row[f"Precision@{k}"] = precision_at_k(y_ranked, k)
        row[f"Hits@{k}"] = hits_at_k(y_ranked, k)

    results.append(row)

results_df = pd.DataFrame(results)

results_df.to_csv(OUT_RESULTS, index=False)

print("\nSaved:")
print(OUT_RESULTS)

print("\nFinal results:")
print(results_df.to_string(index=False))
