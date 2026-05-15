from pathlib import Path
import math
import pandas as pd
import numpy as np
import networkx as nx

from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler


# =====================================================
# INPUT / OUTPUT
# =====================================================
TRAIN_EDGE_FILE = "results/dino_temporal_normalized_v2/dino_train_pre2016_edges.csv"
TEST_EDGE_FILE = "results/dino_temporal_normalized_v2/dino_test_post2015_edges.csv"
HYP_FILE = "results/dino_temporal_normalized_v2/normalized_dino_hypotheses_ranked_validated_v2.csv"

OUTDIR = Path("results/dino_temporal_normalized_ml")
OUTDIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_FILE = OUTDIR / "normalized_dino_node2vec_embeddings.csv"
FINAL_HYP_FILE = OUTDIR / "normalized_dino_hypotheses_node2vec_ml_ranked.csv"
METRICS_FILE = OUTDIR / "normalized_node2vec_ml_temporal_metrics.csv"
ML_PERFORMANCE_FILE = OUTDIR / "semantic_ml_scoring_performance.csv"


# =====================================================
# SETTINGS
# =====================================================
DIMENSIONS = 64
WALK_LENGTH = 20
NUM_WALKS = 100
WINDOW = 5
WORKERS = 2
SEED = 42

NEGATIVE_RATIO = 1
TOP_N = 5000


# =====================================================
# HELPERS
# =====================================================
def edge_key(a, b):
    return tuple(sorted([str(a).strip().lower(), str(b).strip().lower()]))


def build_graph(edge_df):
    G = nx.Graph()

    for _, r in edge_df.iterrows():
        G.add_edge(
            str(r["Source"]),
            str(r["Target"]),
            weight=float(r.get("Weight", 1))
        )

    return G


def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def get_vec(node, emb):
    return emb.get(str(node))


def graph_features(G, u, v):
    if u not in G or v not in G:
        return {
            "CN": 0,
            "WCN": 0,
            "AA": 0,
            "PA": 0,
            "Jaccard": 0,
        }

    common = list(nx.common_neighbors(G, u, v))

    cn = len(common)

    wcn = 0
    aa = 0

    for z in common:
        w1 = G[u][z].get("weight", 1)
        w2 = G[v][z].get("weight", 1)
        wcn += min(w1, w2)

        deg = G.degree(z)
        if deg > 1:
            aa += 1 / math.log(deg)

    pa = G.degree(u) * G.degree(v)

    union = len(set(G.neighbors(u)).union(set(G.neighbors(v))))
    jaccard = cn / union if union else 0

    return {
        "CN": cn,
        "WCN": wcn,
        "AA": aa,
        "PA": pa,
        "Jaccard": jaccard,
    }


def make_features(G, pairs, emb):
    rows = []

    for u, v in pairs:
        gf = graph_features(G, u, v)

        vu = get_vec(u, emb)
        vv = get_vec(v, emb)

        if vu is not None and vv is not None:
            emb_cos = cosine(vu, vv)
            emb_l2 = float(np.linalg.norm(np.asarray(vu) - np.asarray(vv)))
        else:
            emb_cos = 0.0
            emb_l2 = 0.0

        rows.append({
            "Source": u,
            "Target": v,
            "CN": gf["CN"],
            "WCN": gf["WCN"],
            "AA": gf["AA"],
            "PA": gf["PA"],
            "Jaccard": gf["Jaccard"],
            "Embedding_Cosine": emb_cos,
            "Embedding_L2": emb_l2,
        })

    return pd.DataFrame(rows)


def train_node2vec(G):
    print("\nTraining Node2Vec embeddings...")

    node2vec = Node2Vec(
        G,
        dimensions=DIMENSIONS,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        workers=WORKERS,
        seed=SEED,
        weight_key="weight"
    )

    model = node2vec.fit(
        window=WINDOW,
        min_count=1,
        batch_words=4,
        seed=SEED
    )

    emb = {}

    for node in G.nodes():
        emb[str(node)] = model.wv[str(node)]

    emb_rows = []
    for node, vec in emb.items():
        row = {"Node": node}
        for i, val in enumerate(vec):
            row[f"emb_{i}"] = float(val)
        emb_rows.append(row)

    pd.DataFrame(emb_rows).to_csv(EMBEDDING_FILE, index=False)

    print(f"Saved embeddings: {EMBEDDING_FILE}")

    return emb


def sample_negative_pairs(G, positive_pairs, n):
    nodes = list(G.nodes())
    pos_keys = set(edge_key(a, b) for a, b in positive_pairs)

    neg = []
    seen = set()

    rng = np.random.default_rng(SEED)

    while len(neg) < n:
        u, v = rng.choice(nodes, 2, replace=False)
        k = edge_key(u, v)

        if k in pos_keys:
            continue
        if k in seen:
            continue
        if G.has_edge(u, v):
            continue

        seen.add(k)
        neg.append((u, v))

    return neg


def precision_at_k(df, k):
    sub = df.head(k)
    if len(sub) == 0:
        return 0
    return sub["Match"].sum() / len(sub)


def hits_at_k(df, k):
    return int(df.head(k)["Match"].sum())


# =====================================================
# MAIN
# =====================================================
def main():
    print("\nLoading train/test/hypothesis files...")

    train_edges = pd.read_csv(TRAIN_EDGE_FILE, low_memory=False)
    test_edges = pd.read_csv(TEST_EDGE_FILE, low_memory=False)
    hyp = pd.read_csv(HYP_FILE, low_memory=False)

    print(f"Train edges: {len(train_edges)}")
    print(f"Test edges: {len(test_edges)}")
    print(f"Hypotheses: {len(hyp)}")

    G = build_graph(train_edges)

    print(f"Train graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    emb = train_node2vec(G)

    # =====================================================
    # Prepare ML training data
    # positives = train edges
    # negatives = non-existing train graph pairs
    # =====================================================
    pos_pairs = list(zip(train_edges["Source"], train_edges["Target"]))
    neg_pairs = sample_negative_pairs(
        G,
        pos_pairs,
        len(pos_pairs) * NEGATIVE_RATIO
    )

    all_pairs = pos_pairs + neg_pairs
    y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))

    X_df = make_features(G, all_pairs, emb)
    feature_cols = [
        "CN",
        "WCN",
        "AA",
        "PA",
        "Jaccard",
        "Embedding_Cosine",
        "Embedding_L2",
    ]

    X = X_df[feature_cols].fillna(0).values

    # =====================================================
    # ML models with out-of-fold performance
    # =====================================================
    models = {
        "RandomForest_OOF": RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1
        ),
        "GradientBoosting_OOF": GradientBoostingClassifier(
            random_state=SEED
        ),
        "LogisticRegression_OOF": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    perf = []

    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining/evaluating {name}...")

        probs = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        roc = roc_auc_score(y, probs)
        pr = average_precision_score(y, probs)

        perf.append({
            "Model": name,
            "ROC_AUC": roc,
            "PR_AUC": pr
        })

        model.fit(X, y)
        fitted_models[name] = model

    perf_df = pd.DataFrame(perf)
    perf_df.to_csv(ML_PERFORMANCE_FILE, index=False)

    print("\nML performance:")
    print(perf_df.to_string(index=False))

    # =====================================================
    # Score hypotheses
    # =====================================================
    hyp_pairs = list(zip(hyp["Source"], hyp["Target"]))

    H_df = make_features(G, hyp_pairs, emb)

    # preserve existing scores
    H_df["Original_Score"] = hyp["Score"].values
    H_df["Bio_Bonus"] = hyp["Bio_Bonus"].values
    H_df["Common_Neighbors_Original"] = hyp["Common_Neighbors"].values
    H_df["Weighted_Common_Neighbors_Original"] = hyp["Weighted_Common_Neighbors"].values
    H_df["Bridge_Nodes"] = hyp["Bridge_Nodes"].values
    H_df["Match"] = hyp["Match"].values

    H = H_df[feature_cols].fillna(0).values

    rf_prob = fitted_models["RandomForest_OOF"].predict_proba(H)[:, 1]
    gb_prob = fitted_models["GradientBoosting_OOF"].predict_proba(H)[:, 1]
    lr_prob = fitted_models["LogisticRegression_OOF"].predict_proba(H)[:, 1]

    H_df["RF_Prob"] = rf_prob
    H_df["GB_Prob"] = gb_prob
    H_df["LR_Prob"] = lr_prob

    # Ensemble ML score
    H_df["ML_Score"] = (
        0.5 * H_df["GB_Prob"]
        + 0.4 * H_df["RF_Prob"]
        + 0.1 * H_df["LR_Prob"]
    )

    # Normalize original score and ML score
    scaler = MinMaxScaler()

    H_df["Original_Score_Norm"] = scaler.fit_transform(
        H_df[["Original_Score"]]
    )

    H_df["ML_Score_Norm"] = scaler.fit_transform(
        H_df[["ML_Score"]]
    )

    H_df["Final_AI_Score"] = (
        0.55 * H_df["Original_Score_Norm"]
        + 0.45 * H_df["ML_Score_Norm"]
    )

    H_df = H_df.sort_values(
        ["Final_AI_Score", "ML_Score", "Original_Score"],
        ascending=False
    ).reset_index(drop=True)

    H_df.to_csv(FINAL_HYP_FILE, index=False)

    # =====================================================
    # Temporal validation metrics
    # =====================================================
    metrics = []

    for k in [10, 20, 50, 100, 200, 500, 1000]:
        metrics.append({
            "K": k,
            "Hits@K": hits_at_k(H_df, k),
            "Precision@K": precision_at_k(H_df, k)
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_FILE, index=False)

    print("\nNode2Vec + ML temporal validation metrics:")
    print(metrics_df.to_string(index=False))

    print("\nTop validated hypotheses:")
    print(
        H_df[H_df["Match"] == 1][
            [
                "Source",
                "Target",
                "Final_AI_Score",
                "ML_Score",
                "Original_Score",
                "Embedding_Cosine",
                "Common_Neighbors_Original",
                "Bio_Bonus"
            ]
        ].head(30).to_string(index=False)
    )

    print("\nSaved:")
    print(FINAL_HYP_FILE)
    print(METRICS_FILE)
    print(ML_PERFORMANCE_FILE)


if __name__ == "__main__":
    main()
