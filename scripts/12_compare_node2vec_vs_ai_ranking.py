import pandas as pd
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

NODE2VEC_FILE = BASE / "FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses_node2vec_scored.csv"

AI_FILE = BASE / "FINAL_WORKSPACE/ml/dino_pre2016_hypotheses_ai_ranked.csv"

FUTURE_EDGE_FILE = BASE / "FINAL_WORKSPACE/kg/dino_post2015_semantic_edges.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUT_DIR / "node2vec_vs_ai_comparison_metrics.csv"

K_VALUES = [10, 20, 50, 100, 200]

# ===============================
# FUNCTIONS
# ===============================

def pair_key(a, b):
    a = str(a).strip()
    b = str(b).strip()
    return "||".join(sorted([a, b]))

def evaluate(df, score_col, future_pairs, method_name):

    df = df.copy()

    df["pair_key"] = df.apply(
        lambda r: pair_key(r["Source"], r["Target"]),
        axis=1
    )

    df["Temporal_Validated"] = df["pair_key"].apply(
        lambda x: 1 if x in future_pairs else 0
    )

    df = df.sort_values(
        by=score_col,
        ascending=False
    ).reset_index(drop=True)

    total_validated = int(df["Temporal_Validated"].sum())

    rows = []

    for k in K_VALUES:

        topk = df.head(k)

        hits = int(topk["Temporal_Validated"].sum())

        precision = hits / k

        recall = (
            hits / total_validated
            if total_validated > 0 else 0
        )

        rows.append({
            "Method": method_name,
            "Score_Column": score_col,
            "K": k,
            "Hits@K": hits,
            "Precision@K": precision,
            "Recall@K": recall
        })

    return pd.DataFrame(rows)

# ===============================
# LOAD FILES
# ===============================

node2vec_df = pd.read_csv(NODE2VEC_FILE).fillna("")
ai_df = pd.read_csv(AI_FILE).fillna("")
future_df = pd.read_csv(FUTURE_EDGE_FILE).fillna("")

future_df["pair_key"] = future_df.apply(
    lambda r: pair_key(r["source"], r["target"]),
    axis=1
)

future_pairs = set(future_df["pair_key"])

# ===============================
# EVALUATE NODE2VEC
# ===============================

node2vec_metrics = evaluate(
    node2vec_df,
    "Node2Vec_Integrated_Score",
    future_pairs,
    "Node2Vec"
)

# ===============================
# EVALUATE AI RANKER
# ===============================

ai_metrics = evaluate(
    ai_df,
    "Final_AI_Rank_Score",
    future_pairs,
    "AI_Ranker"
)

# ===============================
# COMBINE
# ===============================

combined = pd.concat(
    [node2vec_metrics, ai_metrics],
    ignore_index=True
)

combined.to_csv(
    OUTFILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== NODE2VEC VS AI COMPARISON ==========")

print("\nCombined metrics:")
print(combined.to_string(index=False))

print("\nSaved:")
print(OUTFILE)

# ===============================
# BEST METHODS
# ===============================

best_per_k = combined.sort_values(
    ["K", "Precision@K"],
    ascending=[True, False]
).groupby("K").head(1)

print("\nBest method by K:")
print(best_per_k.to_string(index=False))
