import pandas as pd
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

RANKED_FILE = BASE / "FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses_node2vec_scored.csv"
FUTURE_EDGE_FILE = BASE / "FINAL_WORKSPACE/kg/dino_post2015_semantic_edges.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATED_OUT = OUT_DIR / "strict_temporal_validated_hypotheses.csv"
METRICS_OUT = OUT_DIR / "strict_temporal_validation_metrics.csv"

K_VALUES = [10, 20, 50, 100, 200, 500]

def pair_key(a, b):
    a = str(a).strip()
    b = str(b).strip()
    return "||".join(sorted([a, b]))

hyp = pd.read_csv(RANKED_FILE).fillna("")
future = pd.read_csv(FUTURE_EDGE_FILE).fillna("")

future["pair_key"] = future.apply(
    lambda r: pair_key(r["source"], r["target"]),
    axis=1
)

future_pairs = set(future["pair_key"])

hyp["pair_key"] = hyp.apply(
    lambda r: pair_key(r["Source"], r["Target"]),
    axis=1
)

hyp["Temporal_Validated"] = hyp["pair_key"].apply(
    lambda x: 1 if x in future_pairs else 0
)

hyp = hyp.sort_values(
    by="Node2Vec_Integrated_Score",
    ascending=False
).reset_index(drop=True)

total_validated = int(hyp["Temporal_Validated"].sum())

metrics = []

for k in K_VALUES:
    topk = hyp.head(k)
    hits = int(topk["Temporal_Validated"].sum())
    precision = hits / k
    recall = hits / total_validated if total_validated > 0 else 0

    metrics.append({
        "K": k,
        "Hits@K": hits,
        "Precision@K": precision,
        "Recall@K": recall
    })

metrics_df = pd.DataFrame(metrics)

validated = hyp[hyp["Temporal_Validated"] == 1].copy()

validated.to_csv(
    VALIDATED_OUT,
    index=False,
    encoding="utf-8-sig"
)

metrics_df.to_csv(
    METRICS_OUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== STRICT TEMPORAL VALIDATION ==========")
print("Ranked hypotheses tested:", len(hyp))
print("Future validated hypotheses:", total_validated)

print("\nMetrics:")
print(metrics_df.to_string(index=False))

print("\nSaved validated hypotheses:")
print(VALIDATED_OUT)

print("\nSaved metrics:")
print(METRICS_OUT)

print("\nTop validated hypotheses:")
print(
    validated[
        [
            "Source",
            "Source_Type",
            "Target",
            "Target_Type",
            "Hypothesis_Class",
            "Node2Vec_Integrated_Score",
            "Bridge_Nodes"
        ]
    ].head(30).to_string(index=False)
)
