import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

HYP_FILE = BASE / "FINAL_WORKSPACE/processed/dino_pre2016_priority_hypotheses_node2vec_scored.csv"
FUTURE_EDGE_FILE = BASE / "FINAL_WORKSPACE/kg/dino_post2015_semantic_edges.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUT_DIR / "dino_pre2016_hypotheses_ai_ranked.csv"
SUMMARY_OUT = OUT_DIR / "ai_ranker_training_summary.csv"

# ===============================
# LOAD
# ===============================

hyp = pd.read_csv(HYP_FILE).fillna("")
future_edges = pd.read_csv(FUTURE_EDGE_FILE).fillna("")

# ===============================
# NORMALIZE PAIR KEYS
# ===============================

def pair_key(a, b):
    a = str(a).strip()
    b = str(b).strip()
    return "||".join(sorted([a, b]))

future_edges["pair_key"] = future_edges.apply(
    lambda r: pair_key(r["source"], r["target"]),
    axis=1
)

future_pairs = set(future_edges["pair_key"])

hyp["pair_key"] = hyp.apply(
    lambda r: pair_key(r["Source"], r["Target"]),
    axis=1
)

# Positive if candidate pair appeared after 2015
hyp["Temporal_Label"] = hyp["pair_key"].apply(
    lambda x: 1 if x in future_pairs else 0
)

# ===============================
# FEATURE COLUMNS
# ===============================

FEATURES = [
    "Score",
    "Bridge_Score",
    "Common_Neighbors",
    "Distinct_Bridge_Types",
    "Adamic_Adar",
    "Jaccard",
    "Preferential_Attachment",
    "Degree_Source",
    "Degree_Target",
    "Embedding_Source_Target_Similarity",
    "Embedding_Bridge_Mean_Similarity",
    "Embedding_Bridge_Max_Similarity",
    "Node2Vec_Integrated_Score"
]

for col in FEATURES:
    if col not in hyp.columns:
        hyp[col] = 0

    hyp[col] = pd.to_numeric(
        hyp[col],
        errors="coerce"
    ).fillna(0)

X = hyp[FEATURES]
y = hyp["Temporal_Label"]

print("\n========== AI RANKER TRAINING ==========")
print("Total hypotheses:", len(hyp))
print("Future positives:", int(y.sum()))
print("Future negatives:", int((y == 0).sum()))

if y.sum() < 2:
    print("\nWARNING: Too few positive temporal labels for robust ML training.")
    print("The script will still score using Node2Vec_Integrated_Score only.")
    hyp["ML_Probability"] = 0
    hyp["Final_AI_Rank_Score"] = hyp["Node2Vec_Integrated_Score"]

else:

    stratify_y = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=stratify_y
    )

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            max_depth=5
        ),

        "GradientBoosting": GradientBoostingClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3
        )
    }

    summaries = []
    best_model = None
    best_name = None
    best_ap = -1

    for name, model in models.items():

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = model.decision_function(X_test)

        pred = (prob >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_test, prob)
        except Exception:
            auc = np.nan

        try:
            ap = average_precision_score(y_test, prob)
        except Exception:
            ap = np.nan

        summaries.append({
            "model": name,
            "roc_auc": auc,
            "average_precision": ap,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_positives": int(y_train.sum()),
            "test_positives": int(y_test.sum())
        })

        print("\nModel:", name)
        print("ROC-AUC:", auc)
        print("Average precision:", ap)
        print(classification_report(y_test, pred, zero_division=0))

        if not np.isnan(ap) and ap > best_ap:
            best_ap = ap
            best_model = model
            best_name = name

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    print("\nBest model:", best_name)
    print("Best average precision:", best_ap)

    # ===============================
    # SCORE ALL HYPOTHESES
    # ===============================

    hyp["ML_Probability"] = best_model.predict_proba(X)[:, 1]

    # Final AI score combines ML probability and Node2Vec score
    hyp["Final_AI_Rank_Score"] = (
        hyp["ML_Probability"] * 0.60 +
        hyp["Node2Vec_Integrated_Score"] * 0.40
    )

# ===============================
# SORT AND SAVE
# ===============================

hyp = hyp.sort_values(
    by="Final_AI_Rank_Score",
    ascending=False
)

hyp.to_csv(
    OUTFILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved AI-ranked hypotheses:")
print(OUTFILE)

print("\nSaved training summary:")
print(SUMMARY_OUT)

print("\nTemporal label distribution:")
print(hyp["Temporal_Label"].value_counts())

print("\nTop AI-ranked hypotheses:")
print(
    hyp[
        [
            "Source",
            "Source_Type",
            "Target",
            "Target_Type",
            "Hypothesis_Class",
            "Temporal_Label",
            "ML_Probability",
            "Node2Vec_Integrated_Score",
            "Final_AI_Rank_Score",
            "Bridge_Nodes"
        ]
    ].head(30).to_string(index=False)
)
