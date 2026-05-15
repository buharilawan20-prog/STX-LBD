from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


PRIORITY_FILE = "results/semantic_hypotheses/STX_SEMANTIC_HYPOTHESES_PRIORITY_ALL.csv"
DISCOVERY_FILE = "results/semantic_hypotheses/STX_DISCOVERY_SHORTLIST_BY_CATEGORY.csv"
FUTURE_EDGE_FILE = "data/graphs_semantic/semantic_post2015_edges_filtered.csv"

OUTPUT_DIR = "results/final_ai_discoveries"
OUTPUT_FILE = "FINAL_AI_SCORED_DISCOVERY_SHORTLIST.csv"
ALL_SCORED_FILE = "ALL_SEMANTIC_HYPOTHESES_WITH_ML_SCORE.csv"
METRICS_FILE = "semantic_ml_scoring_metrics.csv"


FEATURES = [
    "Common_Neighbor_Count",
    "Weighted_Common_Neighbor_Score",
    "Adamic_Adar_Score",
    "Preferential_Attachment",
    "Source_Degree",
    "Target_Degree",
    "Source_Weighted_Degree",
    "Target_Weighted_Degree",
    "Semantic_Bonus",
    "Structural_Score",
    "Priority_Bonus",
    "Priority_Score",
    "Has_STX_Gene_Signal",
    "Has_Environment_Signal",
    "Has_Evolution_Signal",
    "Has_Regulation_Signal",
]


def edge_key(a, b):
    return tuple(sorted([str(a).strip().lower(), str(b).strip().lower()]))


def minmax(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    if s.max() == s.min():
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def add_future_labels(df, future):
    future_keys = set()

    for _, row in future.iterrows():
        future_keys.add(edge_key(row["Source"], row["Target"]))

    labels = []
    for _, row in df.iterrows():
        labels.append(1 if edge_key(row["Source"], row["Target"]) in future_keys else 0)

    df["Validated_in_Post2015"] = labels
    return df


def prepare_features(df):
    df = df.copy()

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    for col in [
        "Has_STX_Gene_Signal",
        "Has_Environment_Signal",
        "Has_Evolution_Signal",
        "Has_Regulation_Signal",
    ]:
        df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(df[col])
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

    return df


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    priority = pd.read_csv(PRIORITY_FILE, encoding="utf-8-sig")
    discovery = pd.read_csv(DISCOVERY_FILE, encoding="utf-8-sig")
    future = pd.read_csv(FUTURE_EDGE_FILE, encoding="utf-8-sig")

    priority = add_future_labels(priority, future)
    priority = prepare_features(priority)

    X = priority[FEATURES]
    y = priority["Validated_in_Post2015"].astype(int)

    print(f"Total semantic hypotheses: {len(priority)}")
    print(f"Validated future matches: {y.sum()}")
    print(f"Non-validated: {len(y) - y.sum()}")
    print()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gb_oof = np.zeros(len(priority))
    rf_oof = np.zeros(len(priority))

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        gb = GradientBoostingClassifier(random_state=42)
        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
        )

        gb.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        gb_oof[test_idx] = gb.predict_proba(X_test)[:, 1]
        rf_oof[test_idx] = rf.predict_proba(X_test)[:, 1]

    priority["GB_ML_Score_OOF"] = gb_oof
    priority["RF_ML_Score_OOF"] = rf_oof
    priority["ML_Score"] = (
        0.6 * priority["GB_ML_Score_OOF"] +
        0.4 * priority["RF_ML_Score_OOF"]
    )

    priority["Priority_Score_Norm"] = minmax(priority["Priority_Score"])
    priority["ML_Score_Norm"] = minmax(priority["ML_Score"])

    priority["Final_AI_Discovery_Score"] = (
        0.55 * priority["Priority_Score_Norm"] +
        0.45 * priority["ML_Score_Norm"]
    )

    metrics = pd.DataFrame([
        {
            "Model": "GradientBoosting_OOF",
            "ROC_AUC": roc_auc_score(y, priority["GB_ML_Score_OOF"]),
            "PR_AUC": average_precision_score(y, priority["GB_ML_Score_OOF"]),
        },
        {
            "Model": "RandomForest_OOF",
            "ROC_AUC": roc_auc_score(y, priority["RF_ML_Score_OOF"]),
            "PR_AUC": average_precision_score(y, priority["RF_ML_Score_OOF"]),
        },
        {
            "Model": "Combined_ML_OOF",
            "ROC_AUC": roc_auc_score(y, priority["ML_Score"]),
            "PR_AUC": average_precision_score(y, priority["ML_Score"]),
        },
    ])

    priority = priority.sort_values("Final_AI_Discovery_Score", ascending=False)
    priority.to_csv(outdir / ALL_SCORED_FILE, index=False, encoding="utf-8-sig")
    metrics.to_csv(outdir / METRICS_FILE, index=False, encoding="utf-8-sig")

    # Score only your discovery shortlist
    key_cols = ["Source", "Target", "Hypothesis_Type", "Biological_Category"]

    scored_discovery = discovery.merge(
        priority[
            key_cols + [
                "Validated_in_Post2015",
                "ML_Score",
                "Priority_Score_Norm",
                "ML_Score_Norm",
                "Final_AI_Discovery_Score",
            ]
        ],
        on=key_cols,
        how="left",
    )

    scored_discovery = scored_discovery.sort_values(
        "Final_AI_Discovery_Score",
        ascending=False,
    )

    scored_discovery.to_csv(outdir / OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("ML scoring metrics:")
    print(metrics.to_string(index=False))
    print()
    print(f"Saved all scored hypotheses: {outdir / ALL_SCORED_FILE}")
    print(f"Saved final scored discovery shortlist: {outdir / OUTPUT_FILE}")
    print()
    print("Top final AI-scored discoveries:")
    cols = [
        "Source",
        "Target",
        "Hypothesis_Type",
        "Biological_Category",
        "Priority_Score",
        "ML_Score",
        "Final_AI_Discovery_Score",
        "Validated_in_Post2015",
        "Bridge_Nodes",
    ]
    print(scored_discovery[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
