from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# 🔥 INPUT FILES (IMPORTANT FIX)
HYP_FILE = "results/hypotheses/STX_BIOLOGY_HYPOTHESES.csv"
VAL_FILE = "results/temporal_validation/temporal_validation_results.csv"

OUTPUT_DIR = "results/ml_ranker"


FEATURES = [
    "Structural_Score",
    "Common_Neighbor_Count",
    "Weighted_Common_Neighbor_Score",
    "Adamic_Adar_Score",
    "Preferential_Attachment",
    "Source_Degree",
    "Target_Degree",
    "Source_Weighted_Degree",
    "Target_Weighted_Degree",
]


def precision_at_k(df, score_col, k):
    return df.sort_values(score_col, ascending=False).head(k)["Appears_in_Future"].mean()


def hits_at_k(df, score_col, k):
    return int(df.sort_values(score_col, ascending=False).head(k)["Appears_in_Future"].sum())


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    hyp_df = pd.read_csv(HYP_FILE)
    val_df = pd.read_csv(VAL_FILE)

    # 🔥 Merge on Source + Target
    df = pd.merge(
        hyp_df,
        val_df[["Source", "Target", "Appears_in_Future"]],
        on=["Source", "Target"],
        how="inner"
    )

    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df[FEATURES]
    y = df["Appears_in_Future"]

    models = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42),
        "GB": GradientBoostingClassifier(random_state=42),
    }

    results = []

    for name, model in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        roc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
        pr = cross_val_score(model, X, y, cv=cv, scoring="average_precision").mean()

        model.fit(X, y)

        probs = model.predict_proba(X)[:, 1]

        df[f"{name}_Score"] = probs

        results.append({
            "Model": name,
            "ROC_AUC": roc,
            "PR_AUC": pr,
            "Precision@10": precision_at_k(df, f"{name}_Score", 10),
            "Precision@20": precision_at_k(df, f"{name}_Score", 20),
            "Hits@10": hits_at_k(df, f"{name}_Score", 10),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(outdir / "ml_results.csv", index=False)

    print("\nModel Results:\n")
    print(res_df.to_string(index=False))

    # Feature importance (RF)
    rf = models["RF"]
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nFeature Importance (Random Forest):\n")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
