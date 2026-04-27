from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


HYP_FILE = "results/hypotheses/STX_BIOLOGY_HYPOTHESES.csv"
EMB_FILE = "results/hypotheses/STX_BIOLOGY_HYPOTHESES_with_embeddings.csv"
VAL_FILE = "results/temporal_validation/temporal_validation_results.csv"

OUTPUT_DIR = "results/ml_strict_validation"


STRUCTURAL_FEATURES = [
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

EMBEDDING_FEATURES = [
    "Embedding_Source_Target_Similarity",
    "Embedding_Bridge_Mean_Similarity",
]

ALL_FEATURES = STRUCTURAL_FEATURES + EMBEDDING_FEATURES


def precision_at_k(df, score_col, k):
    top = df.sort_values(score_col, ascending=False).head(k)
    if len(top) == 0:
        return 0.0
    return top["Appears_in_Future"].mean()


def hits_at_k(df, score_col, k):
    top = df.sort_values(score_col, ascending=False).head(k)
    return int(top["Appears_in_Future"].sum())


def evaluate_model(name, model, X_train, X_test, y_train, y_test, test_df):
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)

    preds = (probs >= 0.5).astype(int)

    temp = test_df.copy()
    temp[f"{name}_Score"] = probs

    return {
        "Model": name,
        "Test_ROC_AUC": roc_auc_score(y_test, probs),
        "Test_PR_AUC": average_precision_score(y_test, probs),
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
        "Precision@10": precision_at_k(temp, f"{name}_Score", 10),
        "Precision@20": precision_at_k(temp, f"{name}_Score", 20),
        "Hits@10": hits_at_k(temp, f"{name}_Score", 10),
        "Hits@20": hits_at_k(temp, f"{name}_Score", 20),
    }, temp


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    hyp_df = pd.read_csv(HYP_FILE, encoding="utf-8-sig")
    emb_df = pd.read_csv(EMB_FILE, encoding="utf-8-sig")
    val_df = pd.read_csv(VAL_FILE, encoding="utf-8-sig")

    # Keep embedding columns from scored hypothesis file
    emb_keep = [
        "Source",
        "Target",
        "Embedding_Source_Target_Similarity",
        "Embedding_Bridge_Mean_Similarity",
        "Embedding_Integrated_Score",
    ]

    emb_df = emb_df[emb_keep]

    # Merge structural + embedding + validation label
    df = hyp_df.merge(
        emb_df,
        on=["Source", "Target"],
        how="inner"
    )

    df = df.merge(
        val_df[["Source", "Target", "Appears_in_Future"]],
        on=["Source", "Target"],
        how="inner"
    )

    df[ALL_FEATURES] = df[ALL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["Appears_in_Future"] = df["Appears_in_Future"].astype(int)

    print(f"Total labeled hypotheses: {len(df)}")
    print(f"Positive future matches: {df['Appears_in_Future'].sum()}")
    print(f"Negative/non-future: {len(df) - df['Appears_in_Future'].sum()}")
    print()

    # Strict holdout split
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["Appears_in_Future"],
    )

    y_train = train_df["Appears_in_Future"]
    y_test = test_df["Appears_in_Future"]

    feature_sets = {
        "Structure_Only": STRUCTURAL_FEATURES,
        "Embedding_Only": EMBEDDING_FEATURES,
        "Structure_Plus_Embedding": ALL_FEATURES,
    }

    model_builders = {
        "LogisticRegression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            random_state=42,
        ),
    }

    results = []
    reranked_outputs = []

    for feature_set_name, features in feature_sets.items():
        X_train = train_df[features]
        X_test = test_df[features]

        for model_name, builder in model_builders.items():
            full_name = f"{feature_set_name}_{model_name}"
            model = builder()

            metrics, scored_test = evaluate_model(
                full_name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                test_df,
            )

            metrics["Feature_Set"] = feature_set_name
            results.append(metrics)

            scored_test["Model"] = full_name
            reranked_outputs.append(scored_test)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["Test_PR_AUC", "Test_ROC_AUC"],
        ascending=False,
    )

    results_df.to_csv(outdir / "strict_ml_model_comparison.csv", index=False, encoding="utf-8-sig")

    reranked_df = pd.concat(reranked_outputs, ignore_index=True)
    reranked_df.to_csv(outdir / "strict_ml_scored_test_predictions.csv", index=False, encoding="utf-8-sig")

    print("Strict validation model comparison:")
    print(results_df.to_string(index=False))
    print()
    print(f"Saved results in: {outdir}")

    # Feature importance for best RF combined model
    best_rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
    )

    best_rf.fit(train_df[ALL_FEATURES], y_train)

    importance = pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Importance": best_rf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance.to_csv(outdir / "strict_random_forest_feature_importance.csv", index=False, encoding="utf-8-sig")

    print()
    print("Combined Random Forest feature importance:")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
