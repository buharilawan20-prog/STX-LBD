from __future__ import annotations

from pathlib import Path
import pandas as pd


HYPOTHESIS_FILE = "results/hypotheses/STX_BIOLOGY_HYPOTHESES_with_embeddings.csv"
TEST_EDGE_FILE = "data/graphs/dino_post2015_edges.csv"

OUTPUT_DIR = "results/temporal_validation"
OUTPUT_FILE = "temporal_validation_results.csv"
METRICS_FILE = "temporal_validation_metrics.csv"


def make_edge_key(source: str, target: str) -> tuple[str, str]:
    source = str(source).strip()
    target = str(target).strip()
    return tuple(sorted([source, target]))


def precision_at_k(df: pd.DataFrame, k: int) -> float:
    top_k = df.head(k)
    if len(top_k) == 0:
        return 0.0
    return top_k["Appears_in_Future"].mean()


def hits_at_k(df: pd.DataFrame, k: int) -> int:
    return int(df.head(k)["Appears_in_Future"].sum())


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    hyp_df = pd.read_csv(HYPOTHESIS_FILE, encoding="utf-8-sig")
    test_edges = pd.read_csv(TEST_EDGE_FILE, encoding="utf-8-sig")

    # Build lookup for future edges
    test_edge_map = {}

    for _, row in test_edges.iterrows():
        key = make_edge_key(row["Source"], row["Target"])
        test_edge_map[key] = {
            "Future_Relation": row.get("Relation", ""),
            "Future_Weight": row.get("Weight", 0),
            "Future_Paper_Count": row.get("Paper_Count", 0),
            "Future_Paper_IDs": row.get("Paper_IDs", ""),
            "Future_Years": row.get("Years", ""),
        }

    validation_rows = []

    # Validate each hypothesis
    for rank, (_, row) in enumerate(hyp_df.iterrows(), start=1):
        key = make_edge_key(row["Source"], row["Target"])
        match = test_edge_map.get(key)

        appears = match is not None

        validation_rows.append({
            "Rank": rank,
            "Source": row["Source"],
            "Target": row["Target"],
            "Source_Type": row["Source_Type"],
            "Target_Type": row["Target_Type"],
            "Hypothesis_Type": row["Hypothesis_Type"],
            "Structural_Score": row["Structural_Score"],
            "Embedding_Source_Target_Similarity": row["Embedding_Source_Target_Similarity"],
            "Embedding_Bridge_Mean_Similarity": row["Embedding_Bridge_Mean_Similarity"],
            "Embedding_Integrated_Score": row["Embedding_Integrated_Score"],
            "Bridge_Nodes": row["Bridge_Nodes"],
            "Appears_in_Future": 1 if appears else 0,
            "Future_Relation": match["Future_Relation"] if appears else "",
            "Future_Weight": match["Future_Weight"] if appears else 0,
            "Future_Paper_Count": match["Future_Paper_Count"] if appears else 0,
            "Future_Paper_IDs": match["Future_Paper_IDs"] if appears else "",
            "Future_Years": match["Future_Years"] if appears else "",
        })

    result_df = pd.DataFrame(validation_rows)

    # Save detailed results
    output_path = output_dir / OUTPUT_FILE
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Metrics
    metrics = []

    for k in [10, 20, 50, 100, 200]:
        if k <= len(result_df):
            metrics.append({
                "Metric": f"Hits@{k}",
                "Value": hits_at_k(result_df, k),
            })
            metrics.append({
                "Metric": f"Precision@{k}",
                "Value": precision_at_k(result_df, k),
            })

    total_hits = int(result_df["Appears_in_Future"].sum())
    total_predictions = len(result_df)
    overall_precision = total_hits / total_predictions if total_predictions else 0

    metrics.append({"Metric": "Total_Predictions", "Value": total_predictions})
    metrics.append({"Metric": "Total_Future_Matches", "Value": total_hits})
    metrics.append({"Metric": "Overall_Precision", "Value": overall_precision})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / METRICS_FILE, index=False, encoding="utf-8-sig")

    # Print results
    print(f"Saved temporal validation results: {output_path}")
    print(f"Saved metrics: {output_dir / METRICS_FILE}")
    print()
    print("Temporal validation metrics:")
    print(metrics_df.to_string(index=False))
    print()

    print("Top validated hypotheses:")
    validated = result_df[result_df["Appears_in_Future"] == 1]

    preview_cols = [
        "Rank",
        "Source",
        "Target",
        "Hypothesis_Type",
        "Embedding_Integrated_Score",
        "Future_Paper_Count",
        "Future_Years",
    ]

    if len(validated) > 0:
        print(validated[preview_cols].head(30).to_string(index=False))
    else:
        print("No validated hypotheses found.")


if __name__ == "__main__":
    main()
