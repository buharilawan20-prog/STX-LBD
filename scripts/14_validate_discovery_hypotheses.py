from pathlib import Path
import pandas as pd


DISCOVERY_FILE = "results/semantic_hypotheses/STX_DISCOVERY_SHORTLIST_BY_CATEGORY.csv"
FUTURE_EDGE_FILE = "data/graphs_semantic/semantic_post2015_edges_filtered.csv"

OUTPUT_DIR = "results/discovery_validation"
OUTPUT_FILE = "validated_discovery_hypotheses.csv"
SUMMARY_FILE = "discovery_validation_summary.csv"


def edge_key(a, b):
    return tuple(sorted([str(a).strip().lower(), str(b).strip().lower()]))


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    discovery = pd.read_csv(DISCOVERY_FILE, encoding="utf-8-sig")
    future = pd.read_csv(FUTURE_EDGE_FILE, encoding="utf-8-sig")

    future_map = {}

    for _, row in future.iterrows():
        key = edge_key(row["Source"], row["Target"])
        future_map[key] = {
            "Future_Relation": row.get("Relation", ""),
            "Future_Weight": row.get("Weight", 0),
            "Future_Paper_Count": row.get("Paper_Count", 0),
            "Future_Paper_IDs": row.get("Paper_IDs", ""),
            "Future_Years": row.get("Years", ""),
            "Future_Relation_Hints": row.get("Relation_Hints", ""),
        }

    rows = []

    for _, row in discovery.iterrows():
        key = edge_key(row["Source"], row["Target"])
        match = future_map.get(key)

        validated = match is not None

        rows.append({
            **row.to_dict(),
            "Validated_in_Post2015": 1 if validated else 0,
            "Validation_Status": "Confirmed in post-2015 literature" if validated else "Not yet observed in post-2015 edge set",
            "Future_Relation": match["Future_Relation"] if validated else "",
            "Future_Weight": match["Future_Weight"] if validated else 0,
            "Future_Paper_Count": match["Future_Paper_Count"] if validated else 0,
            "Future_Paper_IDs": match["Future_Paper_IDs"] if validated else "",
            "Future_Years": match["Future_Years"] if validated else "",
            "Future_Relation_Hints": match["Future_Relation_Hints"] if validated else "",
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["Validated_in_Post2015", "Priority_Score"],
        ascending=[False, False],
    )

    result.to_csv(outdir / OUTPUT_FILE, index=False, encoding="utf-8-sig")

    summary = (
        result.groupby("Biological_Category")
        .agg(
            Total_Hypotheses=("Source", "count"),
            Validated=("Validated_in_Post2015", "sum"),
            Validation_Rate=("Validated_in_Post2015", "mean"),
            Mean_Priority_Score=("Priority_Score", "mean"),
        )
        .reset_index()
        .sort_values("Validation_Rate", ascending=False)
    )

    summary.to_csv(outdir / SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved discovery validation: {outdir / OUTPUT_FILE}")
    print(f"Saved summary: {outdir / SUMMARY_FILE}")
    print()
    print("Discovery validation summary:")
    print(summary.to_string(index=False))
    print()
    print("Top validated discovery hypotheses:")
    cols = [
        "Source",
        "Target",
        "Hypothesis_Type",
        "Biological_Category",
        "Priority_Score",
        "Validated_in_Post2015",
        "Future_Paper_Count",
        "Future_Years",
        "Future_Relation_Hints",
    ]
    validated = result[result["Validated_in_Post2015"] == 1]
    print(validated[cols].head(30).to_string(index=False))
    print()
    print("Top not-yet-observed hypotheses:")
    novel = result[result["Validated_in_Post2015"] == 0]
    print(novel[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
