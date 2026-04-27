from __future__ import annotations

from itertools import combinations
from pathlib import Path
import pandas as pd


INPUT_FILE = "data/processed/entities_normalized.csv"
OUTPUT_DIR = "data/graphs"

ALL_EDGES_FILE = "all_edges.csv"
TRAIN_EDGES_FILE = "dino_pre2016_edges.csv"
TEST_EDGES_FILE = "dino_post2015_edges.csv"


def infer_relation(source_type: str, target_type: str) -> str:
    pair = {source_type, target_type}

    if pair == {"GENE", "TOXIN"}:
        return "GENE_ASSOCIATED_WITH_TOXIN"
    if pair == {"GENE", "SPECIES"}:
        return "GENE_PRESENT_IN_SPECIES"
    if pair == {"SPECIES", "TOXIN"}:
        return "SPECIES_ASSOCIATED_WITH_TOXIN"
    if pair == {"ENV_FACTOR", "TOXIN"}:
        return "ENV_FACTOR_ASSOCIATED_WITH_TOXIN"
    if pair == {"ENV_FACTOR", "GENE"}:
        return "ENV_FACTOR_ASSOCIATED_WITH_GENE"
    if pair == {"PROCESS", "GENE"}:
        return "PROCESS_ASSOCIATED_WITH_GENE"
    if pair == {"PROCESS", "TOXIN"}:
        return "PROCESS_ASSOCIATED_WITH_TOXIN"
    if pair == {"PROCESS", "SPECIES"}:
        return "PROCESS_ASSOCIATED_WITH_SPECIES"

    return "CO_OCCURS"


def build_edges(df: pd.DataFrame) -> pd.DataFrame:
    edge_records = []

    for paper_id, pdf in df.groupby("paper_id"):
        year = int(pdf["year"].iloc[0])
        group = str(pdf["group"].iloc[0])

        entities = (
            pdf[["entity_normalized", "entity_type"]]
            .drop_duplicates()
            .sort_values(["entity_type", "entity_normalized"])
        )

        if len(entities) < 2:
            continue

        entity_rows = list(entities.itertuples(index=False, name=None))

        for (src, src_type), (tgt, tgt_type) in combinations(entity_rows, 2):
            if src == tgt:
                continue

            # Make undirected edge canonical by sorting names
            if src.lower() > tgt.lower():
                src, tgt = tgt, src
                src_type, tgt_type = tgt_type, src_type

            relation = infer_relation(src_type, tgt_type)

            edge_records.append({
                "Source": src,
                "Target": tgt,
                "Source_Type": src_type,
                "Target_Type": tgt_type,
                "Relation": relation,
                "Weight": 1,
                "Paper_ID": paper_id,
                "Year": year,
                "Group": group,
            })

    edges = pd.DataFrame(edge_records)

    if edges.empty:
        return edges

    grouped = (
        edges.groupby(
            ["Source", "Target", "Source_Type", "Target_Type", "Relation", "Group"],
            as_index=False
        )
        .agg(
            Weight=("Weight", "sum"),
            Paper_Count=("Paper_ID", "nunique"),
            Paper_IDs=("Paper_ID", lambda x: ";".join(sorted(set(map(str, x))))),
            Years=("Year", lambda x: ";".join(map(str, sorted(set(map(int, x)))))),
        )
    )

    grouped = grouped.sort_values(["Group", "Weight"], ascending=[True, False])

    return grouped


def main():
    input_path = Path(INPUT_FILE)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    required = [
        "paper_id",
        "year",
        "group",
        "entity_type",
        "entity_normalized",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_edges = build_edges(df)

    if all_edges.empty:
        print("No edges generated.")
        return

    train_edges = all_edges[all_edges["Group"] == "train"].copy()
    test_edges = all_edges[all_edges["Group"] == "test"].copy()

    all_edges.to_csv(output_dir / ALL_EDGES_FILE, index=False, encoding="utf-8-sig")
    train_edges.to_csv(output_dir / TRAIN_EDGES_FILE, index=False, encoding="utf-8-sig")
    test_edges.to_csv(output_dir / TEST_EDGES_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved all edges: {output_dir / ALL_EDGES_FILE}")
    print(f"Saved train edges: {output_dir / TRAIN_EDGES_FILE}")
    print(f"Saved test edges: {output_dir / TEST_EDGES_FILE}")
    print()
    print(f"All edges: {len(all_edges)}")
    print(f"Train edges: {len(train_edges)}")
    print(f"Test edges: {len(test_edges)}")
    print()
    print("Top train edges:")
    print(train_edges.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
