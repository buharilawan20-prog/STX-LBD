from __future__ import annotations

from itertools import combinations
from pathlib import Path
import pandas as pd


INPUT_FILE = "data/processed/entities_semantic_merged.csv"
OUTPUT_DIR = "data/graphs_semantic"

ALL_EDGES = "semantic_all_edges.csv"
TRAIN_EDGES = "semantic_pre2016_edges.csv"
TEST_EDGES = "semantic_post2015_edges.csv"


GENERIC_ENTITY_TYPES = {
    "FINDING_SENTENCE"
}

LOW_VALUE_TERMS = {
    "toxic",
    "dinoflagellate",
    "dinoflagellates",
    "marine dinoflagellate",
    "marine dinoflagellates",
}


def infer_relation(type1: str, type2: str) -> str:
    pair = {type1, type2}

    if "GENE" in pair and "TOXIN" in pair:
        return "GENE_ASSOCIATED_WITH_TOXIN"

    if "GENE" in pair and "SPECIES" in pair:
        return "GENE_PRESENT_IN_SPECIES"

    if "ENV_FACTOR" in pair and "TOXIN" in pair:
        return "ENV_FACTOR_ASSOCIATED_WITH_TOXIN"

    if "REGULATION_EXPRESSION" in pair and "GENE" in pair:
        return "REGULATION_ASSOCIATED_WITH_GENE"

    if "GENE_PRESENCE_ABSENCE" in pair and "GENE" in pair:
        return "GENE_PRESENCE_ABSENCE_EVENT"

    if "EVOLUTIONARY_PROCESS" in pair and "GENE" in pair:
        return "GENE_EVOLUTION_RELATION"

    if "EVOLUTIONARY_PROCESS" in pair and "SPECIES" in pair:
        return "SPECIES_EVOLUTION_RELATION"

    if "BIOLOGICAL_MECHANISM" in pair and "GENE" in pair:
        return "GENE_MECHANISM_RELATION"

    if "BIOLOGICAL_MECHANISM" in pair and "TOXIN" in pair:
        return "TOXIN_MECHANISM_RELATION"

    if "TOXIN_PHENOTYPE" in pair and "SPECIES" in pair:
        return "SPECIES_TOXIN_PHENOTYPE_RELATION"

    if "BIOSYNTHETIC_SYSTEM" in pair and "GENE" in pair:
        return "BIOSYNTHETIC_SYSTEM_GENE_RELATION"

    if "GENE_DOMAIN" in pair and "GENE" in pair:
        return "GENE_DOMAIN_RELATION"

    if any(t.startswith("PHRASE_") for t in pair):
        return "SEMANTIC_PHRASE_ASSOCIATION"

    return "SEMANTIC_CO_OCCURS"


def clean_entity_name(x: object) -> str:
    return str(x).strip()


def build_edges(df: pd.DataFrame) -> pd.DataFrame:
    edge_rows = []

    for paper_id, pdf in df.groupby("paper_id"):
        year = int(pdf["year"].iloc[0])
        group = str(pdf["group"].iloc[0])

        # Use finding sentences only as relation evidence, not as normal graph nodes
        finding_relations = (
            pdf[pdf["entity_type"] == "FINDING_SENTENCE"]["relation_hint"]
            .dropna()
            .astype(str)
            .tolist()
        )

        relation_hint = ""
        if finding_relations:
            relation_hint = ";".join(sorted(set([r for r in finding_relations if r])))

        entities = pdf[pdf["entity_type"] != "FINDING_SENTENCE"].copy()

        entities["entity_normalized"] = entities["entity_normalized"].apply(clean_entity_name)

        entities = entities[
            ~entities["entity_normalized"].str.lower().isin(LOW_VALUE_TERMS)
        ]

        entities = entities[
            entities["entity_normalized"].str.len() > 1
        ]

        entities = (
            entities[["entity_normalized", "entity_type"]]
            .drop_duplicates()
            .sort_values(["entity_type", "entity_normalized"])
        )

        if len(entities) < 2:
            continue

        entity_rows = list(entities.itertuples(index=False, name=None))

        for (src, src_type), (tgt, tgt_type) in combinations(entity_rows, 2):
            if src == tgt:
                continue

            if src.lower() > tgt.lower():
                src, tgt = tgt, src
                src_type, tgt_type = tgt_type, src_type

            relation = infer_relation(src_type, tgt_type)

            edge_rows.append({
                "Source": src,
                "Target": tgt,
                "Source_Type": src_type,
                "Target_Type": tgt_type,
                "Relation": relation,
                "Relation_Hint": relation_hint,
                "Weight": 1,
                "Paper_ID": paper_id,
                "Year": year,
                "Group": group,
            })

    edges = pd.DataFrame(edge_rows)

    if edges.empty:
        return edges

    grouped = (
        edges.groupby(
            [
                "Source",
                "Target",
                "Source_Type",
                "Target_Type",
                "Relation",
                "Group",
            ],
            as_index=False,
        )
        .agg(
            Weight=("Weight", "sum"),
            Paper_Count=("Paper_ID", "nunique"),
            Paper_IDs=("Paper_ID", lambda x: ";".join(sorted(set(map(str, x))))),
            Years=("Year", lambda x: ";".join(map(str, sorted(set(map(int, x)))))),
            Relation_Hints=("Relation_Hint", lambda x: ";".join(sorted(set([str(v) for v in x if str(v).strip()])))),
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
        "entity_normalized",
        "entity_type",
        "relation_hint",
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    edges = build_edges(df)

    if edges.empty:
        print("No semantic edges generated.")
        return

    train = edges[edges["Group"] == "train"].copy()
    test = edges[edges["Group"] == "test"].copy()

    edges.to_csv(output_dir / ALL_EDGES, index=False, encoding="utf-8-sig")
    train.to_csv(output_dir / TRAIN_EDGES, index=False, encoding="utf-8-sig")
    test.to_csv(output_dir / TEST_EDGES, index=False, encoding="utf-8-sig")

    print(f"Saved all semantic edges: {output_dir / ALL_EDGES}")
    print(f"Saved train semantic edges: {output_dir / TRAIN_EDGES}")
    print(f"Saved test semantic edges: {output_dir / TEST_EDGES}")
    print()
    print(f"All semantic edges: {len(edges)}")
    print(f"Train semantic edges: {len(train)}")
    print(f"Test semantic edges: {len(test)}")
    print()
    print("Top train semantic edges:")
    cols = [
        "Source",
        "Target",
        "Source_Type",
        "Target_Type",
        "Relation",
        "Weight",
        "Paper_Count",
        "Relation_Hints",
    ]
    print(train[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
