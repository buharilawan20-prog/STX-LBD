from pathlib import Path
from itertools import combinations
import pandas as pd


# =====================================================
# INPUT FILES
# =====================================================
DATASETS = {
    "cyano": {
        "input": "data/processed/entities_cyano_semantic_merged_normalized.csv",
        "outdir": "data/graphs_cyano",
    },
    "dino": {
        "input": "data/processed/entities_dino_semantic_merged_normalized.csv",
        "outdir": "data/graphs_dino",
    },
}


# =====================================================
# SETTINGS
# =====================================================
MIN_EDGE_WEIGHT = 2

BAD_ENTITY_TYPES = {
    "FINDING_SENTENCE",
}

LOW_VALUE_TERMS = {
    "this study",
    "have been",
    "has been",
    "these toxins",
    "toxic",
    "toxigenic",
    "non-toxic",
    "Toxic",
}


# =====================================================
# RELATION RULES
# =====================================================
def infer_relation(type1, type2):
    pair = {type1, type2}

    if "GENE" in pair and "TOXIN" in pair:
        return "GENE_ASSOCIATED_WITH_TOXIN"

    if "ENV_FACTOR" in pair and "TOXIN" in pair:
        return "ENV_FACTOR_ASSOCIATED_WITH_TOXIN"

    if "ENV_FACTOR" in pair and "GENE" in pair:
        return "ENV_FACTOR_ASSOCIATED_WITH_GENE"

    if "BIOLOGICAL_MECHANISM" in pair and "TOXIN" in pair:
        return "TOXIN_MECHANISM_RELATION"

    if "BIOLOGICAL_MECHANISM" in pair and "GENE" in pair:
        return "GENE_MECHANISM_RELATION"

    if "EVOLUTIONARY_PROCESS" in pair and "GENE" in pair:
        return "EVOLUTION_GENE_RELATION"

    if "REGULATION_EXPRESSION" in pair and "GENE" in pair:
        return "REGULATION_GENE_RELATION"

    if "BIOSYNTHETIC_SYSTEM" in pair and "GENE" in pair:
        return "BIOSYNTHETIC_GENE_RELATION"

    if "SPECIES" in pair and "TOXIN" in pair:
        return "SPECIES_ASSOCIATED_WITH_TOXIN"

    if "SPECIES" in pair and "GENE" in pair:
        return "SPECIES_ASSOCIATED_WITH_GENE"

    return "SEMANTIC_CO_OCCURS"


def clean_entity(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def edge_key(a, b):
    a = str(a).lower().strip()
    b = str(b).lower().strip()
    return tuple(sorted([a, b]))


# =====================================================
# BUILD EDGES
# =====================================================
def build_edges(df):
    required = ["paper_id", "year", "entity_normalized", "entity_type"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()

    df["entity_normalized"] = df["entity_normalized"].apply(clean_entity)
    df["entity_type"] = df["entity_type"].astype(str)

    df = df[df["entity_normalized"] != ""]
    df = df[~df["entity_type"].isin(BAD_ENTITY_TYPES)]
    df = df[~df["entity_normalized"].str.lower().isin({x.lower() for x in LOW_VALUE_TERMS})]

    all_edges = []

    for paper_id, pdf in df.groupby("paper_id"):
        year = pdf["year"].dropna()
        year = int(year.iloc[0]) if len(year) > 0 else None

        # remove duplicate entity/type per paper
        ents = (
            pdf[["entity_normalized", "entity_type"]]
            .drop_duplicates()
            .values
            .tolist()
        )

        if len(ents) < 2:
            continue

        for (e1, t1), (e2, t2) in combinations(ents, 2):
            if e1 == e2:
                continue

            k = edge_key(e1, e2)
            relation = infer_relation(t1, t2)

            all_edges.append({
                "Source": k[0],
                "Target": k[1],
                "Source_Raw": e1,
                "Target_Raw": e2,
                "Source_Type": t1,
                "Target_Type": t2,
                "Relation": relation,
                "Paper_ID": paper_id,
                "Year": year,
            })

    edges = pd.DataFrame(all_edges)

    if edges.empty:
        return edges

    grouped = (
        edges.groupby(
            ["Source", "Target", "Relation", "Source_Type", "Target_Type"],
            dropna=False
        )
        .agg(
            Weight=("Paper_ID", "count"),
            Paper_Count=("Paper_ID", "nunique"),
            Paper_IDs=("Paper_ID", lambda x: ";".join(map(str, sorted(set(x))))),
            Years=("Year", lambda x: ";".join(map(str, sorted(set([i for i in x if pd.notna(i)]))))),
        )
        .reset_index()
    )

    grouped["Edge_Key"] = grouped.apply(
        lambda r: str(edge_key(r["Source"], r["Target"])),
        axis=1
    )

    grouped = grouped.sort_values(
        ["Weight", "Paper_Count"],
        ascending=False
    )

    return grouped


# =====================================================
# MAIN
# =====================================================
def main():
    for name, cfg in DATASETS.items():
        infile = Path(cfg["input"])
        outdir = Path(cfg["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)

        if not infile.exists():
            print(f"Missing input for {name}: {infile}")
            continue

        print(f"\nBuilding normalized semantic edges for: {name}")
        print(f"Input: {infile}")

        df = pd.read_csv(infile, encoding="utf-8-sig", low_memory=False)

        edges = build_edges(df)

        all_out = outdir / f"{name}_semantic_edges_normalized_all.csv"
        filt_out = outdir / f"{name}_semantic_edges_normalized_filtered.csv"

        edges.to_csv(all_out, index=False, encoding="utf-8-sig")

        filtered = edges[edges["Weight"] >= MIN_EDGE_WEIGHT].copy()
        filtered.to_csv(filt_out, index=False, encoding="utf-8-sig")

        print(f"Saved all edges: {all_out}")
        print(f"Saved filtered edges: {filt_out}")
        print(f"All edges: {len(edges)}")
        print(f"Filtered edges: {len(filtered)}")

        print("\nTop edges:")
        show_cols = ["Source", "Target", "Relation", "Weight", "Paper_Count"]
        print(filtered[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
