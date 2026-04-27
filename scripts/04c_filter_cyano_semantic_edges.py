from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/graphs_cyano")
OUTPUT_DIR = Path("data/graphs_cyano")

ALL_IN = INPUT_DIR / "cyano_semantic_all_edges.csv"
TRAIN_IN = INPUT_DIR / "cyano_semantic_edges.csv"
TEST_IN = INPUT_DIR / "cyano_semantic_edges.csv"

ALL_OUT = OUTPUT_DIR / "cyano_semantic_edges_filtered.csv"
TRAIN_OUT = OUTPUT_DIR / "cyano_semantic_edges_filtered.csv"
TEST_OUT = OUTPUT_DIR / "cyano_semantic_edges_filtered_unused.csv"

REMOVE_NODES = {
    "Alexandrium",
    "Gymnodinium",
    "Pyrodinium",
    "dinoflagellate",
    "dinoflagellates",
    "marine dinoflagellate",
    "marine dinoflagellates",
    "cyanobacteria",
    "cyanobacterial",
    "toxic",
    "non-toxic",
    "toxin",
    "toxins",
    "shellfish toxin",
    "shellfish toxins",
    "paralytic shellfish poisoning",
    "PSP",
}


REMOVE_EDGE_TYPE_PAIRS = {
    ("TOXIN", "TOXIN"),
    ("TAXON_GROUP", "TAXON_GROUP"),
    ("PHRASE_TAXON", "TAXON_GROUP"),
    ("PHRASE_TAXON", "PHRASE_TAXON"),
}


KEEP_TYPES = {
    "GENE",
    "GENE_DOMAIN",
    "GENE_PRESENCE_ABSENCE",
    "REGULATION_EXPRESSION",
    "EVOLUTIONARY_PROCESS",
    "BIOLOGICAL_MECHANISM",
    "BIOSYNTHETIC_SYSTEM",
    "ENV_FACTOR",
    "SPECIES",
    "TOXIN",
    "TOXIN_PHENOTYPE",
    "PHRASE_GENE_MECHANISM",
    "PHRASE_REGULATION",
    "PHRASE_EVOLUTION",
    "PHRASE_ENVIRONMENT",
    "PHRASE_BIOSYNTHESIS",
    "PHRASE_TOXIN_PHENOTYPE",
}


HIGH_VALUE_KEYWORDS = [
    "sxt",
    "saxitoxin",
    "biosynthesis",
    "gene",
    "expression",
    "regulation",
    "homolog",
    "presence",
    "absence",
    "loss",
    "duplication",
    "evolution",
    "phylogen",
    "divergence",
    "conserved",
    "acquisition",
    "hgt",
    "toxin production",
    "toxin content",
    "toxin profile",
    "temperature",
    "salinity",
    "nitrogen",
    "nitrate",
    "phosphate",
    "nutrient",
    "limitation",
    "stress",
]


def clean_hints(hints):
    if pd.isna(hints):
        return ""
    parts = []
    for h in str(hints).split(";"):
        h = h.strip()
        if h and h.lower() != "nan":
            parts.append(h)
    return ";".join(sorted(set(parts)))


def pair_key(a, b):
    return tuple(sorted([str(a), str(b)]))


def has_high_value_term(row):
    text = f"{row['Source']} {row['Target']} {row['Relation']}".lower()
    return any(k.lower() in text for k in HIGH_VALUE_KEYWORDS)


def filter_edges(df):
    df = df.copy()

    df["Source"] = df["Source"].astype(str).str.strip()
    df["Target"] = df["Target"].astype(str).str.strip()
    df["Source_Type"] = df["Source_Type"].astype(str).str.strip()
    df["Target_Type"] = df["Target_Type"].astype(str).str.strip()

    # Remove generic nodes
    remove_lower = {x.lower() for x in REMOVE_NODES}

    df = df[
        ~df["Source"].str.lower().isin(remove_lower)
        & ~df["Target"].str.lower().isin(remove_lower)
    ].copy()

    # Keep meaningful types
    df = df[
        df["Source_Type"].isin(KEEP_TYPES)
        & df["Target_Type"].isin(KEEP_TYPES)
    ].copy()

    # Remove low-value type pairs
    def remove_type_pair(row):
        p = pair_key(row["Source_Type"], row["Target_Type"])
        return p in {pair_key(a, b) for a, b in REMOVE_EDGE_TYPE_PAIRS}

    df = df[~df.apply(remove_type_pair, axis=1)].copy()

    # Keep high-value edges only
    df = df[df.apply(has_high_value_term, axis=1)].copy()

    # Clean relation hints
    if "Relation_Hints" in df.columns:
        df["Relation_Hints"] = df["Relation_Hints"].apply(clean_hints)

    # Remove exact duplicate edges after filtering
    group_cols = [
        "Source",
        "Target",
        "Source_Type",
        "Target_Type",
        "Relation",
        "Group",
    ]

    df = (
        df.groupby(group_cols, as_index=False)
        .agg(
            Weight=("Weight", "sum"),
            Paper_Count=("Paper_Count", "sum"),
            Paper_IDs=("Paper_IDs", lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";"))))),
            Years=("Years", lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";"))))),
            Relation_Hints=("Relation_Hints", lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";"))))),
        )
    )

    df["Relation_Hints"] = df["Relation_Hints"].apply(clean_hints)

    df = df.sort_values(["Group", "Weight"], ascending=[True, False])

    return df


def run_one(infile, outfile):
    df = pd.read_csv(infile, encoding="utf-8-sig")
    filtered = filter_edges(df)
    filtered.to_csv(outfile, index=False, encoding="utf-8-sig")

    print(f"{infile.name}")
    print(f"  before: {len(df)}")
    print(f"  after : {len(filtered)}")
    print(f"  saved : {outfile}")
    print()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_one(ALL_IN, ALL_OUT)
    run_one(TRAIN_IN, TRAIN_OUT)
    run_one(TEST_IN, TEST_OUT)

    train = pd.read_csv(TRAIN_OUT, encoding="utf-8-sig")

    print("Top filtered train semantic edges:")
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
