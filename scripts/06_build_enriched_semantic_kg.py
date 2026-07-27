import pandas as pd
import itertools
from pathlib import Path

# ===============================
# PATHS
# ===============================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INFILE = BASE / "FINAL_WORKSPACE/processed/all_split_entities_combined_normalized.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/kg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# RELATION MAP
# ===============================

RELATION_MAP = {
    ("DINO_TAXON", "TOXIN"): "taxon_associated_with_toxin",
    ("CYANO_TAXON", "TOXIN"): "taxon_associated_with_toxin",

    ("SXT_GENE", "TOXIN"): "gene_associated_with_toxin",
    ("SXT_GENE", "BIOLOGICAL_PROCESS"): "gene_associated_with_process",

    ("ENV_FACTOR", "TOXIN"): "environment_associated_with_toxin",
    ("ENV_FACTOR", "BIOLOGICAL_PROCESS"): "environment_associated_with_process",

    ("BIOLOGICAL_PROCESS", "TOXIN"): "process_associated_with_toxin",

    ("DETECTION_METHOD", "TOXIN"): "method_associated_with_toxin",

    ("DINO_TAXON", "SXT_GENE"): "taxon_associated_with_gene",
    ("CYANO_TAXON", "SXT_GENE"): "taxon_associated_with_gene",

    ("DINO_TAXON", "BIOLOGICAL_PROCESS"): "taxon_associated_with_process",
    ("CYANO_TAXON", "BIOLOGICAL_PROCESS"): "taxon_associated_with_process",
}

# ===============================
# FUNCTIONS
# ===============================

def get_relation(type1, type2):
    if (type1, type2) in RELATION_MAP:
        return RELATION_MAP[(type1, type2)]

    if (type2, type1) in RELATION_MAP:
        return RELATION_MAP[(type2, type1)]

    return "semantic_cooccurrence"


def build_edges(df, label):

    edges = []

    grouped = df.groupby("document_id")

    for doc_id, group in grouped:

        group = group.drop_duplicates(["entity", "entity_type"])

        entities = group[
            ["entity", "entity_type"]
        ].dropna().drop_duplicates()

        if len(entities) < 2:
            continue

        year = group["year"].iloc[0] if "year" in group.columns else ""
        taxon_scope = group["taxon_scope"].iloc[0] if "taxon_scope" in group.columns else ""
        relevance_class = group["relevance_class"].iloc[0] if "relevance_class" in group.columns else ""

        for (_, row1), (_, row2) in itertools.combinations(
            entities.iterrows(),
            2
        ):

            source = str(row1["entity"]).strip()
            target = str(row2["entity"]).strip()

            source_type = str(row1["entity_type"]).strip()
            target_type = str(row2["entity_type"]).strip()

            if source == "" or target == "":
                continue

            if source == target:
                continue

            relation = get_relation(source_type, target_type)

            edges.append({
                "dataset": label,
                "document_id": doc_id,
                "year": year,
                "taxon_scope": taxon_scope,
                "relevance_class": relevance_class,
                "source": source,
                "source_type": source_type,
                "relation": relation,
                "target": target,
                "target_type": target_type,
                "weight": 1
            })

    edge_df = pd.DataFrame(edges)

    if len(edge_df) == 0:
        return edge_df

    # ===============================
    # CLEAN YEAR COLUMN
    # ===============================

    edge_df["year"] = pd.to_numeric(
        edge_df["year"],
        errors="coerce"
    )

    # ===============================
    # AGGREGATE EDGES
    # ===============================

    edge_df = edge_df.groupby(
        [
            "dataset",
            "source",
            "source_type",
            "relation",
            "target",
            "target_type"
        ],
        as_index=False
    ).agg(
        weight=("weight", "sum"),
        support_documents=(
            "document_id",
            lambda x: ";".join(sorted(set(map(str, x))))
        ),
        first_year=("year", "min"),
        last_year=("year", "max")
    )

    edge_df["first_year"] = edge_df["first_year"].fillna("")
    edge_df["last_year"] = edge_df["last_year"].fillna("")

    edge_df = edge_df.sort_values(
        by="weight",
        ascending=False
    )

    return edge_df


# ===============================
# LOAD ENTITIES
# ===============================

df = pd.read_csv(INFILE).fillna("")

required_cols = [
    "dataset",
    "document_id",
    "year",
    "taxon_scope",
    "relevance_class",
    "entity",
    "entity_type"
]

for col in required_cols:
    if col not in df.columns:
        df[col] = ""

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

df["entity"] = df["entity"].astype(str).str.strip()
df["entity_type"] = df["entity_type"].astype(str).str.strip()

df = df[
    (df["entity"] != "") &
    (df["entity_type"] != "")
].copy()

# ===============================
# BUILD DATASET-SPECIFIC KGs
# ===============================

all_edges = []

for label in sorted(df["dataset"].dropna().unique()):

    print("\n===================================")
    print("BUILDING KG:", label)
    print("===================================")

    sub = df[df["dataset"] == label].copy()

    edge_df = build_edges(sub, label)

    out_file = OUT_DIR / f"{label}_semantic_edges.csv"

    edge_df.to_csv(
        out_file,
        index=False,
        encoding="utf-8-sig"
    )

    all_edges.append(edge_df)

    print("Entity mentions:", len(sub))
    print("Edges:", len(edge_df))

    if len(edge_df) > 0:

        print("\nTop relations:")
        print(edge_df["relation"].value_counts().head(20))

        print("\nTop weighted edges:")
        print(
            edge_df[
                ["source", "relation", "target", "weight"]
            ].head(20).to_string(index=False)
        )

# ===============================
# COMBINED KG
# ===============================

if all_edges:

    combined = pd.concat(
        all_edges,
        ignore_index=True
    )

    combined_out = OUT_DIR / "combined_enriched_semantic_edges.csv"

    combined.to_csv(
        combined_out,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved combined KG:")
    print(combined_out)

    print("\nCombined edges:", len(combined))

else:
    print("\nNo edges were generated.")
