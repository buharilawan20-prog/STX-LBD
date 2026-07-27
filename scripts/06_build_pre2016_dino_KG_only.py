import pandas as pd
from itertools import combinations
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INPUT_FILE = BASE / "FINAL_WORKSPACE/processed/dino_pre2016_entities.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/kg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "dino_pre2016_semantic_edges.csv"

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(INPUT_FILE).fillna("")

print("\nInput file:")
print(INPUT_FILE)

print("\nColumns:")
print(df.columns.tolist())

# ==========================================================
# AUTO-DETECT COLUMNS
# ==========================================================

possible_doc_cols = [
    "document_id", "doc_id", "pmid", "doi", "id", "record_id"
]

possible_entity_cols = [
    "entity", "entity_normalized", "term", "normalized_entity"
]

possible_type_cols = [
    "entity_type", "type", "semantic_type", "category"
]

doc_col = next((c for c in possible_doc_cols if c in df.columns), None)
entity_col = next((c for c in possible_entity_cols if c in df.columns), None)
type_col = next((c for c in possible_type_cols if c in df.columns), None)

if doc_col is None:
    raise ValueError(
        "No document ID column found. Expected one of: "
        + ", ".join(possible_doc_cols)
    )

if entity_col is None:
    raise ValueError(
        "No entity column found. Expected one of: "
        + ", ".join(possible_entity_cols)
    )

if type_col is None:
    raise ValueError(
        "No entity type column found. Expected one of: "
        + ", ".join(possible_type_cols)
    )

print("\nDetected columns:")
print("Document column:", doc_col)
print("Entity column:", entity_col)
print("Type column:", type_col)

# ==========================================================
# CLEAN DATA
# ==========================================================

df = df[[doc_col, entity_col, type_col]].copy()

df.columns = ["document_id", "entity", "entity_type"]

df["document_id"] = df["document_id"].astype(str).str.strip()
df["entity"] = df["entity"].astype(str).str.strip().str.lower()
df["entity_type"] = df["entity_type"].astype(str).str.strip()

df = df[
    (df["document_id"] != "") &
    (df["entity"] != "") &
    (df["entity_type"] != "")
].copy()

# ==========================================================
# NORMALIZE COMMON BROAD TERMS
# ==========================================================

NORMALIZATION_MAP = {
    "dinoflagellates": "dinoflagellate",
    "dinoflagellate": "dinoflagellate",

    "cyanobacteria": "cyanobacteria",
    "cyanobacterial": "cyanobacteria",
    "cyanobacterium": "cyanobacteria",

    "pst": "paralytic_shellfish_toxins",
    "psts": "paralytic_shellfish_toxins",
    "paralytic shellfish toxin": "paralytic_shellfish_toxins",
    "paralytic shellfish toxins": "paralytic_shellfish_toxins",

    "psp": "paralytic_shellfish_poisoning",
    "paralytic shellfish poisoning": "paralytic_shellfish_poisoning",

    "sxta1": "sxta",
    "sxta4": "sxta",
    "sxt-a": "sxta",
    "sxta": "sxta",

    "sxtg": "sxtg",
    "sxtd": "sxtd",
    "sxti": "sxti",

    "hplc-fld": "hplc",
    "lc-ms": "lc_ms",
    "lc-ms/ms": "lc_ms_ms",
    "mass spectrometry": "mass_spectrometry",
    "mouse bioassay": "mouse_bioassay"
}

df["entity"] = df["entity"].replace(NORMALIZATION_MAP)

def normalize_type(entity, old_type):
    if entity == "dinoflagellate":
        return "DINO_TAXON"
    if entity == "cyanobacteria":
        return "CYANO_TAXON"
    if entity in {"sxta", "sxtg", "sxtd", "sxti"}:
        return "SXT_GENE"
    return old_type

df["entity_type"] = df.apply(
    lambda r: normalize_type(r["entity"], r["entity_type"]),
    axis=1
)

# ==========================================================
# RELATION RULES
# ==========================================================

def relation_type(t1, t2):
    types = {t1, t2}

    if "DINO_TAXON" in types and "TOXIN" in types:
        return "taxon_associated_with_toxin"

    if "DINO_TAXON" in types and "SXT_GENE" in types:
        return "taxon_associated_with_gene"

    if "SXT_GENE" in types and "TOXIN" in types:
        return "gene_associated_with_toxin"

    if "SXT_GENE" in types and "BIOLOGICAL_PROCESS" in types:
        return "gene_associated_with_process"

    if "ENV_FACTOR" in types and "TOXIN" in types:
        return "environment_associated_with_toxin"

    if "ENV_FACTOR" in types and "BIOLOGICAL_PROCESS" in types:
        return "environment_associated_with_process"

    if "BIOLOGICAL_PROCESS" in types and "TOXIN" in types:
        return "process_associated_with_toxin"

    if "DETECTION_METHOD" in types and "TOXIN" in types:
        return "method_associated_with_toxin"

    return "semantic_cooccurrence"

# ==========================================================
# BUILD EDGES
# ==========================================================

print("\nDocuments:", df["document_id"].nunique())
print("Entity mentions:", len(df))

edges = []

for doc_id, group in df.groupby("document_id"):

    ents = (
        group[["entity", "entity_type"]]
        .drop_duplicates()
        .values
        .tolist()
    )

    if len(ents) < 2:
        continue

    for (e1, t1), (e2, t2) in combinations(ents, 2):

        if not e1 or not e2 or e1 == e2:
            continue

        # stable undirected ordering
        if e1 > e2:
            e1, e2 = e2, e1
            t1, t2 = t2, t1

        edges.append({
            "dataset": "dino_pre2016",
            "source": e1,
            "source_type": t1,
            "target": e2,
            "target_type": t2,
            "relation": relation_type(t1, t2),
            "document_id": doc_id
        })

edge_df = pd.DataFrame(edges)

if edge_df.empty:
    raise ValueError("No edges generated. Check that the split file contains entity-level rows.")

# ==========================================================
# AGGREGATE EDGES
# ==========================================================

edge_df = (
    edge_df
    .groupby(
        [
            "dataset",
            "source",
            "source_type",
            "target",
            "target_type",
            "relation"
        ],
        as_index=False
    )
    .agg(
        weight=("document_id", "nunique"),
        support_documents=(
            "document_id",
            lambda x: ";".join(sorted(set(map(str, x))))
        )
    )
)

edge_df = edge_df.sort_values(
    "weight",
    ascending=False
)

# ==========================================================
# SAVE
# ==========================================================

edge_df.to_csv(
    OUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print(OUT_FILE)

print("\nKG summary:")
print("Nodes:", len(set(edge_df["source"]) | set(edge_df["target"])))
print("Edges:", len(edge_df))

print("\nTop edges:")
print(edge_df.head(20).to_string(index=False))
