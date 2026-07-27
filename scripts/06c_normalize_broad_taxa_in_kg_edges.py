import pandas as pd
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

KG_DIR = BASE / "FINAL_WORKSPACE/kg"

INPUT_FILES = [
    "dino_pre2016_semantic_edges.csv",
    "dino_post2015_semantic_edges.csv",
    "cyano_all_semantic_edges.csv",
    "combined_enriched_semantic_edges.csv"
]

def normalize_entity(x):
    x = str(x).strip().lower()

    mapping = {
        # dinoflagellate broad terms
        "dinoflagellates": "dinoflagellate",
        "dinoflagellate": "dinoflagellate",

        # cyanobacteria broad terms
        "cyanobacteria": "cyanobacteria",
        "cyanobacterial": "cyanobacteria",
        "cyanobacterium": "cyanobacteria"
    }

    return mapping.get(x, x)

def normalize_type(entity, old_type):
    if entity == "dinoflagellate":
        return "DINO_TAXON"
    if entity == "cyanobacteria":
        return "CYANO_TAXON"
    return old_type

for filename in INPUT_FILES:

    infile = KG_DIR / filename

    if not infile.exists():
        print("Missing:", infile)
        continue

    df = pd.read_csv(infile).fillna("")

    for col in ["source", "target", "source_type", "target_type", "relation", "weight"]:
        if col not in df.columns:
            df[col] = ""

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1)

    # Normalize source and target
    df["source"] = df["source"].apply(normalize_entity)
    df["target"] = df["target"].apply(normalize_entity)

    df["source_type"] = df.apply(
        lambda r: normalize_type(r["source"], r["source_type"]),
        axis=1
    )

    df["target_type"] = df.apply(
        lambda r: normalize_type(r["target"], r["target_type"]),
        axis=1
    )

    # Remove self-loops created by normalization
    df = df[df["source"] != df["target"]].copy()

    # Re-aggregate duplicate edges created by merging
    group_cols = [
        "dataset",
        "source",
        "source_type",
        "relation",
        "target",
        "target_type"
    ]

    existing_group_cols = [c for c in group_cols if c in df.columns]

    agg_dict = {
        "weight": ("weight", "sum")
    }

    if "support_documents" in df.columns:
        agg_dict["support_documents"] = (
            "support_documents",
            lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";"))))
        )

    if "first_year" in df.columns:
        agg_dict["first_year"] = ("first_year", "min")

    if "last_year" in df.columns:
        agg_dict["last_year"] = ("last_year", "max")

    df2 = df.groupby(existing_group_cols, as_index=False).agg(**agg_dict)

    outfile = KG_DIR / filename.replace(".csv", "_taxa_normalized.csv")

    df2.to_csv(outfile, index=False, encoding="utf-8-sig")

    print("\nProcessed:", filename)
    print("Original edges:", len(df))
    print("Normalized edges:", len(df2))
    print("Saved:", outfile)
