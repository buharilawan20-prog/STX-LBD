import pandas as pd
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")
KG_DIR = BASE / "FINAL_WORKSPACE/kg"

PRE = KG_DIR / "dino_pre2016_semantic_edges_taxa_normalized.csv"
POST = KG_DIR / "dino_post2015_semantic_edges_taxa_normalized.csv"
OUT = KG_DIR / "dino_all_semantic_edges_taxa_normalized.csv"

pre = pd.read_csv(PRE).fillna("")
post = pd.read_csv(POST).fillna("")

df = pd.concat([pre, post], ignore_index=True)

df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1)

group_cols = [
    "source",
    "source_type",
    "relation",
    "target",
    "target_type"
]

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

df = df.groupby(group_cols, as_index=False).agg(**agg_dict)

df.insert(0, "dataset", "dino_all")

df.to_csv(OUT, index=False, encoding="utf-8-sig")

print("Saved:")
print(OUT)
print("Edges:", len(df))
