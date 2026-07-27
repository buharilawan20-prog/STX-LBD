import pandas as pd
from pathlib import Path

# ===============================
# PATHS
# ===============================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

INPUT = BASE / "data/processed/stx_enriched_master_corpus_FINAL.csv"

OUT_DIR = BASE / "FINAL_WORKSPACE/splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# LOAD
# ===============================

df = pd.read_csv(INPUT).fillna("")

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

# ===============================
# DINO ONLY
# ===============================

dino_df = df[
    df["taxon_scope"].isin([
        "dinoflagellate",
        "cross_taxa"
    ])
].copy()

# ===============================
# CYANO ONLY
# ===============================

cyano_df = df[
    df["taxon_scope"].isin([
        "cyanobacteria",
        "cross_taxa"
    ])
].copy()

# ===============================
# TEMPORAL SPLITS
# ===============================

dino_pre2016 = dino_df[
    dino_df["year"] <= 2015
].copy()

dino_post2015 = dino_df[
    dino_df["year"] >= 2016
].copy()

# ===============================
# SAVE
# ===============================

dino_pre2016.to_csv(
    OUT_DIR / "dino_pre2016.csv",
    index=False,
    encoding="utf-8-sig"
)

dino_post2015.to_csv(
    OUT_DIR / "dino_post2015.csv",
    index=False,
    encoding="utf-8-sig"
)

cyano_df.to_csv(
    OUT_DIR / "cyano_all.csv",
    index=False,
    encoding="utf-8-sig"
)

# ===============================
# SUMMARY
# ===============================

print("\n========== TEMPORAL SPLITS ==========")

print("Dinoflagellate pre-2016:", len(dino_pre2016))

print("Dinoflagellate post-2015:", len(dino_post2015))

print("Cyanobacteria total:", len(cyano_df))

print("\nSaved files:")
print("dino_pre2016.csv")
print("dino_post2015.csv")
print("cyano_all.csv")
