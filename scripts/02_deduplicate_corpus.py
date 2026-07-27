import pandas as pd
import re
from pathlib import Path

# ===============================
# BASE DIRECTORY
# ===============================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

RAW_DIR = BASE / "data/raw"

OUT_DIR = BASE / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# INPUT FILES
# ===============================

INPUT_FILES = [
    "combined_stx_multidatabase_raw.csv",
    "cyanobacteria_stx_multidatabase_raw.csv",
    "dinoflagellate_stx_multidatabase_raw.csv"
]

# ===============================
# CLEANING FUNCTIONS
# ===============================

def clean_doi(x):
    x = str(x).lower().strip()
    x = x.replace("https://doi.org/", "")
    x = x.replace("http://dx.doi.org/", "")
    return x

def normalize_title(x):
    x = str(x).lower()
    x = re.sub(r"<.*?>", " ", x)
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

# ===============================
# PROCESS EACH RAW FILE
# ===============================

for infile_name in INPUT_FILES:

    print("\n===================================")
    print("PROCESSING:", infile_name)
    print("===================================")

    infile = RAW_DIR / infile_name

    if not infile.exists():
        print("File not found:", infile)
        continue

    df = pd.read_csv(infile).fillna("")

    required_cols = ["title", "doi", "pmid", "source_db"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # ===============================
    # NORMALIZE IDENTIFIERS
    # ===============================

    df["doi_clean"] = df["doi"].apply(clean_doi)
    df["pmid_clean"] = df["pmid"].astype(str).str.strip()
    df["title_clean"] = df["title"].apply(normalize_title)

    # ===============================
    # BUILD DEDUPLICATION KEY
    # Priority: DOI > PMID > normalized title
    # ===============================

    df["dedup_key"] = ""

    df.loc[df["doi_clean"] != "", "dedup_key"] = (
        "doi:" + df["doi_clean"]
    )

    df.loc[
        (df["dedup_key"] == "") &
        (df["pmid_clean"] != ""),
        "dedup_key"
    ] = "pmid:" + df["pmid_clean"]

    df.loc[
        (df["dedup_key"] == "") &
        (df["title_clean"] != ""),
        "dedup_key"
    ] = "title:" + df["title_clean"]

    df = df[df["dedup_key"] != ""].copy()

    # ===============================
    # SOURCE PRIORITY
    # Prefer PubMed metadata, then OpenAlex, then CrossRef
    # ===============================

    df["source_priority"] = df["source_db"].map({
        "PubMed": 1,
        "OpenAlex": 2,
        "CrossRef": 3
    }).fillna(9)

    df = df.sort_values(
        ["dedup_key", "source_priority"]
    )

    # ===============================
    # SPLIT DEDUPLICATED AND DUPLICATES
    # ===============================

    duplicates = df[
        df.duplicated("dedup_key", keep="first")
    ].copy()

    dedup = df.drop_duplicates(
        "dedup_key",
        keep="first"
    ).copy()

    dedup = dedup.drop(columns=["source_priority"])
    duplicates = duplicates.drop(columns=["source_priority"])

    # ===============================
    # SAVE OUTPUTS
    # ===============================

    stem = infile.stem

    dedup_file = OUT_DIR / f"{stem}_deduplicated.csv"
    dup_file = OUT_DIR / f"{stem}_duplicates_removed.csv"

    dedup.to_csv(
        dedup_file,
        index=False,
        encoding="utf-8-sig"
    )

    duplicates.to_csv(
        dup_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ===============================
    # SUMMARY
    # ===============================

    print("Input records:", len(df))
    print("Deduplicated records:", len(dedup))
    print("Duplicates removed:", len(duplicates))

    print("\nSaved:")
    print(dedup_file)

    print("\nDuplicate log:")
    print(dup_file)

    print("\nDatabase distribution after deduplication:")
    print(dedup["source_db"].value_counts())

print("\nAll deduplication completed.")
