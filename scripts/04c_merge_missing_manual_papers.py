import pandas as pd
import re
from pathlib import Path

# ===============================
# PATHS
# ===============================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

ENRICHED_CORPUS = BASE / "data/processed/stx_enriched_master_corpus.csv"

MISSING_MANUAL = BASE / "data/processed/manual_dinoflagellate_papers_missing.csv"

OUT_DIR = BASE / "data/processed"

FINAL_OUT = OUT_DIR / "stx_enriched_master_corpus_FINAL.csv"
ADDED_OUT = OUT_DIR / "manual_missing_papers_added_to_final_corpus.csv"
SUMMARY_OUT = OUT_DIR / "final_corpus_merge_summary.csv"

# ===============================
# FUNCTIONS
# ===============================

def read_csv_safe(path):
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc).fillna("")
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path, encoding="latin1", errors="replace").fillna("")

def normalize_title(x):
    x = str(x).lower()
    x = re.sub(r"<.*?>", " ", x)
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def clean_doi(x):
    x = str(x).lower().strip()
    x = x.replace("https://doi.org/", "")
    x = x.replace("http://dx.doi.org/", "")
    x = x.replace("doi:", "")
    return x

def standardize_columns(df):
    rename_map = {
        "Title": "title",
        "Abstract": "abstract",
        "Journal": "journal",
        "Year": "year",
        "Paper_Id": "paper_id",
        "Paper_ID": "paper_id",
        "Domain": "domain",
        "Group": "group",
        "Collection": "collection",
        "DOI": "doi",
        "PMID": "pmid",
        "URL": "url",
        "Url": "url"
    }

    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required = [
        "title", "abstract", "year", "doi", "pmid", "url",
        "journal", "paper_id", "domain", "group", "collection"
    ]

    for col in required:
        if col not in df.columns:
            df[col] = ""

    return df

def build_key(df):
    df["doi_clean"] = df["doi"].apply(clean_doi)
    df["pmid_clean"] = df["pmid"].astype(str).str.strip()
    df["title_clean"] = df["title"].apply(normalize_title)

    df["master_key"] = ""

    df.loc[df["doi_clean"] != "", "master_key"] = (
        "doi:" + df["doi_clean"]
    )

    df.loc[
        (df["master_key"] == "") &
        (df["pmid_clean"] != ""),
        "master_key"
    ] = "pmid:" + df["pmid_clean"]

    df.loc[
        (df["master_key"] == "") &
        (df["title_clean"] != ""),
        "master_key"
    ] = "title:" + df["title_clean"]

    return df

# ===============================
# LOAD
# ===============================

enriched_df = read_csv_safe(ENRICHED_CORPUS)
missing_df = read_csv_safe(MISSING_MANUAL)

enriched_df = standardize_columns(enriched_df)
missing_df = standardize_columns(missing_df)

enriched_df = build_key(enriched_df)
missing_df = build_key(missing_df)

existing_keys = set(enriched_df["master_key"])

# ===============================
# PREPARE MISSING PAPERS FOR MERGE
# ===============================

to_add = missing_df[
    ~missing_df["master_key"].isin(existing_keys)
].copy()

to_add = to_add[to_add["master_key"] != ""].copy()

# ===============================
# ADD REQUIRED PIPELINE COLUMNS
# ===============================

to_add["corpus_source"] = "manual_dinoflagellate_curation"
to_add["source_db"] = "manual"
to_add["query"] = "manual_curated_dinoflagellate_stx"

to_add["taxon_scope"] = "dinoflagellate"
to_add["relevance_class"] = "dinoflagellate_stx"
to_add["relevance_score"] = 999

to_add["stx_term_count"] = ""
to_add["gene_term_count"] = ""
to_add["dinoflagellate_term_count"] = ""
to_add["cyanobacteria_term_count"] = ""
to_add["environment_term_count"] = ""

to_add["matched_stx_terms"] = ""
to_add["matched_gene_terms"] = ""
to_add["matched_dinoflagellate_terms"] = ""
to_add["matched_cyanobacteria_terms"] = ""
to_add["matched_environment_terms"] = ""

to_add["text"] = (
    to_add["title"].astype(str).str.strip() + ". " +
    to_add["abstract"].astype(str).str.strip()
)

to_add["text"] = to_add["text"].str.replace(
    r"\s+", " ", regex=True
).str.strip()

# ===============================
# ALIGN COLUMNS
# ===============================

all_cols = list(enriched_df.columns)

for col in to_add.columns:
    if col not in all_cols:
        all_cols.append(col)

for col in all_cols:
    if col not in enriched_df.columns:
        enriched_df[col] = ""
    if col not in to_add.columns:
        to_add[col] = ""

enriched_df = enriched_df[all_cols]
to_add = to_add[all_cols]

# ===============================
# MERGE
# ===============================

final_df = pd.concat(
    [enriched_df, to_add],
    ignore_index=True
).fillna("")

final_df = final_df.drop_duplicates(
    "master_key",
    keep="first"
).copy()

# ===============================
# RECREATE DOCUMENT IDS
# ===============================

if "document_id" in final_df.columns:
    final_df = final_df.drop(columns=["document_id"])

final_df.insert(
    0,
    "document_id",
    [f"STX_DOC_{i+1:06d}" for i in range(len(final_df))]
)

# ===============================
# SAVE
# ===============================

final_df.to_csv(FINAL_OUT, index=False, encoding="utf-8-sig")
to_add.to_csv(ADDED_OUT, index=False, encoding="utf-8-sig")

summary = pd.DataFrame({
    "metric": [
        "original_enriched_records",
        "manual_missing_records_detected",
        "manual_missing_records_added",
        "final_corpus_records"
    ],
    "value": [
        len(enriched_df),
        len(missing_df),
        len(to_add),
        len(final_df)
    ]
})

summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

# ===============================
# PRINT SUMMARY
# ===============================

print("\n========== FINAL CORPUS MERGE ==========")
print("Original enriched records:", len(enriched_df))
print("Manual missing records detected:", len(missing_df))
print("Manual missing records added:", len(to_add))
print("Final corpus records:", len(final_df))

print("\nSaved final enriched corpus:")
print(FINAL_OUT)

print("\nSaved added manual papers:")
print(ADDED_OUT)

print("\nSaved merge summary:")
print(SUMMARY_OUT)
