import pandas as pd
import re
from pathlib import Path

# ===============================
# PATHS
# ===============================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

NEW_CORPUS = BASE / "data/processed/stx_enriched_master_corpus.csv"

# Your previous manually curated corpus folder
OLD_CORPUS = Path("/home/bhlabos/LBD/New/New/corpus_master.csv")

OUT_DIR = BASE / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MISSING_OUT = OUT_DIR / "manual_papers_missing_from_enriched_corpus.csv"
MATCHED_OUT = OUT_DIR / "manual_papers_recovered_in_enriched_corpus.csv"
SUMMARY_OUT = OUT_DIR / "manual_vs_enriched_corpus_comparison_summary.csv"

# ===============================
# FUNCTIONS
# ===============================

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
    return x

def standardize_columns(df):
    rename_map = {
        "Title": "title",
        "Abstract": "abstract",
        "Journal": "journal",
        "Year": "year",
        "Paper_Id": "paper_id",
        "Domain": "domain",
        "Group": "group",
        "Collection": "collection",
        "DOI": "doi",
        "doi": "doi",
        "PMID": "pmid",
        "pmid": "pmid",
        "URL": "url",
        "Url": "url",
        "url": "url"
    }

    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    for col in ["title", "abstract", "doi", "pmid", "year", "journal", "paper_id"]:
        if col not in df.columns:
            df[col] = ""

    return df

# ===============================
# LOAD FILES
# ===============================

old_df = pd.read_csv(OLD_CORPUS).fillna("")
new_df = pd.read_csv(NEW_CORPUS).fillna("")

old_df = standardize_columns(old_df)
new_df = standardize_columns(new_df)

# ===============================
# NORMALIZE IDENTIFIERS
# ===============================

old_df["doi_clean"] = old_df["doi"].apply(clean_doi)
new_df["doi_clean"] = new_df["doi"].apply(clean_doi)

old_df["title_clean"] = old_df["title"].apply(normalize_title)
new_df["title_clean"] = new_df["title"].apply(normalize_title)

# ===============================
# LOOKUP SETS
# ===============================

new_doi_set = set(new_df.loc[new_df["doi_clean"] != "", "doi_clean"])
new_title_set = set(new_df.loc[new_df["title_clean"] != "", "title_clean"])

# ===============================
# COMPARE
# ===============================

missing_rows = []
matched_rows = []

for _, row in old_df.iterrows():

    doi = row["doi_clean"]
    title = row["title_clean"]

    doi_match = doi != "" and doi in new_doi_set
    title_match = title != "" and title in new_title_set

    row_dict = row.to_dict()

    if doi_match:
        row_dict["match_type"] = "doi"
        matched_rows.append(row_dict)

    elif title_match:
        row_dict["match_type"] = "title"
        matched_rows.append(row_dict)

    else:
        row_dict["match_type"] = "not_found"
        missing_rows.append(row_dict)

missing_df = pd.DataFrame(missing_rows)
matched_df = pd.DataFrame(matched_rows)

# ===============================
# SAVE
# ===============================

missing_df.to_csv(MISSING_OUT, index=False, encoding="utf-8-sig")
matched_df.to_csv(MATCHED_OUT, index=False, encoding="utf-8-sig")

coverage = ((len(old_df) - len(missing_df)) / len(old_df)) * 100 if len(old_df) > 0 else 0

summary = pd.DataFrame({
    "metric": [
        "manual_corpus_records",
        "enriched_corpus_records",
        "manual_papers_recovered",
        "manual_papers_missing",
        "manual_corpus_recall_percent"
    ],
    "value": [
        len(old_df),
        len(new_df),
        len(matched_df),
        len(missing_df),
        round(coverage, 2)
    ]
})

summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

# ===============================
# PRINT SUMMARY
# ===============================

print("\n========== MANUAL VS ENRICHED CORPUS COMPARISON ==========")
print("Manual curated records:", len(old_df))
print("New enriched corpus records:", len(new_df))
print("Recovered manual papers:", len(matched_df))
print("Missing manual papers:", len(missing_df))
print(f"Manual corpus recall: {coverage:.2f}%")

print("\nSaved recovered papers:")
print(MATCHED_OUT)

print("\nSaved missing papers:")
print(MISSING_OUT)

print("\nSaved summary:")
print(SUMMARY_OUT)

if len(missing_df) > 0:
    print("\nTop missing papers:")
    cols = [c for c in ["paper_id", "title", "journal", "year", "domain", "collection"] if c in missing_df.columns]
    print(missing_df[cols].head(20).to_string(index=False))
