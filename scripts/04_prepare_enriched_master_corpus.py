import pandas as pd
from pathlib import Path
import re

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

IN_DIR = BASE / "data/processed"
OUT_DIR = BASE / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = {
    "combined": "combined_stx_multidatabase_raw_high_confidence.csv",
    "dinoflagellate": "dinoflagellate_stx_multidatabase_raw_high_confidence.csv",
    "cyanobacteria": "cyanobacteria_stx_multidatabase_raw_high_confidence.csv"
}

OUTFILE = OUT_DIR / "stx_enriched_master_corpus.csv"
SUMMARY_FILE = OUT_DIR / "stx_enriched_master_corpus_summary.csv"

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

all_dfs = []

for corpus_label, filename in INPUT_FILES.items():
    path = IN_DIR / filename

    if not path.exists():
        print("Missing:", path)
        continue

    df = pd.read_csv(path).fillna("")
    df["corpus_source"] = corpus_label
    all_dfs.append(df)

if not all_dfs:
    raise FileNotFoundError("No input files found.")

df = pd.concat(all_dfs, ignore_index=True).fillna("")

for col in ["title", "abstract", "year", "doi", "pmid", "url"]:
    if col not in df.columns:
        df[col] = ""

df["doi_clean"] = df["doi"].apply(clean_doi)
df["title_clean"] = df["title"].apply(normalize_title)
df["pmid_clean"] = df["pmid"].astype(str).str.strip()

df["master_key"] = ""

df.loc[df["doi_clean"] != "", "master_key"] = "doi:" + df["doi_clean"]

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

df = df[df["master_key"] != ""].copy()

df["corpus_source"] = df.groupby("master_key")["corpus_source"].transform(
    lambda x: ";".join(sorted(set(x)))
)

df = df.sort_values(
    by=["relevance_score", "year"],
    ascending=[False, False]
)

df = df.drop_duplicates("master_key", keep="first").copy()

df["text"] = (
    df["title"].astype(str).str.strip() + ". " +
    df["abstract"].astype(str).str.strip()
)

df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

df["document_id"] = [
    f"STX_DOC_{i+1:06d}" for i in range(len(df))
]

preferred_cols = [
    "document_id",
    "corpus_source",
    "title",
    "abstract",
    "text",
    "year",
    "doi",
    "pmid",
    "url",
    "source_db",
    "query",
    "relevance_score",
    "relevance_class",
    "taxon_scope",
    "stx_term_count",
    "gene_term_count",
    "dinoflagellate_term_count",
    "cyanobacteria_term_count",
    "environment_term_count",
    "matched_stx_terms",
    "matched_gene_terms",
    "matched_dinoflagellate_terms",
    "matched_cyanobacteria_terms",
    "matched_environment_terms",
    "master_key"
]

existing_cols = [c for c in preferred_cols if c in df.columns]
extra_cols = [c for c in df.columns if c not in existing_cols]

df = df[existing_cols + extra_cols]

df.to_csv(OUTFILE, index=False, encoding="utf-8-sig")

summary = pd.DataFrame({
    "metric": [
        "total_master_records",
        "dinoflagellate_scope",
        "cyanobacteria_scope",
        "cross_taxa_scope",
        "high_confidence_biosynthesis",
        "general_stx",
        "dinoflagellate_stx",
        "cyanobacteria_stx"
    ],
    "count": [
        len(df),
        (df["taxon_scope"] == "dinoflagellate").sum(),
        (df["taxon_scope"] == "cyanobacteria").sum(),
        (df["taxon_scope"] == "cross_taxa").sum(),
        (df["relevance_class"] == "high_confidence_stx_biosynthesis").sum(),
        (df["relevance_class"] == "general_stx").sum(),
        (df["relevance_class"] == "dinoflagellate_stx").sum(),
        (df["relevance_class"] == "cyanobacteria_stx").sum()
    ]
})

summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

print("\nSaved master corpus:")
print(OUTFILE)

print("\nSaved summary:")
print(SUMMARY_FILE)

print("\nMaster corpus records:", len(df))

print("\nTaxon scope:")
print(df["taxon_scope"].value_counts())

print("\nRelevance class:")
print(df["relevance_class"].value_counts())

print("\nCorpus source:")
print(df["corpus_source"].value_counts())
