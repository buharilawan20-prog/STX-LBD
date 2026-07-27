import pandas as pd
import re
from pathlib import Path

# ===============================
# BASE DIRECTORY
# ===============================

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

IN_DIR = BASE / "data/processed"
OUT_DIR = BASE / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = [
    "combined_stx_multidatabase_raw_deduplicated.csv",
    "cyanobacteria_stx_multidatabase_raw_deduplicated.csv",
    "dinoflagellate_stx_multidatabase_raw_deduplicated.csv"
]

# ===============================
# KEYWORDS
# ===============================

STX_TERMS = [
    "saxitoxin",
    "saxitoxins",
    "paralytic shellfish toxin",
    "paralytic shellfish toxins",
    "paralytic shellfish poisoning",
    "pst",
    "psts",
    "stx",
    "neosaxitoxin",
    "gonyautoxin",
    "gonyautoxins",
    "decarbamoylsaxitoxin",
    "dcstx",
    "gtx",
    "gtx1",
    "gtx2",
    "gtx3",
    "gtx4"
]

GENE_TERMS = [
    "sxta",
    "sxta4",
    "sxtb",
    "sxtd",
    "sxtg",
    "sxth",
    "sxti",
    "sxts",
    "sxtu",
    "sxt gene",
    "sxt genes",
    "saxitoxin biosynthesis",
    "saxitoxin biosynthetic",
    "toxin biosynthesis"
]

DINO_TERMS = [
    "dinoflagellate",
    "dinoflagellates",
    "alexandrium",
    "gymnodinium",
    "pyrodinium",
    "gessnerium",
    "protoceratium",
    "alexandrium catenella",
    "alexandrium minutum",
    "alexandrium pacificum",
    "alexandrium tamarense",
    "alexandrium fundyense",
    "gymnodinium catenatum",
    "pyrodinium bahamense",
    "Centrodinium  punctatum",
    "alexandrium ostenfeldii",
]

CYANO_TERMS = [
    "cyanobacteria",
    "cyanobacterium",
    "cyanobacterial",
    "aphanizomenon",
    "dolichospermum",
    "anabaena",
    "cylindrospermopsis",
    "rhabdoderma",
    "lyngbya",
    "planktothrix",
    "scytonema"
]

ENV_TERMS = [
    "temperature",
    "salinity",
    "nitrogen",
    "phosphorus",
    "nutrient",
    "nitrate",
    "phosphate",
    "light",
    "irradiance",
    "climate",
    "warming",
    "bloom",
    "harmful algal bloom",
    "hab",
    "environmental",
    "stress"
]

# ===============================
# FUNCTIONS
# ===============================

def normalize_text(x):
    x = str(x).lower()
    x = re.sub(r"<.*?>", " ", x)
    x = re.sub(r"[^a-z0-9\s\-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def count_terms(text, terms):
    count = 0
    matched = []

    for term in terms:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, text):
            count += 1
            matched.append(term)

    return count, "; ".join(sorted(set(matched)))

def classify_record(row):
    title = normalize_text(row.get("title", ""))
    abstract = normalize_text(row.get("abstract", ""))
    text = title + " " + abstract

    stx_count, stx_hits = count_terms(text, STX_TERMS)
    gene_count, gene_hits = count_terms(text, GENE_TERMS)
    dino_count, dino_hits = count_terms(text, DINO_TERMS)
    cyano_count, cyano_hits = count_terms(text, CYANO_TERMS)
    env_count, env_hits = count_terms(text, ENV_TERMS)

    relevance_score = (
        stx_count * 3 +
        gene_count * 4 +
        dino_count * 2 +
        cyano_count * 2 +
        env_count
    )

    if stx_count >= 1 and gene_count >= 1 and (dino_count >= 1 or cyano_count >= 1):
        relevance_class = "high_confidence_stx_biosynthesis"
    elif stx_count >= 1 and dino_count >= 1:
        relevance_class = "dinoflagellate_stx"
    elif stx_count >= 1 and cyano_count >= 1:
        relevance_class = "cyanobacteria_stx"
    elif stx_count >= 1:
        relevance_class = "general_stx"
    else:
        relevance_class = "low_relevance_or_noise"

    if dino_count >= 1 and cyano_count >= 1:
        taxon_scope = "cross_taxa"
    elif dino_count >= 1:
        taxon_scope = "dinoflagellate"
    elif cyano_count >= 1:
        taxon_scope = "cyanobacteria"
    else:
        taxon_scope = "unspecified"

    return pd.Series({
        "relevance_score": relevance_score,
        "relevance_class": relevance_class,
        "taxon_scope": taxon_scope,
        "stx_term_count": stx_count,
        "gene_term_count": gene_count,
        "dinoflagellate_term_count": dino_count,
        "cyanobacteria_term_count": cyano_count,
        "environment_term_count": env_count,
        "matched_stx_terms": stx_hits,
        "matched_gene_terms": gene_hits,
        "matched_dinoflagellate_terms": dino_hits,
        "matched_cyanobacteria_terms": cyano_hits,
        "matched_environment_terms": env_hits
    })

# ===============================
# PROCESS FILES
# ===============================

for infile_name in INPUT_FILES:

    print("\n===================================")
    print("FILTERING:", infile_name)
    print("===================================")

    infile = IN_DIR / infile_name

    if not infile.exists():
        print("File not found:", infile)
        continue

    df = pd.read_csv(infile).fillna("")

    for col in ["title", "abstract", "year", "doi", "pmid", "url"]:
        if col not in df.columns:
            df[col] = ""

    results = df.apply(classify_record, axis=1)

    df = pd.concat([df, results], axis=1)

    df = df.sort_values(
        by=["relevance_score", "year"],
        ascending=[False, False]
    )

    stem = infile.stem.replace("_deduplicated", "")

    full_out = OUT_DIR / f"{stem}_relevance_scored.csv"
    high_out = OUT_DIR / f"{stem}_high_confidence.csv"
    dino_out = OUT_DIR / f"{stem}_dinoflagellate_relevant.csv"
    cyano_out = OUT_DIR / f"{stem}_cyanobacteria_relevant.csv"
    noise_out = OUT_DIR / f"{stem}_low_relevance_or_noise.csv"

    high_df = df[
        df["relevance_class"].isin([
            "high_confidence_stx_biosynthesis",
            "dinoflagellate_stx",
            "cyanobacteria_stx",
            "general_stx"
        ])
    ].copy()

    dino_df = df[
        df["taxon_scope"].isin(["dinoflagellate", "cross_taxa"])
    ].copy()

    cyano_df = df[
        df["taxon_scope"].isin(["cyanobacteria", "cross_taxa"])
    ].copy()

    noise_df = df[
        df["relevance_class"] == "low_relevance_or_noise"
    ].copy()

    df.to_csv(full_out, index=False, encoding="utf-8-sig")
    high_df.to_csv(high_out, index=False, encoding="utf-8-sig")
    dino_df.to_csv(dino_out, index=False, encoding="utf-8-sig")
    cyano_df.to_csv(cyano_out, index=False, encoding="utf-8-sig")
    noise_df.to_csv(noise_out, index=False, encoding="utf-8-sig")

    print("Total records:", len(df))
    print("High/general STX relevant:", len(high_df))
    print("Dinoflagellate relevant:", len(dino_df))
    print("Cyanobacteria relevant:", len(cyano_df))
    print("Noise/low relevance:", len(noise_df))

    print("\nRelevance classes:")
    print(df["relevance_class"].value_counts())

    print("\nTaxon scope:")
    print(df["taxon_scope"].value_counts())

    print("\nSaved:")
    print(full_out)
    print(high_out)
    print(dino_out)
    print(cyano_out)
    print(noise_out)

print("\nRelevance filtering completed.")
