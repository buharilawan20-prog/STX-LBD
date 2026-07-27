import pandas as pd
import re
from pathlib import Path

BASE = Path("/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT")

IN_DIR = BASE / "FINAL_WORKSPACE/splits"
OUT_DIR = BASE / "FINAL_WORKSPACE/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = {
    "dino_pre2016": "dino_pre2016.csv",
    "dino_post2015": "dino_post2015.csv",
    "cyano_all": "cyano_all.csv"
}

ENTITY_PATTERNS = {
    "TOXIN": [
        "saxitoxin", "stx", "neosaxitoxin", "gonyautoxin",
        "gtx", "paralytic shellfish toxin", "paralytic shellfish toxins",
        "pst", "psts", "paralytic shellfish poisoning"
    ],

    "SXT_GENE": [
        "sxta", "sxta4", "sxtb", "sxtd", "sxtg", "sxth",
        "sxti", "sxts", "sxtu", "sxt gene", "sxt genes",
        "saxitoxin biosynthesis gene", "saxitoxin biosynthesis genes"
    ],

    "DINO_TAXON": [

    "dinoflagellate",
    "dinoflagellates",

    "alexandrium",
    "alexandrium catenella",
    "alexandrium minutum",
    "alexandrium pacificum",
    "alexandrium tamarense",
    "alexandrium fundyense",
    "alexandrium ostenfeldii",
    "alexandrium affine",
    "alexandrium australiense",
    "alexandrium hiranoi",
    "alexandrium leei",
    "alexandrium tamiyavanichii",

    "gymnodinium",
    "gymnodinium catenatum",
    "gymnodinium smaydae",
    "gymnodinium impudicum",

    "pyrodinium",
    "pyrodinium bahamense",

    "centrodinium",
    "centrodinium punctatum",
    "c punctatum",

    "gessnerium",
    "gessnerium catenatum",

    "protoceratium",
    "protoceratium reticulatum"
    ],

    "CYANO_TAXON": [
        "cyanobacteria", "cyanobacterium", "cyanobacterial",
        "aphanizomenon", "dolichospermum", "anabaena",
        "cylindrospermopsis", "rhabdoderma", "lyngbya",
        "planktothrix", "scytonema"
    ],

    "ENV_FACTOR": [
        "temperature", "salinity", "nitrogen", "phosphorus",
        "nitrate", "phosphate", "nutrient", "nutrients",
        "light", "irradiance", "warming", "climate",
        "environmental stress", "bloom", "harmful algal bloom"
    ],

    "BIOLOGICAL_PROCESS": [
        "biosynthesis", "toxin biosynthesis", "saxitoxin biosynthesis",
        "regulation", "expression", "gene expression",
        "transcription", "transcriptome", "transcriptomic",
        "evolution", "phylogeny", "phylogenetic",
        "horizontal gene transfer", "gene loss", "gene duplication",
        "functional divergence", "toxicity", "toxin production"
    ],

    "DETECTION_METHOD": [
        "hplc", "lc-ms", "lc-ms/ms", "mass spectrometry",
        "mouse bioassay", "elisa", "biosensor", "aptasensor",
        "toxin profiling", "chemical analysis"
    ]
}

def normalize_text(x):
    x = str(x).lower()
    x = re.sub(r"<.*?>", " ", x)
    x = re.sub(r"[^a-z0-9\s\-\/]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def find_entities(text):
    entities = []

    norm_text = normalize_text(text)

    for entity_type, terms in ENTITY_PATTERNS.items():
        for term in terms:
            term_norm = normalize_text(term)
            pattern = r"\b" + re.escape(term_norm) + r"\b"

            if re.search(pattern, norm_text):
                entities.append({
                    "entity": term_norm,
                    "entity_type": entity_type
                })

    return entities

all_entity_files = []

for label, filename in INPUT_FILES.items():

    print("\n===================================")
    print("EXTRACTING ENTITIES:", filename)
    print("===================================")

    infile = IN_DIR / filename

    if not infile.exists():
        print("Missing:", infile)
        continue

    df = pd.read_csv(infile).fillna("")

    for col in ["document_id", "title", "abstract", "text", "year", "taxon_scope", "relevance_class"]:
        if col not in df.columns:
            df[col] = ""

    rows = []

    for _, row in df.iterrows():

        doc_id = row["document_id"]
        title = row["title"]
        abstract = row["abstract"]

        text = row["text"] if str(row["text"]).strip() else f"{title}. {abstract}"

        entities = find_entities(text)

        for ent in entities:
            rows.append({
                "dataset": label,
                "document_id": doc_id,
                "year": row["year"],
                "taxon_scope": row["taxon_scope"],
                "relevance_class": row["relevance_class"],
                "entity": ent["entity"],
                "entity_type": ent["entity_type"],
                "title": title
            })

    ent_df = pd.DataFrame(rows)

    out_file = OUT_DIR / f"{label}_entities.csv"
    ent_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    all_entity_files.append(ent_df)

    print("Documents:", len(df))
    print("Entity mentions:", len(ent_df))

    if len(ent_df) > 0:
        print("\nEntity types:")
        print(ent_df["entity_type"].value_counts())

        print("\nTop entities:")
        print(ent_df["entity"].value_counts().head(20))

if all_entity_files:
    combined = pd.concat(all_entity_files, ignore_index=True)

    combined_out = OUT_DIR / "all_split_entities_combined.csv"
    combined.to_csv(combined_out, index=False, encoding="utf-8-sig")

    print("\nSaved combined entity file:")
    print(combined_out)

print("\nEntity extraction completed.")
